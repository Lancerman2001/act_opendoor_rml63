#!/usr/bin/env python3
"""Copy the validated dataset and create a deterministic, episode-level 80/20 split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil


PROJECT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(args):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats

    source = args.source.resolve()
    output = args.output.resolve()
    target = args.dataset_root.resolve() if args.dataset_root else output / "full"
    split_file = args.split_file.resolve() if args.split_file else output / "split.json"
    output = split_file.parent
    require(0 < args.train_ratio < 1, 'train_ratio must be between 0 and 1')
    if split_file.exists():
        from dataset_utils import validate_split
        existing = read_json(split_file)
        validate_split(existing)
        require(existing["seed"] == args.seed, "Existing split uses another seed; choose another output directory")
        require(abs(existing.get('train_ratio', .8) - args.train_ratio) < 1e-12, 'Existing split uses another ratio')
        require(existing['repo_id'] == args.repo_id, 'Existing split uses another repo_id')
        require((split_file.parent / existing['dataset_root']).resolve() == target, 'Existing split points to another dataset')
        if (source / 'meta/info.json').is_file():
            require(existing["source_info_sha256"] == sha256(source / "meta/info.json"), "Existing split source differs")
        require(target.is_dir() and (target / "meta/stats.json").is_file(), "Existing prepared dataset is incomplete")
        require(sha256(target / "meta/stats.json") == existing["train_stats_sha256"], "Prepared normalization statistics changed")
        for relative, digest in existing.get('payload_sha256', {}).items():
            require(sha256(target / relative) == digest, f'Prepared payload changed: {relative}')
            if source.is_dir():
                require(sha256(source / relative) == digest, f'Source payload changed: {relative}; prepare a new split/output')
        print(json.dumps({"status": "already_prepared", "train_episodes": existing['train_episode_count'], "test_episodes": existing['test_episode_count'],
                          "split": str(split_file)}, ensure_ascii=False), flush=True)
        return
    require((source / 'meta/info.json').is_file(), f'No LeRobot v3 dataset at {source}; set dataset.source in the configuration')
    info = read_json(source / "meta/info.json")
    require(info["codebase_version"] == "v3.0", "Source must be LeRobotDataset v3.0")
    require(info['total_episodes'] >= 2 and info['fps'] > 0, 'At least two episodes and a positive fps are required')
    require(source != target and source not in target.parents, 'Dataset destination must be outside the source directory')
    require(not target.exists(), f"Dataset destination exists without split manifest: {target}")
    staging = target.with_name('.' + target.name + '.preparing')
    require(not staging.exists(), f"Unfinished preparation exists: {staging}")
    output.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)

    metadata_files = sorted((source / "meta/episodes").rglob("*.parquet"))
    episodes = sorted([row for path in metadata_files for row in pq.read_table(path).to_pylist()], key=lambda ep: ep['episode_index'])
    require([ep["episode_index"] for ep in episodes] == list(range(info["total_episodes"])),
            "Source episode indices are not contiguous")
    source_manifest = read_json(source / 'source_manifest.json') if (source / 'source_manifest.json').is_file() else episodes
    require(len(source_manifest) == len(episodes), 'Source manifest episode count mismatch')
    require(all(ep['episode_index'] == original['episode_index'] for ep, original in zip(episodes, source_manifest)),
            'Source manifest order mismatch')
    order = list(range(len(episodes)))
    random.Random(args.seed).shuffle(order)
    n_train = max(1, min(len(order) - 1, int(len(order) * args.train_ratio)))
    train_indices, test_indices = sorted(order[:n_train]), sorted(order[n_train:])
    require(not set(train_indices) & set(test_indices), "Train/test episodes overlap")
    require(set(train_indices) | set(test_indices) == set(range(len(episodes))), "Split omits source episodes")

    # Aggregate only selected training episode statistics. Image statistics are also
    # train-only here; the default trainer subsequently uses fixed ImageNet visual stats.
    per_episode_stats = []
    for ep in (episodes[i] for i in train_indices):
        selected = {}
        for key, value in ep.items():
            if key.startswith("stats/"):
                _, feature, statistic = key.split("/", 2)
                selected.setdefault(feature, {})[statistic] = np.asarray(value)
        require(selected, f"Episode {ep['episode_index']} lacks statistics")
        per_episode_stats.append(selected)
    stats = aggregate_stats(per_episode_stats)

    data_tables = [pq.read_table(path) for path in sorted((source / "data").rglob("*.parquet"))]
    data = pa.concat_tables(data_tables)
    require(data.num_rows == info["total_frames"], "Source frame count mismatch")
    indices = data["episode_index"].to_numpy()
    train_mask = np.isin(indices, train_indices)
    test_mask = np.isin(indices, test_indices)
    require(bool(np.all(train_mask ^ test_mask)), "Every frame must belong to exactly one split")
    train_frames = int(train_mask.sum())
    test_frames = int(test_mask.sum())

    # Recompute numeric normalizer quantities from training rows, independently of
    # recorded per-episode summaries; never include held-out frames in normalization.
    for key in ("observation.state", "action"):
        values = np.asarray(data[key].to_pylist(), dtype=np.float64)[train_mask]
        require(values.ndim == 2 and values.shape[0] == train_frames and bool(np.isfinite(values).all()), f"Invalid vector feature {key}")
        for operation in ("min", "max", "mean", "std"):
            expected = getattr(np, operation)(values, axis=0)
            require(np.allclose(stats[key][operation], expected, atol=1e-4, rtol=1e-4),
                    f"Per-episode aggregation disagrees with actual training rows: {key}/{operation}")
            stats[key][operation] = expected
        stats[key]["count"] = np.asarray([train_frames], dtype=np.int64)
    serial_stats = {key: {name: np.asarray(value).tolist() for name, value in entry.items()}
                    for key, entry in stats.items()}

    print(f"Copying validated dataset into {target} (videos are copied without re-encoding)", flush=True)
    shutil.copytree(source, staging, ignore=shutil.ignore_patterns(".cache"))
    shutil.copyfile(staging / "meta/stats.json", staging / "meta/stats_full_dataset.json")
    write_json(staging / "meta/stats.json", serial_stats)
    (staging / "PROJECT_DATASET_NOTE.txt").write_text(
        "This is the full backing storage for both logical train/test selections.\n"
        "Always select episode indices from the configured split manifest. Dataset meta/info.json still describes all source episodes.\n"
        "meta/stats.json contains TRAIN-ONLY normalization statistics; the original complete-dataset values are\n"
        "preserved in meta/stats_full_dataset.json and must not be used for training normalization.\n"
        "The held-out test episodes are never used for batch tuning or checkpoint selection.\n",
        encoding="utf-8")

    # Verify the model input payload did not change during the independent project copy.
    payload_hashes = {}
    for folder in ("data", "videos"):
        for path in sorted((source / folder).rglob("*")):
            if path.is_file():
                relative = path.relative_to(source).as_posix()
                digest = sha256(path)
                require(sha256(staging / relative) == digest, f"Copied file differs: {relative}")
                payload_hashes[relative] = digest
    staging.rename(target)
    write_json(output / "train_stats.json", serial_stats)
    entries = []
    train_set = set(train_indices)
    for ep, original in zip(episodes, source_manifest):
        require(ep["episode_index"] == original["episode_index"], "Source manifest order mismatch")
        entries.append({"episode_index": ep["episode_index"],
                        "split": "train" if ep["episode_index"] in train_set else "test",
                        "frames": ep["length"], "global_from": ep["dataset_from_index"],
                        "global_to": ep["dataset_to_index"], "source_session": original.get("source_session"),
                        "source_episode": original.get("source_episode")})
    split = {
        "status": "prepared", "seed": args.seed,
        "method": "Python random.Random(seed).shuffle episode indices; floor(N * train_ratio), clamped to nonempty splits",
        "split_unit": "whole_episode", "train_ratio": args.train_ratio,
        "dataset_root": os.path.relpath(target, output), "repo_id": args.repo_id, "total_episodes": len(episodes),
        "total_frames": data.num_rows, "train_episode_indices": train_indices,
        "test_episode_indices": test_indices, "train_episode_count": len(train_indices),
        "test_episode_count": len(test_indices), "train_frames": train_frames, "test_frames": test_frames,
        "episode_overlap": [], "normalization_scope": "training episodes only",
        "test_usage": "Final evaluation only; not batch tuning, early stopping or checkpoint selection",
        "source_root_at_preparation": str(source), "source_info_sha256": sha256(source / "meta/info.json"),
        "source_stats_sha256": sha256(source / "meta/stats.json"),
        "train_stats_sha256": sha256(target / "meta/stats.json"), "payload_sha256": payload_hashes,
        "episodes": entries, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(split_file, split)
    for name, ep_indices, n_frames in (("train", train_indices, train_frames), ("test", test_indices, test_frames)):
        write_json(output / f"{name}.json", {"dataset_root": os.path.relpath(target, output), "repo_id": args.repo_id,
                                           "split": name, "episode_indices": ep_indices, "frames": n_frames,
                                           "normalization_scope": "train"})
    print(json.dumps({"status": "prepared", "train_episodes": len(train_indices),
                      "test_episodes": len(test_indices), "train_frames": train_frames,
                      "test_frames": test_frames, "split": str(split_file)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--split-file', type=Path)
    parser.add_argument('--repo-id')
    parser.add_argument('--train-ratio', type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    from project_config import load_config, resolve_path
    cfg = load_config(args.config)['dataset']
    args.source = args.source or resolve_path(cfg['source'])
    if args.output is None:
        args.dataset_root = args.dataset_root or resolve_path(cfg['root'])
        args.split_file = args.split_file or resolve_path(cfg['split_file'])
        args.output = args.split_file.parent
    args.repo_id = args.repo_id or cfg['repo_id']
    args.train_ratio = cfg['train_ratio'] if args.train_ratio is None else args.train_ratio
    args.seed = cfg['seed'] if args.seed is None else args.seed
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    prepare(args)


if __name__ == "__main__":
    main()
