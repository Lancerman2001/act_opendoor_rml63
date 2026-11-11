#!/usr/bin/env python3
"""Evaluate the fixed final ACT checkpoint on the held-out episode split."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


PROJECT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(args, report):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TORCH_HOME", str(PROJECT / ".cache/torch"))
    os.environ.setdefault("HF_HOME", str(PROJECT / ".cache/huggingface"))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    torch.set_num_threads(4)
    if args.device == "cuda":
        require(torch.cuda.is_available(), "CUDA is not available")
        if args.precision == "bf16":
            require(torch.cuda.is_bf16_supported(), "CUDA device does not support bfloat16")
    require(args.samples_per_episode > 0 and args.batch_size > 0, "Sampling and batch sizes must be positive")
    split = read_json(args.split_file)
    from dataset_utils import validate_split
    validate_split(split)
    root = args.dataset_root or args.split_file.parent / split["dataset_root"]
    root = root.resolve()
    info = read_json(root / 'meta/info.json')
    action_dim = info['features']['action']['shape'][0]
    require(info['features']['observation.state']['shape'] == [action_dim],
            'Hold-state baseline requires matching state/action vector dimensions; disable automatic evaluation for other representations')
    require(sha256(root / "meta/stats.json") == split["train_stats_sha256"], "Training normalization stats changed")
    checkpoint = args.checkpoint.resolve()
    require((checkpoint / "model.safetensors").is_file(), "Pass the checkpoint's pretrained_model directory")
    config = ACTConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = args.device
    # A complete checkpoint already contains backbone weights. Avoid downloading
    # ImageNet weights just to overwrite them when reloading the trained model.
    config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(checkpoint, config=config, local_files_only=True, strict=True)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device},
                                "normalizer_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    train_stats = read_json(args.split_file.parent / "train_stats.json")
    for pipeline, expected in ((preprocessor, {"observation.state", "action"}),
                               (postprocessor, {"action"})):
        checked = set()
        for step in pipeline.steps:
            tensor_stats = getattr(step, "_tensor_stats", {})
            for key in ("observation.state", "action"):
                if key not in tensor_stats:
                    continue
                for name in ("mean", "std"):
                    saved = tensor_stats[key][name].detach().cpu().numpy()
                    require(np.allclose(saved, train_stats[key][name], rtol=1e-6, atol=1e-6),
                            f"Checkpoint {key}/{name} does not match training-only statistics")
                checked.add(key)
        require(expected.issubset(checked), f"Checkpoint processor lacks normalization stats: {expected - checked}")

    selected_episodes = split[f"{args.split}_episode_indices"]
    dataset = LeRobotDataset(
        repo_id=split["repo_id"], root=root, episodes=selected_episodes,
        delta_timestamps={"action": [i / info['fps'] for i in range(config.chunk_size)]},
        video_backend=args.video_backend, return_uint8=False,
    )
    episode_records = {entry["episode_index"]: entry for entry in split["episodes"]}
    sample_indices, absolute_indices = [], []
    relative_map = dataset.absolute_to_relative_idx
    for episode_id in selected_episodes:
        record = episode_records[episode_id]
        n = record["frames"]
        count = n if args.all_frames else min(n, args.samples_per_episode)
        absolute = np.linspace(record["global_from"], record["global_to"] - 1, count, dtype=np.int64)
        require(len(set(absolute.tolist())) == count, "Balanced sample indices repeat")
        absolute_indices.extend(int(index) for index in absolute)
        sample_indices.extend(relative_map[int(index)] for index in absolute)
    kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                  pin_memory=args.device == "cuda")
    if args.num_workers:
        kwargs.update(multiprocessing_context="spawn", persistent_workers=False, prefetch_factor=2)
    loader = DataLoader(Subset(dataset, sample_indices), **kwargs)
    dimension_names = info['features']['action'].get('names') or [f'action_{i}' for i in range(action_dim)]
    require(isinstance(dimension_names, list) and len(dimension_names) == action_dim and len(set(dimension_names)) == action_dim,
            'Action feature names must uniquely name every vector dimension')
    totals = {name: np.zeros(action_dim, dtype=np.float64)
              for name in ("first", "chunk", "hold_first", "hold_chunk")}
    sample_count = valid_count = 0
    normalized_absolute_error = 0.0
    per_episode = {ep: {"sample_count": 0, "valid_chunk_action_count": 0,
                       "first_error_sum": np.zeros(action_dim, dtype=np.float64),
                       "hold_first_error_sum": np.zeros(action_dim, dtype=np.float64)} for ep in selected_episodes}
    report.update({"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
                   "split": args.split, "split_file": str(args.split_file.resolve()),
                   "episode_indices": selected_episodes, "source_split_seed": split["seed"],
                   "device": args.device, "precision": args.precision, "video_backend": args.video_backend,
                   "chunk_size": config.chunk_size, "batch_size": args.batch_size,
                   "dataset_split_frames": len(dataset), "planned_samples": len(sample_indices),
                   "sampling": "all_frames" if args.all_frames else "evenly_spaced_within_every_episode_including_ends",
                   "samples_per_episode_limit": None if args.all_frames else args.samples_per_episode,
                   "sample_global_indices": absolute_indices,
                   "normalization_scope": "saved checkpoint processors, checked against training-only stats",
                   "inference_latent": "zero (ACT eval mode), no VAE target conditioning",
                   'action_dimension': action_dim,
                   "metric_units": {name: "degrees" if name.endswith('_deg') else 'dataset_native_units'
                                    for name in dimension_names},
                   "limitations": ["Offline action prediction metrics do not establish real-robot success.",
                                   "Initial-training test evaluation must not be used to tune this checkpoint."]})
    started = time.monotonic()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch_id, batch in enumerate(loader):
            raw_target = batch["action"].clone()
            raw_state = batch["observation.state"].clone()
            valid = (~batch["action_is_pad"]).numpy()
            episode_ids = batch["episode_index"].numpy().astype(int)
            processed = preprocessor(batch)
            autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                        if args.device == "cuda" and args.precision == "bf16" else nullcontext())
            with autocast:
                normalized_prediction = policy.predict_action_chunk(processed)
            require(bool(torch.isfinite(normalized_prediction).all()), "Nonfinite predicted actions")
            prediction = postprocessor(normalized_prediction.float()).detach().cpu().numpy()
            target = raw_target.numpy()
            hold_prediction = np.broadcast_to(raw_state.numpy()[:, None, :], target.shape)
            errors = np.abs(prediction - target)
            hold_errors = np.abs(hold_prediction - target)
            normalized_error = (normalized_prediction.float() - processed["action"].float()).abs()
            normalized_absolute_error += float(
                (normalized_error * processed["action_is_pad"].logical_not().unsqueeze(-1)).sum().item())
            totals["first"] += errors[:, 0].sum(axis=0)
            totals["chunk"] += (errors * valid[:, :, None]).sum(axis=(0, 1))
            totals["hold_first"] += hold_errors[:, 0].sum(axis=0)
            totals["hold_chunk"] += (hold_errors * valid[:, :, None]).sum(axis=(0, 1))
            sample_count += len(episode_ids)
            valid_count += int(valid.sum())
            for i, episode_id in enumerate(episode_ids):
                accumulator = per_episode[int(episode_id)]
                accumulator["sample_count"] += 1
                accumulator["valid_chunk_action_count"] += int(valid[i].sum())
                accumulator["first_error_sum"] += errors[i, 0]
                accumulator["hold_first_error_sum"] += hold_errors[i, 0]
            if (batch_id + 1) % 20 == 0 or sample_count == len(sample_indices):
                print(f"Evaluation {args.split}: {sample_count}/{len(sample_indices)} samples; "
                      f"normalized L1={normalized_absolute_error / (valid_count * action_dim):.6f}", flush=True)
    require(sample_count == len(sample_indices) and valid_count > 0, "Incomplete evaluation")
    report["sample_count"] = sample_count
    report["evaluated_episode_count"] = len(selected_episodes)
    report["valid_chunk_action_count"] = valid_count
    report["padding_excluded"] = True
    report["normalized_masked_l1"] = normalized_absolute_error / (valid_count * action_dim)
    for key, denominator in (("first", sample_count), ("chunk", valid_count),
                             ("hold_first", sample_count), ("hold_chunk", valid_count)):
        values = totals[key] / denominator
        report[f"{key}_action_mae"] = {name: float(value) for name, value in zip(dimension_names, values)}
        if dimension_names == [f'joint_{i}_deg' for i in range(1, 7)] + ['gripper_raw']:
            report[f"{key}_six_joint_mean_mae_deg"] = float(values[:6].mean())
            report[f"{key}_gripper_mae_raw"] = float(values[-1])
    report["episode_metrics"] = []
    for episode_id in selected_episodes:
        acc = per_episode[episode_id]
        n = acc["sample_count"]
        require(n > 0, f"Episode {episode_id} was not evaluated")
        report["episode_metrics"].append({
            "episode_index": episode_id, "samples": n,
            "valid_chunk_action_count": acc["valid_chunk_action_count"],
            "first_action_mae": dict(zip(dimension_names, (acc["first_error_sum"] / n).tolist())),
            "hold_first_action_mae": dict(zip(dimension_names, (acc["hold_first_error_sum"] / n).tolist())),
        })
    report["elapsed_evaluation_seconds"] = round(time.monotonic() - started, 3)
    if args.device == "cuda":
        report["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, help="pretrained_model directory")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--samples-per-episode", type=int)
    parser.add_argument("--all-frames", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", choices=("cuda", "cuda:0", "cpu"))
    parser.add_argument("--precision", choices=("bf16", "fp32"))
    parser.add_argument("--video-backend", choices=("torchcodec", "pyav"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    from project_config import load_config, resolve_path
    cfg = load_config(args.config)
    args.split_file = args.split_file or resolve_path(cfg['dataset']['split_file'])
    args.dataset_root = args.dataset_root or resolve_path(cfg['dataset']['root'])
    for key in ('samples_per_episode', 'all_frames', 'batch_size', 'num_workers'):
        if getattr(args, key) is None:
            setattr(args, key, cfg['evaluation'][key])
    args.device = args.device or cfg['policy']['device']
    if args.device == 'cuda:0':
        args.device = 'cuda'
    args.precision = args.precision or ('bf16' if cfg['training']['mixed_precision'] == 'bf16' else 'fp32')
    args.video_backend = args.video_backend or cfg['dataset']['video_backend']
    args.report = args.report or (args.checkpoint.resolve().parent / 'test_evaluation.json')
    report = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        evaluate(args, report)
        report["status"] = "passed"
        code = 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(f"Evaluation failed: {exc}", flush=True)
        code = 1
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("status", "split", "sample_count", "normalized_masked_l1",
                                                     "first_six_joint_mean_mae_deg", "first_gripper_mae_raw")}
                     | {"report": str(args.report.resolve())}, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
