#!/usr/bin/env python3
"""One isolated, actual ACT forward/backward benchmark; invoked by benchmark_batch.py."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import traceback


GIB = 1024**3


class ResourceSampler:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.samples = []
        self.phase = 'initialization'
        self.stop_event = threading.Event()
        self.nvml = None
        self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nvml = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def read(self):
        import psutil
        item = {"monotonic_s": time.monotonic(), "phase": self.phase,
                "rss_bytes": psutil.Process().memory_info().rss,
                "host_available_bytes": psutil.virtual_memory().available}
        if self.nvml is not None:
            try:
                memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                item.update(gpu_used_bytes=memory.used, gpu_free_bytes=memory.free,
                            gpu_total_bytes=memory.total)
            except Exception:
                pass
            # WSL/laptop drivers may expose only some NVML fields. One unsupported
            # getter must not discard the other measurements or stop a benchmark.
            getters = {
                'gpu_utilization_percent': lambda: self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu,
                'gpu_temperature_c': lambda: self.nvml.nvmlDeviceGetTemperature(self.handle, self.nvml.NVML_TEMPERATURE_GPU),
                'gpu_power_w': lambda: self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000,
                'gpu_enforced_power_limit_w': lambda: self.nvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000,
                'gpu_sm_clock_mhz': lambda: self.nvml.nvmlDeviceGetClockInfo(self.handle, self.nvml.NVML_CLOCK_SM),
                'gpu_pstate': lambda: f'P{int(self.nvml.nvmlDeviceGetPerformanceState(self.handle))}',
            }
            for key, getter in getters.items():
                try:
                    item[key] = getter()
                except Exception:
                    pass
            if 'gpu_total_bytes' in item and all(key in item for key in getters):
                return item
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw,enforced.power.limit,clocks.sm,pstate",
                 "--format=csv,noheader,nounits"], text=True, timeout=3)
            fields = ('gpu_used_bytes', 'gpu_free_bytes', 'gpu_total_bytes',
                      'gpu_utilization_percent', 'gpu_temperature_c', 'gpu_power_w',
                      'gpu_enforced_power_limit_w', 'gpu_sm_clock_mhz', 'gpu_pstate')
            for key, raw in zip(fields, output.splitlines()[0].split(',')):
                if key in item:
                    continue
                raw = raw.strip()
                try:
                    if key == 'gpu_pstate':
                        if raw.startswith('P') and raw[1:].isdigit():
                            item[key] = raw
                    else:
                        value = float(raw)
                        if math.isfinite(value):
                            item[key] = int(value*1024**2) if key.endswith('_bytes') else value
                except ValueError:
                    pass
        except Exception:
            pass
        return item

    def _run(self):
        while not self.stop_event.is_set():
            self.samples.append(self.read())
            self.stop_event.wait(self.interval)

    def start(self):
        self.samples.append(self.read())
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        self.samples.append(self.read())
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass


def benchmark(args, report):
    import random
    import numpy as np
    import psutil
    import torch
    from accelerate import Accelerator
    from importlib.metadata import version
    from torch.utils.data import default_collate
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    if version("lerobot") != "0.7.0":
        raise RuntimeError("Benchmark requires lerobot==0.7.0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support the selected BF16 configuration")
    if psutil.virtual_memory().available < args.min_host_available_gib * GIB:
        report['status'] = 'host_memory_pressure'
        return
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)
    split = json.loads(args.split.read_text())
    episodes = split['train_episode_indices']
    test_episodes = split['test_episode_indices']
    from dataset_utils import validate_split
    validate_split(split)
    stats_sha256 = hashlib.sha256((args.dataset_root/'meta/stats.json').read_bytes()).hexdigest()
    if stats_sha256 != split['train_stats_sha256']:
        raise ValueError('Dataset normalization statistics do not match the fixed training-only statistics')
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=args.repo_id, root=str(args.dataset_root), episodes=episodes,
                              eval_split=0.0, video_backend=args.video_backend),
        policy=ACTConfig(device='cuda', push_to_hub=False, chunk_size=args.chunk_size,
                         n_action_steps=args.chunk_size, optimizer_lr=args.learning_rate,
                         optimizer_lr_backbone=args.learning_rate_backbone,
                         kl_weight=args.kl_weight, vision_backbone=args.vision_backbone,
                         pretrained_backbone_weights=args.pretrained_backbone_weights),
        batch_size=args.batch_size, steps=args.warmup+args.steps,
        seed=args.seed, num_workers=0, env_eval_freq=0,
    )
    cfg.optimizer = cfg.policy.get_optimizer_preset()
    cfg.scheduler = cfg.policy.get_scheduler_preset()
    before = time.monotonic()
    dataset = make_dataset(cfg)
    if len(dataset) != split['train_frames']:
        raise ValueError('Loaded dataset frame count does not match the training split')
    if len(dataset) < args.batch_size:
        raise ValueError('Training dataset has fewer examples than the requested benchmark batch')
    sample_pool_size = min(max(args.sample_pool_size, args.batch_size), len(dataset))
    indices = np.random.default_rng(args.seed).choice(len(dataset), size=sample_pool_size, replace=False)
    # Every candidate uses a prefix of this same deterministic training-only pool.
    batch_indices = indices[:args.batch_size].tolist()
    raw_batch = default_collate([dataset[int(index)] for index in batch_indices])
    camera_keys = dataset.meta.camera_keys
    if not camera_keys or any(raw_batch[key].ndim != 4 or list(raw_batch[key].shape[:2]) != [args.batch_size, 3] for key in camera_keys):
        raise ValueError('Expected RGB camera batches with shape [batch,3,height,width]')
    action_dim = dataset.meta.features['action']['shape'][0]
    state_dim = dataset.meta.features['observation.state']['shape'][0]
    if list(raw_batch['action'].shape) != [args.batch_size,args.chunk_size,action_dim]:
        raise ValueError('Unexpected action chunk shape')
    if list(raw_batch['observation.state'].shape) != [args.batch_size,state_dim]:
        raise ValueError('Unexpected robot state shape')
    sampled_episode_ids = [int(value) for value in raw_batch['episode_index'].reshape(-1).tolist()]
    if not set(sampled_episode_ids).issubset(episodes):
        raise RuntimeError("Benchmark unexpectedly sampled a held-out episode")
    decode_seconds = time.monotonic() - before
    report.update({
        'versions': {name: version(name) for name in ('lerobot', 'torch', 'torchvision', 'accelerate')},
        'gpu': torch.cuda.get_device_name(0), 'cuda_runtime': torch.version.cuda,
        'model': 'ACT', 'mixed_precision': 'bf16', 'chunk_size': args.chunk_size,
        'vision_backbone': cfg.policy.vision_backbone,
        'pretrained_backbone_weights': cfg.policy.pretrained_backbone_weights,
        'image_shape': list(raw_batch[camera_keys[0]].shape[1:]),
        'image_shapes': {key: list(raw_batch[key].shape[1:]) for key in camera_keys},
        'action_dim': action_dim, 'state_dim': state_dim,
        'optimizer': 'AdamW', 'learning_rate': args.learning_rate,
        'learning_rate_backbone': args.learning_rate_backbone, 'kl_weight': args.kl_weight,
        'optimizer_weight_decay': cfg.optimizer.weight_decay,
        'gradient_clip_norm': cfg.optimizer.grad_clip_norm,
        'train_stats_sha256': stats_sha256,
        'train_episode_count': len(episodes), 'held_out_episode_count': len(test_episodes),
        'sampled_dataset_indices': batch_indices, 'sampled_episode_indices': sampled_episode_ids,
        'data_decode_seconds': decode_seconds,
        'measurement_scope': 'Cached, normalized real training batch: forward+backward+gradient clipping+optimizer. Data decoding and preprocessing excluded; actual supervised run measures end-to-end throughput.',
    })
    sampler = ResourceSampler()
    sampler.start()
    baseline = sampler.read()
    report['gpu_baseline'] = baseline
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        # Decoder work must not change model initialization across candidates.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        policy = make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor, _ = make_pre_post_processors(cfg.policy, dataset_stats=dataset.meta.stats)
        optimizer, _ = make_optimizer_and_scheduler(cfg, policy)
        accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
        policy, optimizer = accelerator.prepare(policy, optimizer)
        started = time.monotonic()
        batch = _preprocess_dataset_batch(raw_batch, dataset.meta.camera_keys, {}, preprocessor)
        torch.cuda.synchronize()
        report['preprocessing_seconds'] = time.monotonic() - started
        report['parameter_count'] = sum(param.numel() for param in policy.parameters())
        policy.train()
        sampler.phase = 'warmup'
        losses, timings, gradients, updates = [], [], [], 0
        warmup_peak_allocated = warmup_peak_reserved = 0
        for step in range(args.warmup + args.steps):
            if psutil.virtual_memory().available < args.min_host_available_gib * GIB:
                report['status'] = 'host_memory_pressure'
                break
            if step == args.warmup:
                sampler.phase = 'timed'
                warmup_peak_allocated = torch.cuda.max_memory_allocated()
                warmup_peak_reserved = torch.cuda.max_memory_reserved()
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with accelerator.autocast():
                loss, metrics = policy(batch)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f'Nonfinite loss at step {step}: {loss.item()}')
            accelerator.backward(loss)
            gradient_norm = accelerator.clip_grad_norm_(policy.parameters(), cfg.optimizer.grad_clip_norm)
            if not bool(torch.isfinite(gradient_norm).item()):
                raise FloatingPointError(f'Nonfinite gradient norm at step {step}')
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if accelerator.optimizer_step_was_skipped:
                raise FloatingPointError('Optimizer update was skipped')
            updates += 1
            if step >= args.warmup:
                timings.append(elapsed)
                losses.append(float(loss.item()))
                gradients.append(float(gradient_norm.item()))
            print(f"batch={args.batch_size} step={step+1}/{args.warmup+args.steps} "
                  f"loss={loss.item():.5f} seconds={elapsed:.4f}", flush=True)
        sampler.phase = 'finalizing'
        report.update({
            'optimizer_updates': updates, 'timed_steps_completed': len(timings),
            'warmup_peak_allocated_bytes': warmup_peak_allocated,
            'warmup_peak_reserved_bytes': warmup_peak_reserved,
            'steady_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'steady_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'peak_allocated_bytes': max(warmup_peak_allocated, torch.cuda.max_memory_allocated()),
            'peak_reserved_bytes': max(warmup_peak_reserved, torch.cuda.max_memory_reserved()),
            'step_seconds': timings, 'losses': losses, 'gradient_norms': gradients,
        })
        if timings:
            report.update(median_step_seconds=statistics.median(timings),
                          p90_step_seconds=float(np.percentile(timings,90)),
                          samples_per_second=args.batch_size/statistics.median(timings),
                          first_timed_loss=losses[0], last_timed_loss=losses[-1])
        if report.get('status') != 'host_memory_pressure':
            if len(timings) != args.steps:
                raise RuntimeError('Timed optimizer updates were incomplete')
            report['status'] = 'passed'
    finally:
        sampler.close()
        samples = sampler.samples
        report['resource_samples'] = samples
        timed_samples = [sample for sample in samples if sample.get('phase') == 'timed']
        timed_summary = {'sample_count': len(timed_samples)}
        for key in ('gpu_power_w', 'gpu_enforced_power_limit_w', 'gpu_sm_clock_mhz',
                    'gpu_temperature_c', 'gpu_utilization_percent'):
            values = [sample[key] for sample in timed_samples if key in sample]
            if values:
                timed_summary[key] = {'min': min(values), 'median': statistics.median(values),
                                      'max': max(values), 'sample_count': len(values)}
        timed_summary['gpu_pstates_observed'] = sorted({sample['gpu_pstate'] for sample in timed_samples
                                                       if 'gpu_pstate' in sample})
        report['timed_resource_summary'] = timed_summary
        for source_key, result_key, aggregate in (
            ('gpu_used_bytes','observed_peak_gpu_used_bytes',max),
            ('gpu_free_bytes','observed_min_gpu_free_bytes',min),
            ('gpu_temperature_c','max_gpu_temperature_c',max),
            ('gpu_utilization_percent','max_gpu_utilization_percent',max),
            ('rss_bytes','observed_peak_rss_bytes',max),
            ('host_available_bytes','observed_min_host_available_bytes',min),
        ):
            values = [sample[source_key] for sample in samples if source_key in sample]
            if values:
                report[result_key] = aggregate(values)
        total = baseline.get('gpu_total_bytes', torch.cuda.get_device_properties(0).total_memory)
        baseline_used = baseline.get('gpu_used_bytes',0)
        # Conservative allowance also covers peaks that 100 ms polling could miss.
        estimated = max(report.get('observed_peak_gpu_used_bytes',0),
                        baseline_used + report.get('peak_reserved_bytes',torch.cuda.max_memory_reserved()))
        report['estimated_peak_board_used_bytes'] = estimated
        report['estimated_board_headroom_bytes'] = total - estimated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument('--dataset-root', type=Path, default=project/'data/full')
    parser.add_argument('--split', type=Path, default=project/'data/split.json')
    parser.add_argument('--repo-id', default='local/open_door')
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--sample-pool-size', type=int, default=32)
    parser.add_argument('--chunk-size', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--learning-rate-backbone', type=float, default=1e-5)
    parser.add_argument('--kl-weight', type=float, default=10)
    parser.add_argument('--vision-backbone', default='resnet18')
    parser.add_argument('--pretrained-backbone-weights', default='ResNet18_Weights.IMAGENET1K_V1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--video-backend', default='torchcodec')
    parser.add_argument('--min-host-available-gib', type=float, default=2.0)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.pretrained_backbone_weights.lower() == 'none':
        args.pretrained_backbone_weights = None
    if min(args.batch_size,args.steps,args.sample_pool_size,args.chunk_size,args.cpu_threads) < 1 or args.warmup < 1:
        parser.error('Batch/step/pool/chunk/thread/warmup values must be positive')
    report = {'status':'running','batch_size':args.batch_size,'warmup_steps':args.warmup,
              'timed_steps':args.steps,'seed':args.seed,'started_at_utc':datetime.now(timezone.utc).isoformat()}
    start = time.monotonic()
    try:
        benchmark(args, report)
        exit_code = 0 if report['status']=='passed' else 2
    except Exception as exc:
        message = str(exc)
        report['status'] = 'oom' if 'out of memory' in message.lower() else 'failed'
        report['error'] = {'type':type(exc).__name__,'message':message,'traceback':traceback.format_exc()}
        print(f"BENCHMARK {report['status']}: {exc}", flush=True)
        exit_code = 3 if report['status']=='oom' else 1
    report['elapsed_seconds'] = round(time.monotonic()-start,3)
    report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','batch_size','median_step_seconds','samples_per_second','estimated_board_headroom_bytes')}) ,flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
