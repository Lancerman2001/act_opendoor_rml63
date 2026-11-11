#!/usr/bin/env python3
"""Compare real ACT BF16 batch sizes in isolated subprocesses; never use test episodes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


GIB = 1024**3


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--split', type=Path)
    parser.add_argument('--repo-id')
    parser.add_argument('--batches', nargs='+', type=int)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--chunk-size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--video-backend')
    parser.add_argument('--headroom-gib', type=float)
    parser.add_argument('--min-host-available-gib', type=float, default=2.0)
    parser.add_argument('--candidate-timeout', type=int, default=900)
    parser.add_argument('--report-dir', type=Path)
    args = parser.parse_args()
    from project_config import load_config, resolve_path
    cfg = load_config(args.config)
    for key in ('batches', 'warmup', 'steps', 'headroom_gib'):
        if getattr(args, key) is None:
            setattr(args, key, cfg['benchmark'][key])
    args.dataset_root = args.dataset_root or resolve_path(cfg['dataset']['root'])
    args.split = args.split or resolve_path(cfg['dataset']['split_file'])
    args.report_dir = args.report_dir or resolve_path(cfg['benchmark']['report_dir'])
    args.repo_id = args.repo_id or cfg['dataset']['repo_id']
    args.video_backend = args.video_backend or cfg['dataset']['video_backend']
    args.chunk_size = cfg['policy']['chunk_size'] if args.chunk_size is None else args.chunk_size
    args.seed = cfg['training']['seed'] if args.seed is None else args.seed
    if cfg['training']['mixed_precision'] != 'bf16':
        parser.error('This benchmark measures BF16; choose BF16 or train with an explicit batch size')
    candidates = sorted(set(args.batches))
    if (not candidates or min(candidates) < 1 or args.steps < 1 or args.warmup < 1
            or args.cpu_threads < 1 or args.candidate_timeout < 1 or args.chunk_size < 1):
        parser.error('Batch, warmup, and step counts must be positive')
    if args.report_dir.exists() and any(args.report_dir.iterdir()):
        parser.error(f'Report directory already contains results: {args.report_dir}; choose a new --report-dir')
    args.report_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault('TORCH_HOME',str(project/'.cache/torch'))
    env.setdefault('HF_HOME',str(project/'.cache/huggingface'))
    env.setdefault('OMP_NUM_THREADS',str(args.cpu_threads))
    env.setdefault('MKL_NUM_THREADS',str(args.cpu_threads))
    env['HF_HUB_OFFLINE']='1'
    env['HF_DATASETS_OFFLINE']='1'
    report = {'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
              'candidates':candidates,'warmup_steps':args.warmup,'timed_steps':args.steps,
              'headroom_target_bytes':int(args.headroom_gib*GIB),'results':[],
              'measurement_scope':'Compute-only actual ACT optimizer steps on normalized cached real train examples. Decode/preprocess time is reported separately; actual training measures end-to-end throughput.',
              'split':str(args.split.resolve()),'dataset_root':str(args.dataset_root.resolve()),
              'test_data_used':False}
    report_path = args.report_dir/'summary.json'
    write_report(report_path,report)
    worker = Path(__file__).with_name('benchmark_worker.py')
    start = time.monotonic()
    for batch in candidates:
        child_report = args.report_dir/f'batch_{batch}.json'
        command = [sys.executable,'-u',str(worker),'--dataset-root',str(args.dataset_root),
                   '--split',str(args.split),'--repo-id',args.repo_id,'--batch-size',str(batch),
                   '--warmup',str(args.warmup),'--steps',str(args.steps),
                   '--sample-pool-size',str(max(candidates)), '--chunk-size',str(args.chunk_size),
                   '--seed',str(args.seed),'--cpu-threads',str(args.cpu_threads),
                   '--learning-rate',str(cfg['policy']['optimizer_lr']),
                   '--learning-rate-backbone',str(cfg['policy']['optimizer_lr_backbone']),
                   '--kl-weight',str(cfg['policy']['kl_weight']),
                   '--vision-backbone',cfg['policy']['vision_backbone'],
                   '--pretrained-backbone-weights',str(cfg['policy']['pretrained_backbone_weights']),
                   '--video-backend',args.video_backend,'--min-host-available-gib',str(args.min_host_available_gib),
                   '--report',str(child_report)]
        print(f'Benchmarking batch {batch}; details: {args.report_dir/f"batch_{batch}.log"}',flush=True)
        with (args.report_dir/f'batch_{batch}.log').open('w') as log:
            try:
                completed = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                           timeout=args.candidate_timeout)
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
        if child_report.exists():
            result = json.loads(child_report.read_text())
        else:
            result = {'batch_size':batch,'status':'timeout' if returncode==124 else 'failed',
                      'error':{'message':'Worker did not produce a report; inspect its log'}}
        result['returncode'] = returncode
        if result['status'] == 'passed' and returncode != 0:
            result['status'] = 'failed'
            result['error'] = {'message':'Worker reported success but exited abnormally; inspect its log'}
        report['results'].append(result)
        write_report(report_path,report)
        print(json.dumps({k:result.get(k) for k in ('batch_size','status','samples_per_second','estimated_board_headroom_bytes')}) ,flush=True)
        if result['status'] in ('oom','host_memory_pressure'):
            report['stopped_after'] = {'batch_size':batch,'reason':result['status']}
            break
        if result['status'] != 'passed':
            report['status']='failed'
            report['stopped_after']={'batch_size':batch,'reason':result['status']}
            break
    passed = [r for r in report['results'] if r['status']=='passed']
    safe = [r for r in passed if r.get('estimated_board_headroom_bytes',-1) >= args.headroom_gib*GIB
            and r.get('observed_min_host_available_bytes',0) >= args.min_host_available_gib*GIB]
    if safe:
        best = max(r['samples_per_second'] for r in safe)
        # Prefer a smaller batch when throughput differs by at most five percent.
        selected = min((r for r in safe if r['samples_per_second'] >= 0.95*best),key=lambda r:r['batch_size'])
        report['recommended_batch_size']=selected['batch_size']
        report['recommendation_reason']='Smallest measured batch within 5% of best safe compute throughput, preserving configured whole-GPU and host-RAM headroom; validate with the supervised end-to-end run.'
    else:
        report['recommended_batch_size']=None
        report['recommendation_reason']='No candidate passed the requested memory headroom; inspect failures or benchmark a smaller batch.'
    if report['status'] != 'failed':
        report['status']='completed' if passed else 'failed'
    report['elapsed_seconds']=round(time.monotonic()-start,3)
    report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    write_report(report_path,report)
    print(json.dumps({'status':report['status'],'recommended_batch_size':report['recommended_batch_size'],
                      'report':str(report_path.resolve())}),flush=True)
    return 0 if report['status']=='completed' and report['recommended_batch_size'] is not None else 1


if __name__=='__main__':
    raise SystemExit(main())
