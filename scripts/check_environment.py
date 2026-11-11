#!/usr/bin/env python3
"""Validate configured CUDA precision, video decoding and headless plotting."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import ctypes
import ctypes.util
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FFMPEG_HELP = (
    'Install FFmpeg AND its shared libraries. On Ubuntu/Debian: '
    'sudo apt-get update && sudo apt-get install -y ffmpeg. '
    'A standalone ffmpeg executable alone is insufficient for TorchCodec.'
)


def check_ffmpeg_libraries():
    libraries = {}
    for name in ('avcodec', 'avformat', 'avutil', 'swresample', 'swscale'):
        soname = ctypes.util.find_library(name)
        if not soname:
            raise RuntimeError(f'Missing shared library {name}. {FFMPEG_HELP}')
        try:
            ctypes.CDLL(soname)
        except OSError as exc:
            raise RuntimeError(f'Cannot load {soname}. {FFMPEG_HELP}') from exc
        libraries[name] = soname
    return libraries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'data/full',
                        help='Optional dataset for an additional video decode check')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/environment.json')
    parser.add_argument('--precision', choices=('bf16', 'fp16', 'no'), default='bf16',
                        help='Training mixed precision to validate; no means float32')
    args = parser.parse_args()
    import torch
    import torchvision  # noqa: F401 -- validate the compiled extension import
    import av  # noqa: F401 -- validate the installed video package
    import matplotlib

    if version('lerobot') != '0.7.0':
        raise RuntimeError('This project targets LeRobot 0.7.0')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check the NVIDIA driver/GPU PyTorch installation; CPU fallback is disabled.')
    bf16_supported = torch.cuda.is_bf16_supported()
    if args.precision == 'bf16' and not bf16_supported:
        raise RuntimeError('This CUDA device does not support the configured BF16 training mode')
    x = torch.randn(128, 128, device='cuda')
    precision_context = (nullcontext() if args.precision == 'no' else
                         torch.autocast('cuda', dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float16))
    with precision_context:
        y = x @ x
    if not bool(torch.isfinite(y).all()):
        raise RuntimeError(f'CUDA/{args.precision} matrix multiplication produced nonfinite values')
    torch.cuda.synchronize()
    prop = torch.cuda.get_device_properties(0)
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError(f'ffmpeg was not found on PATH. {FFMPEG_HELP}')
    libraries = check_ffmpeg_libraries()
    try:
        ffmpeg_version = subprocess.check_output([ffmpeg, '-version'], text=True, stderr=subprocess.STDOUT).splitlines()[0]
        from torchcodec.decoders import VideoDecoder
    except (subprocess.SubprocessError, ImportError, RuntimeError) as exc:
        raise RuntimeError(f'FFmpeg/TorchCodec could not initialize. {FFMPEG_HELP}\n{exc}') from exc

    report = {
        'status': 'passed', 'python': platform.python_version(),
        'packages': {p: version(p) for p in ('lerobot', 'torch', 'torchvision', 'torchcodec', 'av', 'accelerate', 'matplotlib')},
        'cuda_runtime': torch.version.cuda, 'gpu': prop.name,
        'compute_capability': f'{prop.major}.{prop.minor}',
        'gpu_total_bytes': prop.total_memory, 'bf16_supported': bf16_supported,
        'precision_tested': args.precision, 'cuda_operation': 'passed', 'ffmpeg': ffmpeg,
        'ffmpeg_version': ffmpeg_version, 'ffmpeg_shared_libraries': libraries,
    }
    # Validate decode even on a new clone with no private dataset attached.
    with tempfile.TemporaryDirectory(prefix='act-environment-') as temporary:
        clip = Path(temporary) / 'decode-smoke.mkv'
        subprocess.run([ffmpeg, '-nostdin', '-loglevel', 'error', '-f', 'lavfi', '-i',
                        'color=c=red:s=64x48:r=5', '-frames:v', '2', '-c:v', 'ffv1', str(clip)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        frame = VideoDecoder(str(clip), device='cpu')[0]
        if tuple(frame.shape) != (3, 48, 64):
            raise RuntimeError(f'Unexpected synthetic video shape: {tuple(frame.shape)}')
        report['synthetic_video_decode_shape'] = list(frame.shape)
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        figure, axis = plt.subplots(figsize=(2, 1))
        axis.plot([0, 1], [1, 0])
        figure.savefig(Path(temporary) / 'plot-smoke.png')
        plt.close(figure)
        report['matplotlib_agg_render'] = 'passed'

    videos = sorted((args.dataset_root.expanduser() / 'videos').rglob('*.mp4'))
    report['dataset_video_checked'] = bool(videos)
    if videos:
        frame = VideoDecoder(str(videos[0]), device='cpu')[0]
        if frame.ndim != 3 or frame.shape[0] != 3 or min(frame.shape[1:]) <= 0:
            raise RuntimeError(f'Unexpected dataset video shape: {tuple(frame.shape)}')
        report['video_decode_shape'] = list(frame.shape)
    report['nvidia_smi'] = subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu',
        '--format=csv,noheader'], text=True).strip()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
