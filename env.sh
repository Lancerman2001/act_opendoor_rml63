#!/usr/bin/env bash
# Source this file; all project caches and model assets stay inside this project.
export ACT_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# ROS/Conda shells can inject unrelated packages into a virtual environment.
# Keep this project's Python imports inside its own environment.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PATH="$ACT_PROJECT_ROOT/.venv/bin:$ACT_PROJECT_ROOT/tools/ffmpeg/usr/bin:$PATH"
export HF_HOME="$ACT_PROJECT_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$ACT_PROJECT_ROOT/.cache/torch"
export XDG_CACHE_HOME="$ACT_PROJECT_ROOT/.cache"
export MPLCONFIGDIR="$ACT_PROJECT_ROOT/.cache/matplotlib"
export MPLBACKEND=Agg
export UV_CACHE_DIR="$ACT_PROJECT_ROOT/.cache/uv"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NO_COLOR=1
