#!/usr/bin/env bash

set -euo pipefail

MAMBA_EXE="${HOME}/.local/bin/micromamba"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"

"${MAMBA_EXE}" run -n dino_wm python -c 'import sys; print(sys.version)'
"${MAMBA_EXE}" run -n dino_wm python -c 'import hydra, omegaconf; print("hydra ok")'
"${MAMBA_EXE}" run -n dino_wm python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())'
"${MAMBA_EXE}" run -n dino_wm python -c 'import mujoco_py; print("mujoco_py ok")'
