#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <run_dir> <config_name> [hydra_override ...]"
  exit 2
fi

RUN_DIR="$1"
CONFIG_NAME="$2"
shift 2

MAMBA_EXE="${HOME}/.local/bin/micromamba"
REPO_DIR="/mnt/c/Users/zack/Documents/GNN3"

export DATASET_DIR="${HOME}/dino_wm_data"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export MUJOCO_GL="egl"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export WANDB_MODE=offline

cd "${RUN_DIR}"

"${MAMBA_EXE}" run -n dino_wm python "${REPO_DIR}/train.py" \
  --config-name "${CONFIG_NAME}" \
  hydra.run.dir=. \
  hydra.output_subdir=null \
  "$@"
