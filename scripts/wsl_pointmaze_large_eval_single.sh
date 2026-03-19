#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <label> <model_name> <seed>"
  exit 2
fi

LABEL="$1"
MODEL_NAME="$2"
SEED="$3"

MAMBA_EXE="${HOME}/.local/bin/micromamba"
REPO_DIR="/mnt/c/Users/zack/Documents/GNN3"
CKPT_BASE="${REPO_DIR}/course_runs"
STATUS_DIR="${REPO_DIR}/logs/large_eval_status"

export DATASET_DIR="${HOME}/dino_wm_data"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export MUJOCO_GL="egl"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export WANDB_MODE=offline

N_EVALS="${N_EVALS:-10}"
PLAN_SAMPLES="${PLAN_SAMPLES:-16}"
PLAN_TOPK="${PLAN_TOPK:-4}"
PLAN_OPT_STEPS="${PLAN_OPT_STEPS:-2}"
PLOT_ROLLOUTS="${PLOT_ROLLOUTS:-False}"
SAVE_VIDEO="${SAVE_VIDEO:-False}"
N_PLOT_SAMPLES="${N_PLOT_SAMPLES:-0}"

mkdir -p "${STATUS_DIR}"
cd "${REPO_DIR}"

DONE_FLAG="${STATUS_DIR}/${LABEL}.done"
FAIL_FLAG="${STATUS_DIR}/${LABEL}.fail"

rm -f "${FAIL_FLAG}"

"${MAMBA_EXE}" run -n dino_wm python plan.py \
  --config-name plan_point_maze_wsl.yaml \
  ckpt_base_path="${CKPT_BASE}" \
  model_name="${MODEL_NAME}" \
  seed="${SEED}" \
  n_evals="${N_EVALS}" \
  n_plot_samples="${N_PLOT_SAMPLES}" \
  plot_rollouts="${PLOT_ROLLOUTS}" \
  save_video="${SAVE_VIDEO}" \
  planner.sub_planner.num_samples="${PLAN_SAMPLES}" \
  planner.sub_planner.topk="${PLAN_TOPK}" \
  planner.sub_planner.opt_steps="${PLAN_OPT_STEPS}" \
  planner.n_taken_actions=1

touch "${DONE_FLAG}"
