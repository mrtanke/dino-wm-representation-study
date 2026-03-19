#!/usr/bin/env bash

set -euo pipefail

MAMBA_EXE="${HOME}/.local/bin/micromamba"
REPO_DIR="/mnt/c/Users/zack/Documents/GNN3"
CKPT_BASE="${REPO_DIR}/course_runs"

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
SEED="${SEED:-0}"
PLOT_ROLLOUTS="${PLOT_ROLLOUTS:-False}"
SAVE_VIDEO="${SAVE_VIDEO:-False}"
N_PLOT_SAMPLES="${N_PLOT_SAMPLES:-0}"

cd "${REPO_DIR}"

run_eval() {
  local label="$1"
  local model_name="$2"
  echo "==== ${label} (${model_name}) ===="
  "${MAMBA_EXE}" run -n dino_wm python plan.py \
    --config-name plan_point_maze_wsl.yaml \
    ckpt_base_path="${CKPT_BASE}" \
    model_name="${model_name}" \
    seed="${SEED}" \
    n_evals="${N_EVALS}" \
    n_plot_samples="${N_PLOT_SAMPLES}" \
    plot_rollouts="${PLOT_ROLLOUTS}" \
    save_video="${SAVE_VIDEO}" \
    planner.sub_planner.num_samples="${PLAN_SAMPLES}" \
    planner.sub_planner.topk="${PLAN_TOPK}" \
    planner.sub_planner.opt_steps="${PLAN_OPT_STEPS}" \
    planner.n_taken_actions=1
}

run_eval "patch_det_large_eval" "2026-03-18/19-12-06"
run_eval "cls_det_large_eval" "2026-03-18/20-01-01"
run_eval "patch_gauss_large_eval" "2026-03-18/20-20-39"
run_eval "cls_gauss_large_eval" "2026-03-18/20-45-03"
