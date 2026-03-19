#!/usr/bin/env bash

set -u -o pipefail

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

run_eval() {
  local label="$1"
  local model_name="$2"
  local plan_seed="$3"
  local done_flag="${STATUS_DIR}/${label}.done"
  local fail_flag="${STATUS_DIR}/${label}.fail"

  if [[ -f "${done_flag}" ]]; then
    echo "==== skip ${label} (${model_name}, seed=${plan_seed}) [done] ===="
    return 0
  fi

  rm -f "${fail_flag}"
  echo "==== start ${label} (${model_name}, seed=${plan_seed}) @ $(date '+%F %T') ===="

  if "${MAMBA_EXE}" run -n dino_wm python plan.py \
    --config-name plan_point_maze_wsl.yaml \
    ckpt_base_path="${CKPT_BASE}" \
    model_name="${model_name}" \
    seed="${plan_seed}" \
    n_evals="${N_EVALS}" \
    n_plot_samples="${N_PLOT_SAMPLES}" \
    plot_rollouts="${PLOT_ROLLOUTS}" \
    save_video="${SAVE_VIDEO}" \
    planner.sub_planner.num_samples="${PLAN_SAMPLES}" \
    planner.sub_planner.topk="${PLAN_TOPK}" \
    planner.sub_planner.opt_steps="${PLAN_OPT_STEPS}" \
    planner.n_taken_actions=1; then
    touch "${done_flag}"
    echo "==== done ${label} @ $(date '+%F %T') ===="
  else
    touch "${fail_flag}"
    echo "==== fail ${label} @ $(date '+%F %T') ===="
  fi
}

run_eval "seed0_patch_det" "2026-03-18/19-12-06" "0"
run_eval "seed0_cls_det" "2026-03-18/20-01-01" "0"
run_eval "seed0_patch_gauss" "2026-03-18/20-20-39" "0"
run_eval "seed0_cls_gauss" "2026-03-18/20-45-03" "0"

run_eval "seed1_patch_det" "2026-03-18/21-12-22" "1"
run_eval "seed1_cls_det" "2026-03-18/21-50-26" "1"
run_eval "seed1_patch_gauss" "2026-03-18/22-09-15" "1"
run_eval "seed1_cls_gauss" "2026-03-18/22-33-51" "1"

run_eval "seed2_patch_det" "2026-03-18/22-54-50" "2"
run_eval "seed2_cls_det" "2026-03-18/23-32-43" "2"
run_eval "seed2_patch_gauss" "2026-03-18/23-51-26" "2"
run_eval "seed2_cls_gauss" "2026-03-19/00-16-03" "2"

echo "==== large eval matrix batch completed @ $(date '+%F %T') ===="
