#!/usr/bin/env bash

set -euo pipefail

MAMBA_EXE="${HOME}/.local/bin/micromamba"
REPO_DIR="/mnt/c/Users/zack/Documents/GNN3"
OUT_BASE="${REPO_DIR}/course_runs"

export DATASET_DIR="${HOME}/dino_wm_data"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export MUJOCO_GL="egl"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export WANDB_MODE=offline

cd "${REPO_DIR}"

EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_ROLLOUT="${N_ROLLOUT:-64}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEEDS="${SEEDS:-0}"
PLAN_EVALS="${PLAN_EVALS:-3}"
PLAN_SAMPLES="${PLAN_SAMPLES:-16}"
PLAN_TOPK="${PLAN_TOPK:-4}"
PLAN_OPT_STEPS="${PLAN_OPT_STEPS:-2}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

latest_model_name() {
  local before_file="$1"
  local after_file="$2"
  comm -13 <(sort "${before_file}") <(sort "${after_file}") | tail -n 1
}

snapshot_models() {
  find course_runs/outputs -mindepth 2 -maxdepth 2 -type d | sed 's#course_runs/outputs/##' | sort
}

train_and_plan() {
  local run_name="$1"
  local encoder="$2"
  local predictor="$3"
  local seed="$4"
  local disable_decoder="$5"

  local before_file
  local after_file
  before_file="$(mktemp)"
  after_file="$(mktemp)"

  snapshot_models > "${before_file}"

  local train_cmd=(
    "${MAMBA_EXE}" run -n dino_wm python train.py
    --config-name train_wsl.yaml
    ckpt_base_path=./course_runs
    "encoder=${encoder}"
    "training.epochs=${EPOCHS}"
    "training.batch_size=${BATCH_SIZE}"
    "training.seed=${seed}"
    "env.num_workers=${NUM_WORKERS}"
    "env.dataset.n_rollout=${N_ROLLOUT}"
  )

  if [[ "${predictor}" != "vit" ]]; then
    train_cmd+=("predictor=${predictor}")
  fi

  if [[ "${disable_decoder}" == "true" ]]; then
    train_cmd+=("has_decoder=False" "model.train_decoder=False")
  fi

  log "train start: ${run_name}"
  "${train_cmd[@]}"
  log "train done: ${run_name}"

  snapshot_models > "${after_file}"
  local model_name
  model_name="$(latest_model_name "${before_file}" "${after_file}")"
  rm -f "${before_file}" "${after_file}"

  if [[ -z "${model_name}" ]]; then
    log "ERROR: could not identify model output for ${run_name}"
    return 1
  fi

  log "plan start: ${run_name} using model ${model_name}"
  "${MAMBA_EXE}" run -n dino_wm python plan.py \
    --config-name plan_point_maze_wsl.yaml \
    ckpt_base_path="${OUT_BASE}" \
    model_name="${model_name}" \
    seed="${seed}" \
    n_evals="${PLAN_EVALS}" \
    n_plot_samples=1 \
    planner.sub_planner.num_samples="${PLAN_SAMPLES}" \
    planner.sub_planner.topk="${PLAN_TOPK}" \
    planner.sub_planner.opt_steps="${PLAN_OPT_STEPS}" \
    planner.n_taken_actions=1
  log "plan done: ${run_name}"
}

for seed in ${SEEDS}; do
  train_and_plan "patch_det_seed${seed}" "dino" "vit" "${seed}" "false"
  train_and_plan "cls_det_seed${seed}" "dino_cls" "vit" "${seed}" "true"
  train_and_plan "patch_gauss_seed${seed}" "dino" "vit_gaussian" "${seed}" "true"
  train_and_plan "cls_gauss_seed${seed}" "dino_cls" "vit_gaussian" "${seed}" "true"
done

log "formal matrix batch completed"
