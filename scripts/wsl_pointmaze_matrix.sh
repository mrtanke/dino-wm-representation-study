#!/usr/bin/env bash

set -euo pipefail

MAMBA_EXE="${HOME}/.local/bin/micromamba"
REPO_DIR="/mnt/c/Users/zack/Documents/GNN3"

export DATASET_DIR="${HOME}/dino_wm_data"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export MUJOCO_GL="egl"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export WANDB_MODE=offline

cd "${REPO_DIR}"

run_cmd() {
  local title="$1"
  shift
  echo "==== ${title} ===="
  "$@"
}

# Small PointMaze matrix:
# 1. DINOv2 patch + deterministic
# 2. DINOv2 CLS   + deterministic
# 3. DINOv2 patch + gaussian
# 4. DINOv2 CLS   + gaussian
#
# Existing sanity checkpoints can be re-planned by overriding MODEL_NAME_*
# to save reruns when needed.

PATCH_DET_MODEL_NAME="${PATCH_DET_MODEL_NAME:-}"
CLS_DET_MODEL_NAME="${CLS_DET_MODEL_NAME:-}"
PATCH_GAUSS_MODEL_NAME="${PATCH_GAUSS_MODEL_NAME:-}"
CLS_GAUSS_MODEL_NAME="${CLS_GAUSS_MODEL_NAME:-}"

if [[ -z "${PATCH_DET_MODEL_NAME}" ]]; then
  run_cmd train_patch_deterministic \
    "${MAMBA_EXE}" run -n dino_wm python train.py \
    --config-name train_wsl.yaml \
    ckpt_base_path=./course_runs \
    encoder=dino \
    training.epochs=1 \
    training.batch_size=4 \
    env.num_workers=0 \
    env.dataset.n_rollout=32
  echo "Set PATCH_DET_MODEL_NAME manually from the newest course_runs/outputs path before planning."
fi

if [[ -n "${PATCH_DET_MODEL_NAME}" ]]; then
  run_cmd plan_patch_deterministic \
    "${MAMBA_EXE}" run -n dino_wm python plan.py \
    --config-name plan_point_maze_wsl.yaml \
    ckpt_base_path="${REPO_DIR}/course_runs" \
    model_name="${PATCH_DET_MODEL_NAME}" \
    n_evals=1 \
    n_plot_samples=1 \
    planner.sub_planner.num_samples=8 \
    planner.sub_planner.topk=2 \
    planner.sub_planner.opt_steps=1 \
    planner.n_taken_actions=1
fi

if [[ -z "${CLS_DET_MODEL_NAME}" ]]; then
  run_cmd train_cls_deterministic \
    "${MAMBA_EXE}" run -n dino_wm python train.py \
    --config-name train_wsl.yaml \
    ckpt_base_path=./course_runs \
    encoder=dino_cls \
    has_decoder=False \
    model.train_decoder=False \
    training.epochs=1 \
    training.batch_size=4 \
    env.num_workers=0 \
    env.dataset.n_rollout=32
  echo "Set CLS_DET_MODEL_NAME manually from the newest course_runs/outputs path before planning."
fi

if [[ -n "${CLS_DET_MODEL_NAME}" ]]; then
  run_cmd plan_cls_deterministic \
    "${MAMBA_EXE}" run -n dino_wm python plan.py \
    --config-name plan_point_maze_wsl.yaml \
    ckpt_base_path="${REPO_DIR}/course_runs" \
    model_name="${CLS_DET_MODEL_NAME}" \
    n_evals=1 \
    n_plot_samples=1 \
    planner.sub_planner.num_samples=8 \
    planner.sub_planner.topk=2 \
    planner.sub_planner.opt_steps=1 \
    planner.n_taken_actions=1
fi

if [[ -z "${PATCH_GAUSS_MODEL_NAME}" ]]; then
  run_cmd train_patch_gaussian \
    "${MAMBA_EXE}" run -n dino_wm python train.py \
    --config-name train_wsl.yaml \
    ckpt_base_path=./course_runs \
    predictor=vit_gaussian \
    encoder=dino \
    has_decoder=False \
    model.train_decoder=False \
    training.epochs=1 \
    training.batch_size=4 \
    env.num_workers=0 \
    env.dataset.n_rollout=32
  echo "Set PATCH_GAUSS_MODEL_NAME manually from the newest course_runs/outputs path before planning."
fi

if [[ -n "${PATCH_GAUSS_MODEL_NAME}" ]]; then
  run_cmd plan_patch_gaussian \
    "${MAMBA_EXE}" run -n dino_wm python plan.py \
    --config-name plan_point_maze_wsl.yaml \
    ckpt_base_path="${REPO_DIR}/course_runs" \
    model_name="${PATCH_GAUSS_MODEL_NAME}" \
    n_evals=1 \
    n_plot_samples=1 \
    planner.sub_planner.num_samples=8 \
    planner.sub_planner.topk=2 \
    planner.sub_planner.opt_steps=1 \
    planner.n_taken_actions=1
fi

if [[ -z "${CLS_GAUSS_MODEL_NAME}" ]]; then
  run_cmd train_cls_gaussian \
    "${MAMBA_EXE}" run -n dino_wm python train.py \
    --config-name train_wsl.yaml \
    ckpt_base_path=./course_runs \
    predictor=vit_gaussian \
    encoder=dino_cls \
    has_decoder=False \
    model.train_decoder=False \
    training.epochs=1 \
    training.batch_size=4 \
    env.num_workers=0 \
    env.dataset.n_rollout=32
  echo "Set CLS_GAUSS_MODEL_NAME manually from the newest course_runs/outputs path before planning."
fi

if [[ -n "${CLS_GAUSS_MODEL_NAME}" ]]; then
  run_cmd plan_cls_gaussian \
    "${MAMBA_EXE}" run -n dino_wm python plan.py \
    --config-name plan_point_maze_wsl.yaml \
    ckpt_base_path="${REPO_DIR}/course_runs" \
    model_name="${CLS_GAUSS_MODEL_NAME}" \
    n_evals=1 \
    n_plot_samples=1 \
    planner.sub_planner.num_samples=8 \
    planner.sub_planner.topk=2 \
    planner.sub_planner.opt_steps=1 \
    planner.n_taken_actions=1
fi
