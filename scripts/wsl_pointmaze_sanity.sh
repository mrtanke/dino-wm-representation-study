#!/usr/bin/env bash

set -euo pipefail

MAMBA_EXE="${HOME}/.local/bin/micromamba"
export DATASET_DIR="${HOME}/dino_wm_data"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export MUJOCO_GL="egl"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export WANDB_MODE=offline

cd /mnt/c/Users/zack/Documents/GNN3

run_cmd() {
  local title="$1"
  shift
  echo "==== ${title} ===="
  "$@"
}

run_cmd plan_pointmaze_pretrained \
  "${MAMBA_EXE}" run -n dino_wm python plan.py \
  --config-name plan_point_maze_wsl.yaml \
  ckpt_base_path="${HOME}/dino_wm_ckpts" \
  model_name=point_maze \
  n_evals=1 \
  n_plot_samples=1 \
  planner.sub_planner.num_samples=8 \
  planner.sub_planner.topk=2 \
  planner.sub_planner.opt_steps=1 \
  planner.n_taken_actions=1

run_cmd train_pointmaze_dino_patch_sanity \
  "${MAMBA_EXE}" run -n dino_wm python train.py \
  --config-name train_wsl.yaml \
  ckpt_base_path=./course_runs \
  encoder=dino \
  training.epochs=1 \
  training.batch_size=4 \
  env.num_workers=0 \
  env.dataset.n_rollout=32

run_cmd train_pointmaze_dino_cls_sanity \
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

run_cmd train_pointmaze_dino_gaussian_sanity \
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
