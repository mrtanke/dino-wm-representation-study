#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${DATASET_DIR:-}" ]]; then
  echo "DATASET_DIR is not set."
  echo "Example: export DATASET_DIR=/path/to/data"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="course_project_logs/${STAMP}"
mkdir -p "${LOG_DIR}"

echo "Logs will be written to ${LOG_DIR}"

run_cmd() {
  local name="$1"
  shift
  echo "==== ${name} ===="
  echo "$*" | tee "${LOG_DIR}/${name}.cmd.txt"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

run_cmd plan_pointmaze_pretrained \
  python plan.py --config-name plan_point_maze.yaml model_name=point_maze

run_cmd train_pointmaze_dino_patch \
  python train.py --config-name train.yaml env=point_maze encoder=dino frameskip=5 num_hist=3

run_cmd train_pointmaze_dino_cls \
  python train.py --config-name train.yaml env=point_maze encoder=dino_cls frameskip=5 num_hist=3

echo "Deterministic representation experiments completed."
echo "Add the stochastic predictor command after the gaussian transition config is implemented."
