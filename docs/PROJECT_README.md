# Course Project README

## Overview

This repository contains a course-project extension of DINO-WM focused on two questions:

1. Representation study:
   compare `DINOv2 patch` and `DINOv2 CLS`
2. Dynamics study:
   compare `deterministic` and `gaussian` latent transition models

The implementation currently supports:

- pretrained PointMaze planning
- deterministic patch-token training
- deterministic CLS training with decoder disabled
- Gaussian latent-transition training
- Gaussian checkpoint planning
- full `patch/cls x deterministic/gaussian` sanity matrix on PointMaze

## Important Files

- Main report draft:
  [COURSE_REPORT_DRAFT.md](C:/Users/zack/Documents/GNN3/docs/COURSE_REPORT_DRAFT.md)
- Execution log:
  [EXECUTION_LOG.md](C:/Users/zack/Documents/GNN3/docs/EXECUTION_LOG.md)
- Experiment table:
  [EXPERIMENT_TRACKER.csv](C:/Users/zack/Documents/GNN3/docs/EXPERIMENT_TRACKER.csv)

Core code changes:

- WSL planning config:
  `conf/plan_point_maze_wsl.yaml`
- WSL training config:
  `conf/train_wsl.yaml`
- Gaussian predictor config:
  `conf/predictor/vit_gaussian.yaml`
- DINOv2 encoder pin:
  `models/dino.py`
- Gaussian transition implementation:
  `models/vit.py`
  `models/visual_world_model.py`

Primary scripts:

- `scripts/wsl_verify_env.sh`
- `scripts/wsl_pointmaze_sanity.sh`
- `scripts/wsl_pointmaze_matrix.sh`
- `scripts/wsl_pointmaze_formal_matrix.sh`
- `scripts/wsl_pointmaze_large_eval.sh`
- `scripts/wsl_pointmaze_large_eval_matrix.sh`
- `scripts/wsl_pointmaze_large_eval_single.sh`

## Environment Notes

This project was run in WSL2 Ubuntu 24.04.

Key requirements:

- Python 3.9
- MuJoCo 2.1
- CUDA-enabled PyTorch
- dataset extracted under `$HOME/dino_wm_data`
- checkpoints extracted under `$HOME/dino_wm_ckpts`

Important practical note:

- use `env.num_workers=0` for local sanity runs on WSL2 if the repository lives under `/mnt/c`

## Sanity Commands

### 1. Pretrained PointMaze planning

```bash
python plan.py \
  --config-name plan_point_maze_wsl.yaml \
  ckpt_base_path=$HOME/dino_wm_ckpts \
  model_name=point_maze \
  n_evals=1 \
  n_plot_samples=1 \
  planner.sub_planner.num_samples=8 \
  planner.sub_planner.topk=2 \
  planner.sub_planner.opt_steps=1 \
  planner.n_taken_actions=1
```

### 2. Deterministic patch training

```bash
python train.py \
  --config-name train_wsl.yaml \
  ckpt_base_path=./course_runs \
  encoder=dino \
  training.epochs=1 \
  training.batch_size=4 \
  env.num_workers=0 \
  env.dataset.n_rollout=32
```

### 3. Deterministic CLS training

```bash
python train.py \
  --config-name train_wsl.yaml \
  ckpt_base_path=./course_runs \
  encoder=dino_cls \
  has_decoder=False \
  model.train_decoder=False \
  training.epochs=1 \
  training.batch_size=4 \
  env.num_workers=0 \
  env.dataset.n_rollout=32
```

### 4. Gaussian patch training

```bash
python train.py \
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
```

### 5. Gaussian checkpoint planning

```bash
python plan.py \
  --config-name plan_point_maze_wsl.yaml \
  ckpt_base_path=/mnt/c/Users/zack/Documents/GNN3/course_runs \
  model_name=2026-03-18/18-29-45 \
  n_evals=1 \
  n_plot_samples=1 \
  planner.sub_planner.num_samples=8 \
  planner.sub_planner.topk=2 \
  planner.sub_planner.opt_steps=1 \
  planner.n_taken_actions=1
```

## Current Status

Completed:

- full `patch/cls x deterministic/gaussian` sanity matrix
- full three-seed formal matrix with `epochs=3`, `n_rollout=64`, `n_evals=3`
- larger-sample seed-0 planning follow-up for:
  - deterministic patch
  - deterministic CLS
  - Gaussian patch

Partially completed:

- larger-sample planning reevaluation

Pending:

- larger-sample seed-0 planning for `CLS + gaussian`
- larger-sample seed-1 planning for all four configurations
- larger-sample seed-2 planning for all four configurations
- third representation such as DINOv3 or V-JEPA

Progress summary:

- the minimum course-project scope is complete
- the stricter planning reevaluation is still in progress
- representation expansion beyond DINOv2 patch/CLS has not been implemented

Estimated remaining time:

- `seed0 CLS + gaussian`, two `n_evals=5` shards: about `0.5` to `1.0` hour
- `seed1`, four models, two shards each: about `2.0` to `3.0` hours
- `seed2`, four models, two shards each: about `2.0` to `3.0` hours
- results aggregation and doc cleanup: about `0.5` to `1.0` hour

Estimated time to finish the current larger-sample reevaluation plan:

- about `5` to `8` hours total

## Current Sanity Results

| Run | Encoder | Transition | Key result |
|---|---|---|---|
| E1_plan | DINOv2 patch | deterministic | `success_rate=1.0`, `mean_state_dist=0.9909` |
| E1_train | DINOv2 patch | deterministic | `train_loss=1.8771`, `val_loss=1.3130` |
| E2_train | DINOv2 CLS | deterministic | `train_loss=1.7640`, `val_loss=1.2005` |
| E2_plan | DINOv2 CLS | deterministic | `success_rate=1.0`, `mean_state_dist=0.5267` |
| E3_train | DINOv2 patch | gaussian | `val_z_mse_loss=0.0818`, `val_loss=-2.3765` |
| E3_plan | DINOv2 patch | gaussian | `success_rate=1.0`, `mean_state_dist=2.3579` |
| E4_train | DINOv2 CLS | gaussian | `val_z_mse_loss=0.0329`, `val_loss=-2.6678` |
| E4_plan | DINOv2 CLS | gaussian | `success_rate=1.0`, `mean_state_dist=1.1232` |

## Formal Matrix Results

| Run | Encoder | Transition | Training result | Planning result |
|---|---|---|---|---|
| F1 | DINOv2 patch | deterministic | `train_loss=0.0800`, `val_loss=0.0440` | `success_rate=1.0`, `mean_state_dist=3.2298` |
| F2 | DINOv2 CLS | deterministic | `train_loss=0.0414`, `val_loss=0.0238` | `success_rate=1.0`, `mean_state_dist=2.1996` |
| F3 | DINOv2 patch | gaussian | `train_loss=-2.9815`, `val_loss=-3.1822` | `success_rate=1.0`, `mean_state_dist=3.6079` |
| F4 | DINOv2 CLS | gaussian | `train_loss=-3.0505`, `val_loss=-3.2116` | `success_rate=1.0`, `mean_state_dist=1.7849` |

## Three-Seed Summary

The formal matrix was repeated for `seed=0,1,2`.

| Encoder | Transition | Mean train loss | Mean val loss | Mean final state distance |
|---|---|---:|---:|---:|
| DINOv2 patch | deterministic | 0.0825 | 0.0429 | 3.2806 |
| DINOv2 CLS | deterministic | 0.0411 | 0.0240 | 3.3047 |
| DINOv2 patch | gaussian | -2.9815 | -3.1785 | 3.7772 |
| DINOv2 CLS | gaussian | -3.0751 | -3.2282 | 3.3277 |

Interpretation:

- `success_rate=1.0` saturated in all formal runs, so it is not very discriminative here
- `mean_state_dist` is the more useful comparison metric in the current setup
- on the current 3-seed PointMaze matrix, `patch + deterministic` has the lowest mean final state distance

## Larger-Sample Planning Follow-Up

To address the fact that `n_evals=3` was too small, planning was extended to a larger-sample follow-up.

Important engineering note:

- direct `n_evals=10` was not stable for every configuration under WSL2
- the stable workaround was to split evaluation into two shards with `n_evals=5`
- using base seeds `0` and `5` yields evaluation seeds `1..5` and `6..10`

Completed larger-sample outputs:

| Run | Checkpoint | Eval strategy | Final state distance |
|---|---|---|---:|
| L1 | `2026-03-18/19-12-06` | direct `n_evals=10` | 3.1738 |
| L2a | `2026-03-18/20-01-01` | shard A, `n_evals=5`, `seed=0` | 3.7136 |
| L2b | `2026-03-18/20-01-01` | shard B, `n_evals=5`, `seed=5` | 3.3445 |
| L3a | `2026-03-18/20-20-39` | shard A, `n_evals=5`, `seed=0` | 4.4740 |
| L3b | `2026-03-18/20-20-39` | shard B, `n_evals=5`, `seed=5` | 4.2024 |
| L4a | `2026-03-18/20-45-03` | shard A, `n_evals=5`, `seed=0` | 4.2965 |
| L4b | `2026-03-18/20-45-03` | shard B, `n_evals=5`, `seed=5` | 3.3520 |
| L5a | `2026-03-18/21-12-22` | shard A, `n_evals=5`, `seed=1` | 4.0346 |
| L5b | `2026-03-18/21-12-22` | shard B, `n_evals=5`, `seed=6` | 4.2033 |

Current seed-0 larger-sample picture:

- deterministic patch:
  direct `n_evals=10`, `mean_state_dist = 3.1738`
- deterministic CLS:
  two-shard average over 10 episodes, `mean_state_dist ~= 3.5291`
- Gaussian patch:
  two-shard average over 10 episodes, `mean_state_dist ~= 4.3382`
- Gaussian CLS:
  two-shard average over 10 episodes, `mean_state_dist ~= 3.8242`

Current seed-1 larger-sample picture:

- deterministic patch:
  two-shard average over 10 episodes, `mean_state_dist ~= 4.1189`

Practical interpretation:

- larger-sample evaluation does not support the earlier idea that Gaussian patch is better
- deterministic CLS remains competitive with deterministic patch on PointMaze
- on seed 0, `CLS + gaussian` is better than `patch + gaussian` but still behind deterministic patch
- `mean_state_dist` remains more informative than `success_rate`

## Recommended Final Steps

1. Expand each sanity run into a longer training run.
2. Collect planning metrics over multiple seeds.
3. Fill `docs/EXPERIMENT_TRACKER.csv` with final results.
4. Polish `docs/COURSE_REPORT_DRAFT.md` into the final submission.
