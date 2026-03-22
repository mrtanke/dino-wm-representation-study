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
  - Gaussian CLS
- larger-sample seed-1 planning follow-up for:
  - deterministic patch
  - Gaussian patch
  - Gaussian CLS
- larger-sample seed-2 planning follow-up for:
  - deterministic patch
  - Gaussian patch

Partially completed:

- larger-sample planning reevaluation

Pending:

- third representation such as DINOv3 or V-JEPA

Progress summary:

- the minimum course-project scope is complete
- the patch-focused larger-sample reevaluation is complete
- the remaining `seed2 CLS` line is now optional unfinished work
- both optional `seed2 CLS` larger-sample branches stalled before `final_eval`
- report cleanup no longer depends on any further CLS reevaluation
- representation expansion beyond DINOv2 patch/CLS has not been implemented
- exploratory pretrained planning sanity has now started on `Wall` and `PushT`
- `Wall` completed successfully under the local WSL setup
- `Wall` also completed successfully under a stronger planning budget
- the reduced-budget `PushT` sanity stalled with poor results
- a stronger-budget `PushT` retry improved planning distance substantially, but still stalled before `final_eval`
- an official-like `PushT` retry completed successfully

Estimated remaining time:

- results aggregation and doc cleanup: about `0.5` to `1.0` hour

Estimated time to finish the current closeout plan:

- about `0.5` to `1.0` hour total for final cleanup and final aggregation

## Final Closeout Summary

The cleanest current reading of the project is:

- main controlled evidence comes from the PointMaze `3-seed x 4-setting` matrix
- `DINOv2 patch + deterministic` is the strongest completed baseline
- `DINOv2 CLS + deterministic` stays close to patch on PointMaze
- Gaussian training objectives improve, but planning does not improve consistently
- the patch-side larger-sample reevaluation is complete enough for a stable closeout story
- the CLS-side larger-sample reevaluation should be treated as partial evidence
- `Wall` pretrained sanity passed cleanly
- `PushT` pretrained sanity improved under a stronger budget, but still did not reach a completed successful evaluation

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
| L7a | `2026-03-18/22-09-15` | shard A, `n_evals=5`, `seed=1` | 3.2888 |
| L7b | `2026-03-18/22-09-15` | shard B, `n_evals=5`, `seed=6` | 4.5844 |
| L8a | `2026-03-18/22-33-51` | shard A, `n_evals=5`, `seed=1` | 3.8905 |
| L8b | `2026-03-18/22-33-51` | shard B, `n_evals=5`, `seed=6` | 4.0224 |
| L9a | `2026-03-18/22-54-50` | shard A, `n_evals=5`, `seed=2` | 3.5163 |
| L9b | `2026-03-18/22-54-50` | shard B, `n_evals=5`, `seed=7` | 3.4006 |
| L10a | `2026-03-18/23-51-26` | shard A, `n_evals=5`, `seed=2` | 3.4006 |
| L10b | `2026-03-18/23-51-26` | shard B, `n_evals=5`, `seed=7` | 5.3276 |

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
- deterministic CLS:
  only shard B completed so far, `mean_state_dist = 3.9672`
  note:
  shard A was attempted twice but not completed, and this group was later dropped from the larger-sample follow-up plan
- Gaussian patch:
  two-shard average over 10 episodes, `mean_state_dist ~= 3.9366`
- Gaussian CLS:
  two-shard average over 10 episodes, `mean_state_dist ~= 3.9565`

Current seed-2 larger-sample picture:

- deterministic patch:
  two-shard average over 10 episodes, `mean_state_dist ~= 3.4584`
- Gaussian patch:
  two-shard average over 10 episodes, `mean_state_dist ~= 4.3641`
- deterministic CLS:
  shard A was started but later stopped without `final_eval`
- Gaussian CLS:
  shard A was attempted twice, including one overnight retry, and still failed to reach `final_eval`

Practical interpretation:

- larger-sample evaluation does not support the earlier idea that Gaussian patch is better
- deterministic CLS remains competitive with deterministic patch on PointMaze
- on seed 0, `CLS + gaussian` is better than `patch + gaussian` but still behind deterministic patch
- on seed 1, both Gaussian variants are in the same broad range as deterministic patch, but neither overturns the broader formal-matrix conclusion
- on seed 2, deterministic patch again looks better than patch Gaussian under the larger-sample follow-up
- `mean_state_dist` remains more informative than `success_rate`

## Cross-Environment Pretrained Sanity

To check whether the local setup generalizes beyond PointMaze, pretrained planning sanity checks were also started on `Wall` and `PushT`.

### Wall

- config:
  `conf/plan_wall_wsl.yaml`
- dataset:
  `$HOME/dino_wm_data/wall_single`
- checkpoint:
  `$HOME/dino_wm_ckpts/outputs/wall_single`
- base output:
  `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260322164743_wall_single_gH5`
- stronger-budget output:
  `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260322205947_wall_single_gH5`
- base result:
  `success_rate = 1.0`, `mean_state_dist = 1.7964`, `mean_visual_dist = 0.5799`
- stronger-budget result:
  `success_rate = 1.0`, `mean_state_dist = 4.1456`, `mean_visual_dist = 1.1377`

### PushT

- config:
  `conf/plan_pusht_wsl.yaml`
- dataset:
  `$HOME/dino_wm_data/pusht_noise`
- checkpoint:
  `$HOME/dino_wm_ckpts/outputs/pusht`
- reduced-budget output:
  `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260322165419_pusht_gH5`
- stronger-budget retry output:
  `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260322190605_pusht_gH5`
- official-like retry output:
  `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260322210051_pusht_gH5`
- reduced-budget status:
  stalled at `step 74` with `mpc/success_rate = 0.0` and very high `mean_state_dist`
- stronger-budget retry status:
  improved to roughly the `60-70` `mean_state_dist` range by `step 52`, but still stalled before `final_eval`
- official-like retry result:
  `success_rate = 1.0`, `mean_state_dist = 20.7644`, `mean_visual_dist = 2.7361`

## Recommended Final Steps

1. Do not continue `seed1 CLS + deterministic` shard-A reruns.
2. Treat the full `seed2 CLS` larger-sample branch as `stalled / optional unfinished`.
3. Do not spend more time on new optional CLS reruns in the main closeout path.
4. Move straight to final aggregation and report cleanup.
5. Build one compact final summary table for formal runs and one for larger-sample follow-up.
6. Polish `docs/COURSE_REPORT_DRAFT.md` into the final submission.

## Collaboration-Friendly Next Steps

For anyone joining the project now, there are several good ways to help without needing to touch the whole codebase at once.

### Documentation and Writing Path

This is a good option if the goal is to strengthen the final submission quickly.

Useful tasks:

- tighten the report language around `success_rate` saturation
- align the wording across `README`, `PROJECT_README`, and the report draft
- convert current experimental notes into a cleaner final narrative
- add a short abstract and a cleaner conclusion section

### Analysis and Table Cleanup Path

This is a good option if the goal is to make the evidence easier to read.

Useful tasks:

- turn the completed larger-sample shard results into one compact summary table
- check that averages reported in prose match the experiment tracker
- prepare one final comparison centered on `mean_state_dist`
- separate `completed`, `partial`, and `optional` evidence clearly

### Optional Experiment Continuation Path

This is a good option if someone wants to keep pushing the empirical side.

Useful tasks:

- summarize the completed larger-sample results and clearly mark the incomplete CLS rows as partial evidence
- check that tracker notes, output folders, and report wording all line up
- keep collaboration docs in sync with the final closeout status

This path is useful, but no longer the only important one. The current project already has a strong patch-focused larger-sample story.

### Reproducibility and Handoff Path

This is a good option if the goal is to make collaboration smoother.

Useful tasks:

- verify that scripts, tracker rows, checkpoint names, and documented commands all line up
- add a short rerun guide for the main completed experiments
- make sure the collaboration repository stays synced with the latest local docs
