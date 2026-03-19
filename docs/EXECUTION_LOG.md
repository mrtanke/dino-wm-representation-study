# DINO-WM Local Execution Log

Date: 2026-03-18

## Scope

This log records what was actually executed for the course-project version of DINO-WM in this repository. It focuses on reproducibility, engineering changes, and the current experiment status.

## Repository Baseline

- Upstream repository cloned locally:
  `https://github.com/gaoyuezhou/dino_wm`
- Main entry points inspected:
  `train.py`, `plan.py`, `conf/train.yaml`, `conf/plan_point_maze.yaml`
- Existing representation configs confirmed:
  `conf/encoder/dino.yaml`
  `conf/encoder/dino_cls.yaml`

## Native Windows Blockers

The project could not be executed reliably in the native Windows environment because the official repository targets Linux.

Observed blockers:

- Default local Python was not compatible with the repo requirements.
- Required dependencies were missing, including `hydra` and `mujoco_py`.
- The official setup expected a Linux-style MuJoCo installation.
- Dataset paths and checkpoints were not configured.

Conclusion:

- Baseline execution was moved to WSL2 Ubuntu instead of native Windows.

## WSL2 Environment Setup

Execution environment:

- WSL distro: `Ubuntu-24.04`
- Python environment manager: `micromamba`
- Environment name: `dino_wm`
- Python: `3.9.19`
- PyTorch: `2.3.0+cu121`
- MuJoCo: `2.1`

Validated in WSL:

- `hydra` imports successfully
- `torch.cuda.is_available() == True`
- `mujoco_py` imports successfully

## Downloaded Resources

Dataset and checkpoints were downloaded from the official DINO-WM OSF release and extracted inside WSL.

Paths used:

- Dataset:
  `/home/zack/dino_wm_data/point_maze`
- Official checkpoints:
  `/home/zack/dino_wm_ckpts/outputs`

## Local Compatibility Changes

### 1. WSL Hydra configs

To avoid Slurm dependencies in the original configs:

- added `conf/plan_point_maze_wsl.yaml`
- added `conf/train_wsl.yaml`

These use the local Hydra basic launcher and write outputs into the repository workspace.

### 2. Stable DINOv2 loading

File changed:

- `models/dino.py`

Change made:

- pinned DINOv2 torch hub loading to ref `ebc1cba`

Reason:

- the moving upstream `main` branch uses Python syntax that breaks under Python 3.9
- pinning the ref makes the project reproducible in the required environment

### 3. Gaussian dynamics extension

Files added or updated:

- `conf/predictor/vit_gaussian.yaml`
- `models/vit.py`
- `models/visual_world_model.py`

Implementation summary:

- predictor now supports `output_mode=gaussian`
- stochastic mode outputs `(mu, logvar)`
- training uses Gaussian NLL with clamped `logvar`
- a small MSE auxiliary term stabilizes optimization
- planning uses `mu` as the rollout prediction

### 4. Predictor checkpoint compatibility fix

File updated:

- `models/vit.py`

Reason:

- older deterministic checkpoints were saved before the new Gaussian predictor head existed
- during planning, those checkpoints could load a `ViTPredictor` instance without the new `output_head` attribute

Fix:

- `forward()` now falls back safely when `output_head` is absent

Impact:

- old deterministic checkpoints remain usable after the stochastic extension was added

## Important Practical Notes

### 1. WSL + /mnt/c DataLoader issue

When the repository is run from `/mnt/c/...`, using `env.num_workers > 0` can stall training during initialization or the first data load.

Working local setting:

- `env.num_workers=0`

This was used for all local sanity runs.

### 2. CLS decoder incompatibility

The existing decoder assumes a spatial patch grid. `DINOv2 CLS` produces a single global token, so decoder-side reconstruction fails without additional adaptation.

Practical fix used in this project:

- disable the decoder for CLS runs

This is acceptable for the course project because the decoder is optional in the original DINO-WM pipeline.

## Executed Runs

### E1_plan: official pretrained PointMaze planning

Command:

```bash
python plan.py --config-name plan_point_maze_wsl.yaml \
  ckpt_base_path=$HOME/dino_wm_ckpts \
  model_name=point_maze \
  n_evals=1 \
  n_plot_samples=1 \
  planner.sub_planner.num_samples=8 \
  planner.sub_planner.topk=2 \
  planner.sub_planner.opt_steps=1 \
  planner.n_taken_actions=1
```

Result:

- `success_rate = 1.0`
- `mean_state_dist = 0.9909`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260318003850_point_maze_gH5`

### E1_train: deterministic DINOv2 patch sanity training

Command:

```bash
python train.py --config-name train_wsl.yaml \
  ckpt_base_path=./course_runs \
  encoder=dino \
  training.epochs=1 \
  training.batch_size=4 \
  env.num_workers=0 \
  env.dataset.n_rollout=32
```

Result:

- `train_loss = 1.8771`
- `val_loss = 1.3130`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/course_runs/outputs/2026-03-18/18-15-20`

### E2_train: deterministic DINOv2 CLS sanity training

Initial failure:

- decoder reconstruction shape mismatch because CLS has no patch grid

Working command:

```bash
python train.py --config-name train_wsl.yaml \
  ckpt_base_path=./course_runs \
  encoder=dino_cls \
  has_decoder=False \
  model.train_decoder=False \
  training.epochs=1 \
  training.batch_size=4 \
  env.num_workers=0 \
  env.dataset.n_rollout=32
```

Result:

- `train_loss = 1.7640`
- `val_loss = 1.2005`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/course_runs/outputs/2026-03-18/18-23-17`

### E3_train: Gaussian DINOv2 patch sanity training

Command:

```bash
python train.py --config-name train_wsl.yaml \
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

Result:

- `train_loss = -1.1801`
- `val_loss = -2.3765`
- `train_z_mse_loss = 0.4202`
- `val_z_mse_loss = 0.0818`
- `train_z_logvar_mean = -2.2194`
- `val_z_logvar_mean = -3.1655`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/course_runs/outputs/2026-03-18/18-29-45`

Note:

- negative total loss is valid here because Gaussian NLL can be negative

### E3_plan: planning with Gaussian checkpoint

Command:

```bash
python plan.py --config-name plan_point_maze_wsl.yaml \
  ckpt_base_path=/mnt/c/Users/zack/Documents/GNN3/course_runs \
  model_name=2026-03-18/18-29-45 \
  n_evals=1 \
  n_plot_samples=1 \
  planner.sub_planner.num_samples=8 \
  planner.sub_planner.topk=2 \
  planner.sub_planner.opt_steps=1 \
  planner.n_taken_actions=1
```

Result:

- `success_rate = 1.0`
- `mean_state_dist = 2.3579`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260318183539_2026-03-18_18-29-45_gH5`

### E1_trained_plan: planning with trained deterministic patch checkpoint

Command:

```bash
python plan.py --config-name plan_point_maze_wsl.yaml \
  ckpt_base_path=/mnt/c/Users/zack/Documents/GNN3/course_runs \
  model_name=2026-03-18/18-15-20 \
  n_evals=1 \
  n_plot_samples=1 \
  planner.sub_planner.num_samples=8 \
  planner.sub_planner.topk=2 \
  planner.sub_planner.opt_steps=1 \
  planner.n_taken_actions=1
```

Result:

- `success_rate = 1.0`
- `mean_state_dist = 0.9120`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260318185810_2026-03-18_18-15-20_gH5`

### E2_plan: planning with trained deterministic CLS checkpoint

Command:

```bash
python plan.py --config-name plan_point_maze_wsl.yaml \
  ckpt_base_path=/mnt/c/Users/zack/Documents/GNN3/course_runs \
  model_name=2026-03-18/18-23-17 \
  n_evals=1 \
  n_plot_samples=1 \
  planner.sub_planner.num_samples=8 \
  planner.sub_planner.topk=2 \
  planner.sub_planner.opt_steps=1 \
  planner.n_taken_actions=1
```

Result:

- `success_rate = 1.0`
- `mean_state_dist = 0.5267`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260318185810_2026-03-18_18-23-17_gH5`

### E4_train: Gaussian DINOv2 CLS sanity training

Command:

```bash
python train.py --config-name train_wsl.yaml \
  ckpt_base_path=./course_runs \
  predictor=vit_gaussian \
  encoder=dino_cls \
  has_decoder=False \
  model.train_decoder=False \
  training.epochs=1 \
  training.batch_size=4 \
  env.num_workers=0 \
  env.dataset.n_rollout=32
```

Result:

- `train_loss = -2.0987`
- `val_loss = -2.6678`
- `val_z_mse_loss = 0.0329`
- `val_z_logvar_mean = -3.5105`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/course_runs/outputs/2026-03-18/18-58-38`

### E4_plan: planning with Gaussian CLS checkpoint

Command:

```bash
python plan.py --config-name plan_point_maze_wsl.yaml \
  ckpt_base_path=/mnt/c/Users/zack/Documents/GNN3/course_runs \
  model_name=2026-03-18/18-58-38 \
  n_evals=1 \
  n_plot_samples=1 \
  planner.sub_planner.num_samples=8 \
  planner.sub_planner.topk=2 \
  planner.sub_planner.opt_steps=1 \
  planner.n_taken_actions=1
```

Result:

- `success_rate = 1.0`
- `mean_state_dist = 1.1232`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260318190212_2026-03-18_18-58-38_gH5`

## Current Project Status

Implemented and executed:

- `DINOv2 patch + deterministic`
- `DINOv2 CLS + deterministic`
- `DINOv2 patch + gaussian`
- `DINOv2 CLS + gaussian`

Not yet implemented:

- `DINOv3` integration
- `V-JEPA` integration
- larger multi-seed benchmark runs

## Recommendation

The implementation stage is complete enough for a course-project milestone. The next step should focus on reporting and on expanding sanity runs into longer, repeatable experiments.

## Formal Matrix Batch

A longer PointMaze batch launcher was added:

- `scripts/wsl_pointmaze_formal_matrix.sh`

Current default use:

- 4 model combinations
- configurable seeds
- configurable epoch count
- training followed immediately by planning

The first longer batch was started on 2026-03-18 with:

- `seed = 0`
- `epochs = 3`
- `n_rollout = 64`
- `n_evals = 3`

Current log file:

- `/mnt/c/Users/zack/Documents/GNN3/logs/pointmaze_formal_matrix_seed0.log`

Completed outputs from the first formal batch:

- deterministic patch:
  `/mnt/c/Users/zack/Documents/GNN3/course_runs/outputs/2026-03-18/19-12-06`
- deterministic CLS:
  `/mnt/c/Users/zack/Documents/GNN3/course_runs/outputs/2026-03-18/20-01-01`
- Gaussian patch:
  `/mnt/c/Users/zack/Documents/GNN3/course_runs/outputs/2026-03-18/20-20-39`
- Gaussian CLS:
  `/mnt/c/Users/zack/Documents/GNN3/course_runs/outputs/2026-03-18/20-45-03`

Summary from the first formal batch:

- deterministic patch:
  `train_loss = 0.0800`, `val_loss = 0.0440`, `success_rate = 1.0`, `mean_state_dist = 3.2298`
- deterministic CLS:
  `train_loss = 0.0414`, `val_loss = 0.0238`, `success_rate = 1.0`, `mean_state_dist = 2.1996`
- Gaussian patch:
  `train_loss = -2.9815`, `val_loss = -3.1822`, `success_rate = 1.0`, `mean_state_dist = 3.6079`
- Gaussian CLS:
  `train_loss = -3.0505`, `val_loss = -3.2116`, `success_rate = 1.0`, `mean_state_dist = 1.7849`

The additional formal batches for `seed = 1` and `seed = 2` were also completed. Final epoch losses and final planning metrics were recorded in `docs/EXPERIMENT_TRACKER.csv`.

Three-seed summary:

- deterministic patch:
  mean `train_loss = 0.0825`, mean `val_loss = 0.0429`, mean `mean_state_dist = 3.2806`
- deterministic CLS:
  mean `train_loss = 0.0411`, mean `val_loss = 0.0240`, mean `mean_state_dist = 3.3047`
- Gaussian patch:
  mean `train_loss = -2.9815`, mean `val_loss = -3.1785`, mean `mean_state_dist = 3.7772`
- Gaussian CLS:
  mean `train_loss = -3.0751`, mean `val_loss = -3.2282`, mean `mean_state_dist = 3.3277`

Important interpretation note:

- all formal runs still ended with `final_eval/success_rate = 1.0`
- however, this was measured with only `n_evals = 3`
- the `mean_state_dist` metric shows meaningful differences even when the success rate saturates

## Larger-Sample Planning Evaluation

### Motivation

The three-seed formal matrix still used only `n_evals = 3` during planning. This was enough to validate the full loop, but it was not enough to treat `success_rate` as a strong comparison metric because all runs saturated at `1.0`.

To improve the evaluation:

- planning visualization was disabled during larger runs
- planning video saving was disabled
- `eval_seed` generation in `plan.py` was fixed so that `seed = 0` no longer reused the same evaluation seed for every sample

Files updated for this:

- `planning/evaluator.py`
- `planning/mpc.py`
- `plan.py`
- `scripts/wsl_pointmaze_large_eval.sh`
- `scripts/wsl_pointmaze_large_eval_matrix.sh`
- `scripts/wsl_pointmaze_large_eval_single.sh`

### Stability issue

Direct `n_evals = 10` runs were not consistently stable in WSL2.

Observed behavior:

- deterministic patch completed successfully with `n_evals = 10`
- other configurations sometimes stopped mid-rollout without a Python traceback
- WSL occasionally reported `Wsl/Service/E_UNEXPECTED`

Conclusion:

- the most reliable workaround was to split larger evaluation into two runs with `n_evals = 5`
- using base seeds `0` and `5` yields evaluation seed sets `1..5` and `6..10`
- together these two shards cover 10 evaluation episodes while keeping each run smaller and more stable

### Completed larger-sample runs

#### L1: deterministic patch, seed-0 checkpoint, direct `n_evals = 10`

Command:

```bash
python plan.py --config-name plan_point_maze_wsl.yaml \
  ckpt_base_path=/mnt/c/Users/zack/Documents/GNN3/course_runs \
  model_name=2026-03-18/19-12-06 \
  seed=0 \
  n_evals=10 \
  n_plot_samples=0 \
  plot_rollouts=False \
  save_video=False \
  planner.sub_planner.num_samples=16 \
  planner.sub_planner.topk=4 \
  planner.sub_planner.opt_steps=2 \
  planner.n_taken_actions=1
```

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 3.1738460684`
- `final_eval/mean_visual_dist = 0.4748990221`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260319012420_2026-03-18_19-12-06_gH5`

#### L2a: deterministic CLS, seed-0 checkpoint, shard A with `n_evals = 5`

Command:

```bash
python plan.py --config-name plan_point_maze_wsl.yaml \
  ckpt_base_path=/mnt/c/Users/zack/Documents/GNN3/course_runs \
  model_name=2026-03-18/20-01-01 \
  seed=0 \
  n_evals=5 \
  n_plot_samples=0 \
  plot_rollouts=False \
  save_video=False \
  planner.sub_planner.num_samples=16 \
  planner.sub_planner.topk=4 \
  planner.sub_planner.opt_steps=2 \
  planner.n_taken_actions=1
```

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 3.7136193550`
- `final_eval/mean_visual_dist = 0.4988440689`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260319210713_2026-03-18_20-01-01_gH5`

#### L2b: deterministic CLS, seed-0 checkpoint, shard B with `n_evals = 5`

Command:

```bash
python plan.py --config-name plan_point_maze_wsl.yaml \
  ckpt_base_path=/mnt/c/Users/zack/Documents/GNN3/course_runs \
  model_name=2026-03-18/20-01-01 \
  seed=5 \
  n_evals=5 \
  n_plot_samples=0 \
  plot_rollouts=False \
  save_video=False \
  planner.sub_planner.num_samples=16 \
  planner.sub_planner.topk=4 \
  planner.sub_planner.opt_steps=2 \
  planner.n_taken_actions=1
```

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 3.3445054831`
- `final_eval/mean_visual_dist = 0.4883875425`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260319210914_2026-03-18_20-01-01_gH5`

Estimated combined seed-0 CLS deterministic mean over 10 evaluation episodes:

- `mean_state_dist ~= 3.5291`

#### L3a: Gaussian patch, seed-0 checkpoint, shard A with `n_evals = 5`

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 4.4740299645`
- `final_eval/mean_visual_dist = 0.4716982887`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260319211111_2026-03-18_20-20-39_gH5`

#### L3b: Gaussian patch, seed-0 checkpoint, shard B with `n_evals = 5`

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 4.2023629437`
- `final_eval/mean_visual_dist = 0.5148650085`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260319211152_2026-03-18_20-20-39_gH5`

Estimated combined seed-0 patch Gaussian mean over 10 evaluation episodes:

- `mean_state_dist ~= 4.3382`

#### L4a: Gaussian CLS, seed-0 checkpoint, shard A with `n_evals = 5`

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 4.2964758912`
- `final_eval/mean_visual_dist = 0.5237484056`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260319221318_2026-03-18_20-45-03_gH5`

#### L4b: Gaussian CLS, seed-0 checkpoint, shard B with `n_evals = 5`

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 3.3519865192`
- `final_eval/mean_visual_dist = 0.4812220982`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260319221626_2026-03-18_20-45-03_gH5`

Estimated combined seed-0 CLS Gaussian mean over 10 evaluation episodes:

- `mean_state_dist ~= 3.8242`

#### L5a: deterministic patch, seed-1 checkpoint, shard A with `n_evals = 5`

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 4.0345669560`
- `final_eval/mean_visual_dist = 0.4448992878`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260319222040_2026-03-18_21-12-22_gH5`

#### L5b: deterministic patch, seed-1 checkpoint, shard B with `n_evals = 5`

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 4.2033059716`
- `final_eval/mean_visual_dist = 0.4675621811`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260319222509_2026-03-18_21-12-22_gH5`

Estimated combined seed-1 patch deterministic mean over 10 evaluation episodes:

- `mean_state_dist ~= 4.1189`

### Current larger-sample status

Completed:

- seed-0 deterministic patch
- seed-0 deterministic CLS, 2 shards
- seed-0 Gaussian patch, 2 shards
- seed-0 Gaussian CLS, 2 shards
- seed-1 deterministic patch, 2 shards

Not yet completed:

- seed-1 deterministic CLS
- seed-1 Gaussian patch
- seed-1 Gaussian CLS
- all larger-sample runs for seed 2

Estimated remaining runtime from the current state:

- seed-0 Gaussian CLS, 2 shards: about `0.5` to `1.0` hour
- seed-1 larger-sample reevaluation, 8 shards total: about `2.0` to `3.0` hours
- seed-2 larger-sample reevaluation, 8 shards total: about `2.0` to `3.0` hours
- final aggregation and documentation pass: about `0.5` to `1.0` hour

Estimated total remaining time for the current reevaluation plan:

- about `5` to `8` hours

Overall project-status interpretation:

- the core course-project deliverable is already complete
- the current work is focused on making the planning comparison more statistically credible
- the broader representation expansion proposed at the start of the project has not been implemented yet

Current takeaway from larger-sample PointMaze evaluation:

- the smaller `n_evals = 3` formal matrix was directionally useful
- however, the larger-sample seed-0 comparison currently supports:
  - deterministic patch better than patch Gaussian
  - deterministic CLS competitive with deterministic patch
  - CLS Gaussian better than patch Gaussian on seed 0, but still weaker than deterministic patch
  - success rate still saturates, so `mean_state_dist` remains the more useful metric
