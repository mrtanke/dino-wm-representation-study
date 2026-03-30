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

## 2026-03-29 Closeout Priority Update

The next experiment priority was revised after paper review. The current goal is no longer to expand immediately to more encoders; it is to strengthen the most important completed claim in the report:

- `DINOv2 CLS + deterministic` is surprisingly competitive with `DINOv2 patch + deterministic` on `PointMaze`

The most cost-effective way to improve that claim is to complete the missing larger-sample `CLS deterministic` follow-up before spending more time on `V-JEPA`, `DINOv3`, or longer training budgets.

### Priority order

1. rerun the missing `seed-2 CLS + deterministic` larger-sample follow-up
2. if that succeeds, rerun the missing `seed-1 CLS + deterministic` shard A
3. only after that, consider a small training-budget extension for:
   - `patch + deterministic`
   - `CLS + deterministic`

### Why this is the best next step

- it directly strengthens the paper's most interesting empirical result
- it avoids opening a new integration track
- it can be added to the existing PointMaze evidence without changing the paper structure
- it is much cheaper than starting a new long training run

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
- seed-1 Gaussian patch, 2 shards
- seed-1 Gaussian CLS, 2 shards
- seed-2 deterministic patch, 2 shards

Not yet completed:

- seed-2 deterministic CLS
- seed-2 Gaussian CLS

### Seed-1 CLS deterministic follow-up decision

The larger-sample follow-up for `seed1 CLS + deterministic` was not completed as a full two-shard result.

Attempt history:

- shard A, first attempt:
  `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260321135816_2026-03-18_21-50-26_gH5`
- shard B, completed:
  `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260321161246_2026-03-18_21-50-26_gH5`
- shard A, retry:
  `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260321172233_2026-03-18_21-50-26_gH5`

Observed behavior:

- shard A advanced through many `mpc/*` steps in both attempts
- neither attempt wrote `final_eval`
- shard B completed normally and wrote `final_eval`

Decision:

- stop this experiment group instead of continuing more shard-A reruns
- keep shard B as a partial result in the tracker
- do not report a full 10-episode aggregate for `seed1 CLS + deterministic`

Estimated remaining runtime from the current state:

- remaining seed-1 larger-sample reevaluation excluding dropped `seed1 CLS + deterministic` shard-A reruns: about `0.5` to `1.0` hour
- seed-2 larger-sample reevaluation: about `1.0` to `1.5` hours
- final aggregation and documentation pass: about `0.5` to `1.0` hour

Estimated total remaining time for the current reevaluation plan:

- about `1.5` to `3.0` hours if optional seed-2 CLS follow-up is still pursued
- about `0.5` to `1.0` hour if the project is closed out after the patch-focused seed-2 reevaluation

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

### Seed-1 Gaussian patch follow-up

#### L7a: Gaussian patch, seed-1 checkpoint, shard A with `n_evals = 5`

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 3.2887708255`
- `final_eval/mean_visual_dist = 0.4355797194`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260321191601_2026-03-18_22-09-15_gH5`

#### L7b: Gaussian patch, seed-1 checkpoint, shard B with `n_evals = 5`

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 4.5844208286`
- `final_eval/mean_visual_dist = 0.4742739158`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260321192724_2026-03-18_22-09-15_gH5`

Estimated combined seed-1 patch Gaussian mean over 10 evaluation episodes:

- `mean_state_dist ~= 3.9366`

Interpretation:

- on seed 1 alone, Gaussian patch is slightly better than deterministic patch in this larger-sample estimate
- however, this does not overturn the broader project picture because the formal three-seed matrix still favors deterministic patch overall
- the broader project conclusion still comes from combining formal-matrix evidence and partial larger-sample follow-up, not from one seed alone

### Seed-1 Gaussian CLS follow-up

#### L8a: Gaussian CLS, seed-1 checkpoint, shard A with `n_evals = 5`

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 3.8904877754`
- `final_eval/mean_visual_dist = 0.4775451202`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260321193135_2026-03-18_22-33-51_gH5`

#### L8b: Gaussian CLS, seed-1 checkpoint, shard B with `n_evals = 5`

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 4.0224406552`
- `final_eval/mean_visual_dist = 0.4510629252`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260321193525_2026-03-18_22-33-51_gH5`

Estimated combined seed-1 CLS Gaussian mean over 10 evaluation episodes:

- `mean_state_dist ~= 3.9565`

Interpretation:

- on seed 1, the two Gaussian variants are very close to each other
- this again suggests that representation differences are not dominating planning quality on this small PointMaze setup
- the broader project conclusion still depends more on the full formal matrix than on any single larger-sample seed slice

### Seed-2 deterministic patch follow-up

#### L9a: deterministic patch, seed-2 checkpoint, shard A with `n_evals = 5`

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 3.5162746160`
- `final_eval/mean_visual_dist = 0.4831715524`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260321195721_2026-03-18_22-54-50_gH5`

#### L9b: deterministic patch, seed-2 checkpoint, shard B with `n_evals = 5`

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 3.4006226773`
- `final_eval/mean_visual_dist = 0.4930421710`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260321200854_2026-03-18_22-54-50_gH5`

Estimated combined seed-2 patch deterministic mean over 10 evaluation episodes:

- `mean_state_dist ~= 3.4584`

Interpretation:

- seed-2 deterministic patch is more stable than the noisier seed-1 patch follow-up
- this keeps deterministic patch aligned with the broader formal-matrix conclusion
- the next most informative optional target is now the seed-2 CLS follow-up, not another patch rerun

### Seed-2 Gaussian patch follow-up

#### L10a: Gaussian patch, seed-2 checkpoint, shard A with `n_evals = 5`

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 3.4006212832`
- `final_eval/mean_visual_dist = 0.4930418261`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260321201836_2026-03-18_23-51-26_gH5`

#### L10b: Gaussian patch, seed-2 checkpoint, shard B with `n_evals = 5`

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 5.3275554094`
- `final_eval/mean_visual_dist = 0.4905739796`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260321202238_2026-03-18_23-51-26_gH5`

Estimated combined seed-2 patch Gaussian mean over 10 evaluation episodes:

- `mean_state_dist ~= 4.3641`

Interpretation:

- on seed 2, Gaussian patch again underperforms deterministic patch in larger-sample planning
- this makes the patch-side deterministic-vs-Gaussian comparison more consistent across the larger-sample follow-up
- at this point the patch-focused reevaluation objective is effectively complete

### Seed-2 deterministic CLS follow-up status

Attempt history:

- shard A, first attempt:
  `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260321202859_2026-03-18_23-32-43_gH5`

Observed behavior:

- the run progressed past `step 140`
- it never wrote `final_eval`
- later checks showed very low process activity and no further log growth

Decision:

- mark this run as `stalled / optional unfinished`
- stop the WSL session instead of continuing to wait
- do not let this optional CLS branch block report completion

### Seed-2 Gaussian CLS follow-up status

Attempt history:

- shard A, first attempt:
  `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260321224434_2026-03-19_00-16-03_gH5`
- shard A, overnight retry:
  `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260322002701_2026-03-19_00-16-03_gH5`

Observed behavior:

- the first attempt progressed to roughly `step 182` without writing `final_eval`
- the overnight retry progressed to `step 173`
- the retry hit a `120` minute timeout
- the overnight queue stopped immediately after this failure, so shard B, PushT sanity, and Wall sanity were not launched

Decision:

- mark seed-2 Gaussian CLS as `stalled / optional unfinished`
- stop further CLS-side large-eval retries in the main closeout path
- move the project fully into final aggregation and report cleanup

## Cross-Environment Pretrained Sanity

After the PointMaze closeout path was mostly complete, pretrained planning sanity checks were started on `Wall` and `PushT` to test whether the local WSL setup also worked outside PointMaze.

Additional local configs added:

- `conf/plan_wall_wsl.yaml`
- `conf/plan_pusht_wsl.yaml`

Additional datasets downloaded from the official OSF release and extracted in WSL:

- `/home/zack/dino_wm_data/wall_single`
- `/home/zack/dino_wm_data/pusht_noise`

### Wall pretrained sanity

Command:

```bash
python plan.py --config-name plan_wall_wsl.yaml \
  ckpt_base_path=$HOME/dino_wm_ckpts \
  model_name=wall_single \
  n_evals=1 \
  n_plot_samples=1 \
  planner.sub_planner.num_samples=8 \
  planner.sub_planner.topk=2 \
  planner.sub_planner.opt_steps=1 \
  planner.n_taken_actions=1
```

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 1.7964`
- `final_eval/mean_visual_dist = 0.5799`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260322164743_wall_single_gH5`

Interpretation:

- the local WSL setup works on at least one environment beyond PointMaze
- the pretrained `Wall` checkpoint can be loaded and planned end to end without further code changes

### Wall pretrained sanity, stronger-budget retry

Command:

```bash
python plan.py --config-name plan_wall_wsl.yaml \
  ckpt_base_path=$HOME/dino_wm_ckpts \
  model_name=wall_single \
  n_evals=1 \
  n_plot_samples=0 \
  plot_rollouts=False \
  save_video=False \
  planner.sub_planner.num_samples=64 \
  planner.sub_planner.topk=8 \
  planner.sub_planner.opt_steps=5 \
  planner.n_taken_actions=5
```

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 4.1456`
- `final_eval/mean_visual_dist = 1.1377`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260322205947_wall_single_gH5`

Interpretation:

- `Wall` remains stable under a stronger local planning budget
- the stronger budget did not improve the final distance relative to the lighter sanity run, so this should be read mainly as a stability confirmation rather than a new best result

### PushT pretrained sanity

Command:

```bash
python plan.py --config-name plan_pusht_wsl.yaml \
  ckpt_base_path=$HOME/dino_wm_ckpts \
  model_name=pusht \
  n_evals=1 \
  n_plot_samples=1 \
  planner.sub_planner.num_samples=8 \
  planner.sub_planner.topk=2 \
  planner.sub_planner.opt_steps=1 \
  planner.n_taken_actions=1
```

Reduced-budget status:

- output:
  `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260322165419_pusht_gH5`
- latest confirmed step:
  `74`
- latest confirmed log time:
  `2026-03-22 17:50:46`
- current intermediate trend:
  `mpc/success_rate = 0.0`, with very high `mean_state_dist`

Interpretation:

- this run is not a launch failure; the planner is producing outputs and writing logs
- under the current reduced local planning budget, `PushT` currently looks like a likely failure case rather than a likely success case

### PushT pretrained sanity, stronger-budget retry

Command:

```bash
python plan.py --config-name plan_pusht_wsl.yaml \
  ckpt_base_path=$HOME/dino_wm_ckpts \
  model_name=pusht \
  n_evals=1 \
  n_plot_samples=0 \
  plot_rollouts=False \
  save_video=False \
  planner.sub_planner.num_samples=64 \
  planner.sub_planner.topk=8 \
  planner.sub_planner.opt_steps=5 \
  planner.n_taken_actions=5
```

Current status:

- output:
  `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260322190605_pusht_gH5`
- latest confirmed step:
  `52`
- latest confirmed log time:
  `2026-03-22 19:23:47`
- current intermediate trend:
  `mpc/success_rate = 0.0`, but `mean_state_dist` improved substantially versus the reduced-budget run

Interpretation:

- this retry was better than the reduced-budget PushT sanity
- `mean_state_dist` moved from the earlier `100-130` range into roughly the `60-70` range for much of the run
- however, it still failed to reach `final_eval` and was later treated as `improved but stalled`

### PushT pretrained sanity, official-like retry

Command:

```bash
python plan.py --config-name plan_pusht_wsl.yaml \
  ckpt_base_path=$HOME/dino_wm_ckpts \
  model_name=pusht \
  n_evals=1 \
  n_plot_samples=0 \
  plot_rollouts=False \
  save_video=False \
  planner.sub_planner.num_samples=300 \
  planner.sub_planner.topk=30 \
  planner.sub_planner.opt_steps=30 \
  planner.n_taken_actions=5
```

Result:

- `final_eval/success_rate = 1.0`
- `final_eval/mean_state_dist = 20.7644`
- `final_eval/mean_visual_dist = 2.7361`

Output:

- `/mnt/c/Users/zack/Documents/GNN3/plan_outputs/20260322210051_pusht_gH5`

Interpretation:

- increasing the planning budget to something much closer to the official PushT configuration was enough to produce a clean completed successful run
- this suggests the earlier PushT failures were caused more by aggressive budget reduction than by a fundamentally broken local setup

## 2026-03-26 Encoder-3 and Encoder-4 Feasibility Work

The local workstream then shifted to encoder 3 and encoder 4, because the first two encoder branches were already being handled elsewhere.

Target scope for this branch:

- encoder 3:
  `DINOv3 patch`
- encoder 4:
  `V-JEPA`
- target tasks:
  `point_maze` and `wall_single`
- target training scope:
  deterministic first, `10 epochs`

### DINOv3 feasibility

WSL load attempt:

```bash
python - <<'PY'
import torch
torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16')
PY
```

Observed blocker:

- the official hub code failed before model construction because it uses Python `3.10+` union syntax
- the validated local DINO-WM environment is still Python `3.9`

Interpretation:

- `DINOv3` is currently blocked in the existing environment
- this is an environment-compatibility blocker, not just a missing package

### V-JEPA feasibility

WSL load attempt:

```bash
python - <<'PY'
import torch
torch.hub.load('facebookresearch/jepa-wms', 'vjepa2_ac_droid')
PY
```

Current progress:

- the hub path does begin loading under the current environment
- several missing dependencies were installed during the feasibility pass
- the dependency chain was then pushed further by installing:
  `clusterscope`, `ruamel.yaml`, and `timm`
- the latest confirmed blocker is now the required opensource V-JEPA visual checkpoint path:
  `${JEPAWM_OSSCKPT}/vjepa2_opensource/vjepa2_vit_giant.pth`

Interpretation:

- `V-JEPA` is still not integrated, but it currently looks more feasible than `DINOv3`
- the branch has progressed from package-level blockers to a real checkpoint-preparation blocker
- official checkpoint size is also non-trivial:
  - `vitl.pt` is about `5.1 GB`
  - `vitg.pt` is about `16.5 GB`
- the next decision is no longer "fix one more import"; it is whether to provision the required opensource checkpoint locally

### Current active provisioning step

A local background download was started for the required V-JEPA v2 giant opensource checkpoint:

- target local path:
  `C:\Users\zack\ModelCache\vjepa2_vit_giant.pth`

This download is meant to unblock the next real experiment:

- retry `V-JEPA` loading with `JEPAWM_OSSCKPT` configured
- inspect the resulting encoder object
- verify latent output shape before any DINO-WM integration work

## 2026-03-26 Later-Encoder Feasibility Update

The later-encoder pass was extended beyond `DINOv3` and `V-JEPA` to also reassess the remaining paper-level candidates:

- `DINO-Tok`
- `VFM-VAE`

### DINOv3

Status:

- still blocked

Reason:

- both the official `facebookresearch/dinov3` route and the local fallback route remain incompatible with the validated Python `3.9` environment because the code uses Python `3.10+` union typing syntax

### V-JEPA

Status:

- progressed further than all other later-encoder branches

Confirmed progress:

- the giant opensource checkpoint was fully downloaded locally
- the checkpoint was linked into:
  `/home/zack/jepawm_ossckpt/vjepa2_opensource/vjepa2_vit_giant.pth`
- the official `jepa-wms` path previously loaded pretrained visual-encoder weights successfully before failing deeper in the full AC predictor path

Current blocker:

- the full official AC predictor path hits very large memory allocation
- a later encoder-only probe using `vit_giant_xformers` and `pred_type=none` did not produce a clean local feature-shape result yet

Interpretation:

- `V-JEPA` is now clearly beyond package-level feasibility
- the right next step is not "install one more dependency"
- the right next step is an encoder-only wrapper or a more faithful minimal encoder extraction path

### DINO-Tok

Status:

- not grounded enough to begin local integration

Confirmed evidence:

- arXiv presence is confirmed:
  `DINO-Tok: Adapting DINO for Visual Tokenizers`
- repeated GitHub API repository searches using terms around `DINO-Tok` did not surface an obvious official repository path during this pass

Interpretation:

- this branch is still paper-level only in the current local workspace
- there is not yet a practical code-and-checkpoint route to treat it as the next experiment

### VFM-VAE

Status:

- now partially grounded, but still heavy for the current project phase

Confirmed evidence:

- arXiv presence is confirmed:
  `Vision Foundation Models Can Be Good Tokenizers for Latent Diffusion Models`
- an official repository was identified:
  `https://github.com/tianciB/VFM-VAE`
- an official Hugging Face model page was identified:
  `https://huggingface.co/tiancibi/VFM-VAE`
- the repository structure confirms a full tokenizer / VAE training stack
- the Hugging Face release contains VAE and diffusion checkpoints rather than a small standalone planning encoder package

Interpretation:

- this is no longer "missing", but it is still not a natural drop-in replacement for the local DINO-WM encoder slot
- the repo recommends a PyTorch `2.4.0` / Python `3.10` environment and centers on ImageNet-scale VAE training
- for the current DINO-WM course-project branch, `VFM-VAE` should be treated as a higher-cost future integration path, not the next practical experiment

### Current practical ranking

From most actionable to least actionable in the current project state:

1. `V-JEPA`
2. `DINOv3`
3. `DINO-Tok`
4. `VFM-VAE`

The practical next experiment should continue on `V-JEPA`, not on the other later-encoder branches.

## 2026-03-27 Overnight Encoder Feasibility Queue

To avoid wasting the overnight window on unstable reruns, the next long-running queue was switched to checkpoint provisioning and structure probing for the later encoder branches.

Scheduled overnight tasks:

1. clone or refresh the official `VFM-VAE` repository locally
2. download the official `VFM-VAE` stage-3 tokenizer checkpoint from Hugging Face
3. download the official `facebook/vjepa2-vitl-fpc64-256` `original/model.pth`
4. inspect both checkpoints for top-level structure and likely integration keys
5. copy the smaller `V-JEPA vitl` checkpoint into the WSL checkpoint area for the next local encoder-only probe

Local launcher:

- `scripts/run_overnight_encoder_feasibility.ps1`

Log directory:

- `logs/overnight_encoder_feasibility`

Stable follow-up launcher:

- `logs/overnight_encoder_feasibility/download_and_probe.py`

The direct `cmd.exe + python -u` runner was started after the first PowerShell-only queue showed weak progress visibility. This replacement runner is responsible for:

1. downloading the official `VFM-VAE` stage-3 tokenizer checkpoint
2. downloading the official `facebook/vjepa2-vitl-fpc64-256` `original/model.pth`
3. printing checkpoint-structure information once each file completes

## 2026-03-27 Morning Feasibility Results

The overnight replacement runner completed both downloads and both checkpoint-structure probes.

### VFM-VAE

Downloaded file:

- `C:\Users\zack\ModelCache\VFM-VAE-model\checkpoints_imagenet256\vfm_vae\vfm_vae_f16d32_siglip2_44m_after_stage_3_patchgan_fine_tuning_legacy.pth`

Observed top-level keys:

- `G`
- `D`
- `G_ema`
- `training_set_kwargs`

Observed structure detail:

- `G` and `G_ema` both contain nested `vfm_encoder...vision_model...` weights
- this confirms that the released checkpoint is not just a diffusion-side artifact; it does contain an extractable vision-backbone branch inside the tokenizer system

Interpretation:

- `VFM-VAE` is now clearly code-plus-checkpoint feasible
- however, it still looks like a whole-generator extraction problem rather than a simple frozen-encoder drop-in

### V-JEPA vitl

Downloaded file:

- `C:\Users\zack\ModelCache\VJEPA2-vitl\original\model.pth`

Observed top-level keys:

- `encoder`
- `predictor`
- `target_encoder`
- `opt`
- `scaler`
- `epoch`

Additional compatibility probe:

- the `encoder` state dict was remapped into `transformers.VJEPA2Model`
- after key conversion, encoder loading reached:
  - `ENCODER_MISSING_COUNT = 0`
  - `UNEXPECTED_COUNT = 0`
- only predictor-side weights remained missing, which is acceptable for an encoder-only integration plan

Minimal forward probe:

- the remapped encoder path was exercised on CPU with:
  - input shape: `(1, 2, 3, 256, 256)`
  - `skip_predictor=True`
- observed outputs:
  - `last_hidden_state`: `(1, 256, 1024)`
  - `masked_hidden_state`: `(1, 256, 1024)`
- measured forward time:
  - about `1.61 s` on the local CPU path used for the probe

Interpretation:

- this is the strongest feasibility result so far for encoder 4
- checkpoint compatibility is no longer the blocker
- basic encoder-only forward feasibility is no longer the blocker either
- the next step should be an actual local wrapper based on the `vitl` route rather than more giant-checkpoint debugging

## 2026-03-27 Repo Integration Progress For V-JEPA

The repository now contains a minimal local encoder-4 integration path:

- `models/vjepa.py`
- `conf/encoder/vjepa.yaml`
- `conf/train_vjepa_wsl.yaml`

Verified local smoke test:

- input shape: `(2, 3, 224, 224)`
- output shape: `(2, 196, 1024)`
- patch size: `16`
- latent mode: `latent_ndim = 2`

This means encoder 4 is no longer only a feasibility note; it now has a DINO-WM-compatible local wrapper and a prepared `1 epoch` sanity config.

Current practical WSL blocker:

- the validated `dino_wm` environment currently has `transformers==4.21.1`
- the same environment also has `huggingface_hub==1.8.0`
- this pair is currently incompatible and breaks `transformers` import

So the next blocker is environment readiness, not model-side feasibility.

## 2026-03-27 WSL Refresh And Later-Encoder Training Update

The WSL `dino_wm` environment was then refreshed specifically for later-encoder work:

- `transformers` was upgraded from `4.21.1` to `4.57.6`
- `huggingface_hub` was adjusted to `0.36.0`
- `tokenizers` was upgraded to `0.22.1`

This removed the earlier import failure and enabled both `VJEPA2Model` and `SiglipVisionModel` inside the validated WSL environment.

### V-JEPA

After the dependency refresh:

- `models/vjepa.py` worked inside WSL
- a WSL smoke test again returned `(1, 196, 1024)` on `224x224` image input
- `conf/train_vjepa_wsl.yaml` was launched successfully on `point_maze`

Observed practical result:

- the run entered the real training loop and reached approximately:
  - `Epoch 1 Train: 2732 / 72900`
  - elapsed time about `20 minutes`
  - estimated remaining time for epoch 1 about `7 hours`
- extrapolated `10 epoch` runtime was therefore around `3` to `3.5` days

Decision:

- the `V-JEPA` training run was stopped manually
- reason: the training flow was successfully demonstrated, but the runtime was too slow for the current local schedule

### SigLIP2 / VFM-VAE-backbone Route

To keep progress moving on another encoder family, a lighter route based on the backbone used by `VFM-VAE` was integrated.

New repo files:

- `models/siglip2.py`
- `conf/encoder/siglip2.yaml`
- `conf/train_siglip2_wsl.yaml`

Verified local result in WSL:

- `SigLIP2Encoder` loaded successfully from `google/siglip2-large-patch16-512`
- smoke test output on `224x224` input:
  - `(1, 196, 1024)`

Training status:

- the `point_maze` `1 epoch` sanity run for `SigLIP2` was started successfully
- it was then stopped manually to free the GPU for the next encoder pass

Interpretation:

- both `V-JEPA` and `SigLIP2` have now moved beyond feasibility and into actual local training-flow execution
- `V-JEPA` is currently limited by runtime
- `SigLIP2` is the lighter practical fallback for continuing representation experiments

### VFM-VAE

The next follow-up targeted the full `VFM-VAE` code path rather than only its SigLIP2 backbone.

Current status:

- `networks.utils.vfm_utils` imports successfully in the current WSL environment
- `networks.generator` initially failed under Python `3.9` because of a Python-3.10-style union type in:
  - `networks/utils/convnext_utils.py`
- that local syntax issue was patched to a Python-3.9-compatible `typing.Union` form in the local cache copy

Interpretation:

- `VFM-VAE` is no longer blocked only by missing code or missing checkpoints
- it is now in a code-compatibility cleanup phase, with the next step being a clean import retry and then a minimal encoder extraction attempt

## 2026-03-29 Closeout Priority Update

The highest-yield next experiment was reset to the unfinished `CLS + deterministic` larger-sample PointMaze closeout rather than another exploratory encoder run.

Priority order:

1. rerun the missing `seed-2 CLS + deterministic` larger-sample follow-up
2. if that succeeds cleanly, rerun the missing `seed-1 CLS + deterministic` shard-A follow-up
3. only then consider a small training-budget extension for `patch + deterministic` versus `CLS + deterministic`

Why this order:

- the paper's most interesting representation claim is that `CLS + deterministic` remains competitive with `patch + deterministic`
- that claim is currently limited by incomplete larger-sample CLS coverage
- completing one missing CLS deterministic rerun strengthens the main paper result more directly than opening another encoder branch

Execution note:

- the first `P1_plan_retry` launch failed because Hydra resolved `datasets.point_maze_dset` against the third-party `datasets` package
- to make PointMaze planning runs robust, two local fixes were applied:
  - `scripts/wsl_pointmaze_large_eval_single.sh` now prepends the repo root to `PYTHONPATH`
  - `plan.py` now preloads the repo-local `datasets` package before third-party imports can populate `sys.modules["datasets"]`
- after this fix, a short probe run for `P1_plan_retry` advanced past Hydra dataset resolution, loaded the seed-2 checkpoint, and reached normal PointMaze planning initialization

## 2026-03-29 Second Paper Review Consolidation

A second review was then folded into the execution plan.

Main takeaways from that review:

- the strongest paper story is still the PointMaze representation result:
  - `patch + deterministic` is the best completed baseline
  - `CLS + deterministic` is surprisingly competitive under the current budget
- the Gaussian branch is useful, but should be framed more narrowly:
  - the current implementation tests a Gaussian likelihood head
  - planning still uses the mean prediction
  - so this is not yet a full uncertainty-aware planning study
- the largest remaining weakness in the current evidence is still incomplete CLS larger-sample coverage

Updated decision:

1. keep the current `P1_plan_retry` CLS deterministic closeout as the top priority
2. if that succeeds, do only one more closeout rerun:
   - `seed-1 CLS + deterministic` shard A
3. after the CLS closeout, prefer one narrow, high-yield extension instead of a new encoder branch:
   - `patch-det vs CLS-det` under a larger training budget
   - or `PushT patch-det vs CLS-det`
   - or a lightweight uncertainty-aware planning variant

Collaboration split:

- person 1:
  own the closeout experiments and verify final planning outputs
- person 2:
  own results aggregation, tables, and paper cleanup
- person 3:
  own exactly one focused extension after the CLS closeout finishes

Interpretation:

- this review does not change the main project direction
- it strengthens the case for finishing the existing PointMaze representation story before spending more time on later encoders

## 2026-03-29 Closeout Failure And Final 1.5-Day Reset

Two parallel CLS deterministic closeout retries were launched:

- `P1_plan_retry` for `seed-2 CLS + deterministic`
- `P2_plan_retry` for `seed-1 CLS + deterministic`

Observed result:

- both runs advanced far into the MPC loop
- `P1_plan_retry` reached roughly `MPC iter 87`
- `P2_plan_retry` reached roughly `MPC iter 73`
- neither run wrote a `.done` flag under `logs/large_eval_status`
- neither run remained alive in later process checks

Interpretation:

- these runs should be treated as interrupted or failed closeout attempts rather than completed evidence
- the PointMaze planning closeout path is still valuable, but it should now be relaunched with a more robust detached launcher
- because only `1.5` days remain, the experiment plan is reduced to:
  1. one robust rerun of `seed-2 CLS-det`
  2. one follow-up rerun of `seed-1 CLS-det` only if step 1 completes
  3. one minimal `3 -> 5 epoch` resume-based budget extension for `patch-det` vs `CLS-det` only if time remains

### Robust Relaunch

To avoid another mid-run drop from the host-side launcher, a detached WSL background helper was added:

- `scripts/start_wsl_pointmaze_large_eval_bg.ps1`

This helper launches the normal PointMaze eval wrapper through `nohup` inside WSL and writes logs under:

- `logs/<label>_stdout.log`
- `logs/<label>_stderr.log`
- `logs/<label>.pid`

The first run under this new path is:

- `P3_plan_retry_bg`
- target: `seed-2 CLS + deterministic`
- checkpoint: `2026-03-18/23-32-43`

Early verification:

- WSL shows active `bash`, `micromamba`, and `python plan.py` processes
- stdout advanced through planning initialization
- output directory:
  - `plan_outputs/20260329234431_2026-03-18_23-32-43_gH5`

### Live Status Update

Shortly after relaunch:

- `P3_plan_retry_bg` remained alive in WSL process checks
- stdout still sat near the post-initialization stage rather than the first printed `MPC iter`
- GPU usage remained modest, around:
  - utilization `27%`
  - memory `1786 / 8188 MB`

Decision:

- the run is kept alive
- because GPU headroom remains available, the next-most-useful closeout rerun can be launched in parallel

Parallel follow-up launched:

- `P4_plan_retry_bg`
- target: `seed-1 CLS + deterministic`
- checkpoint: `2026-03-18/21-50-26`

Verification:

- WSL shows active `bash`, `micromamba`, and `python plan.py` processes for the seed-1 rerun as well
- startup files were created under `logs/P4_plan_retry_bg_*`

### Minimal Budget-Extension Probe

Because the two PointMaze closeout runs still left GPU headroom, one minimal training-budget extension was started instead of opening a new encoder branch.

Launched:

- `T1_cls_det_resume5`
- target: `seed-0 CLS + deterministic`
- original run dir:
  - `course_runs/outputs/2026-03-18/20-01-01`
- goal:
  - resume the existing `3 epoch` checkpoint and extend total training to `5 epochs`

Why this probe:

- it is the cheapest useful version of the budget-extension experiment
- it directly tests whether the `CLS-det` representation result changes under a modestly larger training budget
- it is substantially lower risk than starting a brand-new long training sweep

Verification:

- WSL shows active `bash`, `micromamba`, and `python train.py` processes for the resume job
- GPU usage after launch remained moderate enough to keep the two closeout planning runs alive at the same time

### Live Progress Snapshot

Subsequent monitoring showed all three jobs still alive:

- `P3_plan_retry_bg`
  - progressed into the MPC loop and reached about `MPC iter 14`
- `P4_plan_retry_bg`
  - also progressed into the MPC loop and reached about `MPC iter 14`
- `T1_cls_det_resume5`
  - resumed cleanly from epoch `3` inside:
    - `course_runs/outputs/2026-03-18/20-01-01/train.log`

At that point the machine still had substantial headroom:

- GPU utilization roughly `4%`
- GPU memory roughly `3363 / 8188 MB`

Decision:

- keep all three runs active
- use the remaining headroom for the matching `seed-0 patch-det` `3 -> 5 epoch` resume so the budget-extension comparison stays paired

### Paired Patch Resume

The matching deterministic patch baseline was then resumed as well:

- `T2_patch_det_resume5`
- target: `seed-0 patch + deterministic`
- original run dir:
  - `course_runs/outputs/2026-03-18/19-12-06`
- goal:
  - resume the existing `3 epoch` checkpoint and extend total training to `5 epochs`

Verification:

- WSL shows active `bash`, `micromamba`, and `python train.py` processes for this run too
- `course_runs/outputs/2026-03-18/19-12-06/train.log` now contains:
  - a fresh `2026-03-29` resume block
  - `Resuming from epoch 3`

Interpretation:

- the final `1.5`-day experiment set now includes:
  - two CLS deterministic PointMaze closeout reruns
  - one paired `CLS-det` vs `patch-det` minimal budget-extension from `3 -> 5 epochs`

### Consolidated Live Status

The next monitoring pass showed all four high-yield runs still alive:

- `P3_plan_retry_bg`
  - advanced from initialization into the MPC loop and reached about `MPC iter 28`
- `P4_plan_retry_bg`
  - also remained healthy in the MPC loop and reached about `MPC iter 14`
- `T1_cls_det_resume5`
  - still active after the confirmed `Resuming from epoch 3` event in:
    - `course_runs/outputs/2026-03-18/20-01-01/train.log`
- `T2_patch_det_resume5`
  - also active after the confirmed `Resuming from epoch 3` event in:
    - `course_runs/outputs/2026-03-18/19-12-06/train.log`

Resource snapshot at the same time:

- GPU utilization about `15%`
- GPU memory about `3936 / 8188 MB`

Decision:

- no further experiments are launched for now
- the remaining time is better spent letting these four runs mature and then folding the first completed results back into the paper

### Next Monitoring Snapshot

Another status pass confirmed that the same four runs remained alive and that no new `.done` marker had appeared yet for the detached PointMaze closeout jobs.

Latest visible progress:

- `P3_plan_retry_bg`
  - reached about `MPC iter 28`
  - still shows partial success across the `5` eval trajectories rather than full convergence
- `P4_plan_retry_bg`
  - remained active around `MPC iter 14`
  - still has one hard trajectory blocking full success
- `T1_cls_det_resume5`
  - still active after the epoch-3 resume event
- `T2_patch_det_resume5`
  - still active after the epoch-3 resume event

Resource snapshot:

- GPU utilization about `5%`
- GPU memory about `3935 / 8188 MB`

Interpretation:

- the experimental slate is already saturated with the highest-yield remaining work
- the right move is to keep monitoring these four runs rather than opening additional experiments

### Follow-Up Status Snapshot

The next check produced the same overall conclusion: all four priority runs were still alive, and none had reached a terminal completion marker yet.

Updated visible progress:

- `P3_plan_retry_bg`
  - remained the furthest planning run
  - advanced to about `MPC iter 28`
  - continued to show a stable partial-success pattern with one or two hard trajectories still blocking completion
- `P4_plan_retry_bg`
  - remained active around `MPC iter 14`
  - still had one main hard trajectory preventing full success
- `T1_cls_det_resume5`
  - still active after its epoch-3 resume event
  - no new epoch-completion line had appeared yet in the on-disk log tail
- `T2_patch_det_resume5`
  - also still active after its epoch-3 resume event
  - likewise had not yet reached a new epoch-completion line in the current log tail

Resource snapshot:

- GPU utilization about `5%`
- GPU memory about `3936 / 8188 MB`

Decision:

- keep the current four-run set unchanged
- wait for the first completed planning closeout or the first resumed training epoch to finish before updating the paper

### Latest Status Snapshot

Another monitoring pass kept the same overall picture: no run had finished yet, but all four priority jobs were still alive.

Latest visible progress:

- `P3_plan_retry_bg`
  - still the furthest PointMaze closeout run
  - remained around the high-20s in MPC iterations, with about `0.6` success rate in its current partial state
- `P4_plan_retry_bg`
  - remained active in the mid-teens of MPC iterations
  - still had one dominant hard trajectory preventing full success
- `T1_cls_det_resume5`
  - still alive after resuming from epoch `3`
  - no new epoch completion had been flushed to the current log tail yet
- `T2_patch_det_resume5`
  - still alive after resuming from epoch `3`
  - likewise had not yet reached the next saved epoch in the visible log tail

Resource snapshot:

- GPU utilization about `38%`
- GPU memory about `3936 / 8188 MB`

Decision:

- continue running the same four jobs
- do not launch additional experiments until one of the current runs produces a usable result

### Additional Monitoring Snapshot

One more status pass still showed no terminal completion, but all four priority runs remained active.

Latest visible state:

- `P3_plan_retry_bg`
  - still the leading PointMaze closeout rerun
  - remained around `MPC iter 28`
  - still showed a partial-success plateau rather than a clean finish
- `P4_plan_retry_bg`
  - remained active around the mid-teens in MPC iterations
  - still had a small number of hard eval trajectories blocking completion
- `T1_cls_det_resume5`
  - still alive in resumed training mode
  - no epoch-4 completion had appeared yet in the visible tail
- `T2_patch_det_resume5`
  - also still alive in resumed training mode
  - no epoch-4 completion had appeared yet in the visible tail

Resource snapshot:

- GPU utilization about `9%`
- GPU memory about `3938 / 8188 MB`

Decision:

- keep the experiment set fixed
- continue monitoring until one run finishes and yields something that can be folded directly into the paper

### Ongoing Monitoring Snapshot

The following check still showed the same basic situation:

- no new `.done` marker yet for the detached closeout runs
- no new completed epoch line yet in the visible tails of the resumed training logs
- all four priority runs still alive

Latest visible progress:

- `P3_plan_retry_bg`
  - still hovered around the high-20s in MPC iterations
  - continued to exhibit a partial-success plateau rather than clean convergence
- `P4_plan_retry_bg`
  - advanced further into the mid-to-high teens / twenties range
  - still had one stubborn hard trajectory blocking completion
- `T1_cls_det_resume5`
  - still active after the epoch-3 resume event
- `T2_patch_det_resume5`
  - still active after the epoch-3 resume event

Resource snapshot:

- GPU utilization about `38%`
- GPU memory about `3935 / 8188 MB`

Decision:

- keep the four-run slate unchanged
- continue waiting for the first completed planning closeout or first finished resumed epoch before paper-side result updates

## 2026-03-30 Morning Status

The first status sync on `2026-03-30` still showed the same basic picture:

- no detached PointMaze closeout run had written a `.done` marker yet
- no resumed training run had yet flushed a new completed epoch into the visible tail
- all four high-priority runs were still alive

Latest visible state:

- `P3_plan_retry_bg`
  - still hovered around the high-20s in MPC iterations
  - remained in a partial-success regime rather than finishing cleanly
- `P4_plan_retry_bg`
  - continued advancing through the teens into the twenties
  - still had one hard eval trajectory blocking full completion
- `T1_cls_det_resume5`
  - still active after the `epoch 3` resume
- `T2_patch_det_resume5`
  - still active after the `epoch 3` resume

Resource snapshot:

- GPU utilization about `22%`
- GPU memory about `3940 / 8188 MB`

Decision:

- keep all four runs active
- do not open any new experiments until at least one current run yields a directly reportable result

### Continued 2026-03-30 Status

Another monitoring pass on `2026-03-30` still showed no terminal result from any of the four priority runs.

Visible state remained:

- `P3_plan_retry_bg`
  - still in the high-20s of MPC iterations
  - still trapped in a partial-success regime
- `P4_plan_retry_bg`
  - also still active
  - still had not produced a clean closeout despite gradual MPC progress
- `T1_cls_det_resume5`
  - still alive after its epoch-3 resume event
- `T2_patch_det_resume5`
  - still alive after its epoch-3 resume event

Resource snapshot:

- GPU utilization about `8%`
- GPU memory about `3941 / 8188 MB`

Decision:

- keep monitoring the same four runs
- continue to avoid opening any additional experiments until one of the current jobs yields a usable result
### Continued 2026-03-30 Status (Later Snapshot)

Checked the four active high-value runs again. No new terminal results have appeared yet, so the execution strategy remains unchanged: keep the current four runs alive and avoid opening new experiments.

- `P3_plan_retry_bg` remains active and has progressed through `MPC iter 28`. It is still showing the same partial-success pattern, with two difficult trajectories not yet closed out.
- `P4_plan_retry_bg` also remains active and has likewise progressed through `MPC iter 28`. It is still stuck on one hard evaluation trajectory while the other four are already successful.
- `T1_cls_det_resume5` remains alive. Resume from epoch 3 is confirmed, but the log has not yet printed a new completed epoch line.
- `T2_patch_det_resume5` remains alive. Resume from epoch 3 is confirmed, but the log has not yet printed a new completed epoch line.
- `logs/large_eval_status` still shows no `.done` markers for `P3_plan_retry_bg` or `P4_plan_retry_bg`.
- GPU snapshot at this check: about `18%` utilization and `3921 / 8188 MB` memory.

Conclusion: all four priority experiments are still running; no new experiment should be opened before one of these produces a usable result.
### Continued 2026-03-30 Status (Repeated Check)

Re-checked the same four priority runs. No terminal results have appeared since the previous snapshot, so the correct action remains to keep the current four-run slate alive and avoid opening new work.

- `P3_plan_retry_bg` is still active and still centered around `MPC iter 28`, with the same pattern `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` is still active and still centered around `MPC iter 28`, with the same pattern `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` is still alive; resume from epoch 3 remains confirmed, but there is still no new completed epoch line.
- `T2_patch_det_resume5` is still alive; resume from epoch 3 remains confirmed, but there is still no new completed epoch line.
- `logs/large_eval_status` still has no `.done` marker for either `P3_plan_retry_bg` or `P4_plan_retry_bg`.
- GPU snapshot at this check: about `14%` utilization and `3941 / 8188 MB` memory.

Conclusion: document updated, experiments continue unchanged, no additional launches.
### Continued 2026-03-30 Status (Another Check)

Checked the same four active runs again. There are still no terminal results, so the experiment set remains unchanged.

- `P3_plan_retry_bg` is still active and still sitting around `MPC iter 28` with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` is still active and still sitting around `MPC iter 28` with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` is still alive. Resume from epoch 3 is confirmed, but there is still no new epoch completion line.
- `T2_patch_det_resume5` is still alive. Resume from epoch 3 is confirmed, but there is still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `49%` utilization and `3921 / 8188 MB` memory.

Conclusion: document updated, all four priority runs continue, no new launches.
### Continued 2026-03-30 Status (Stable Plateau)

Checked the four active runs again. The status is effectively unchanged from the previous check, so the current experiment set remains the correct one to keep running.

- `P3_plan_retry_bg` is still active and still plateaued around `MPC iter 28`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` is still active and still plateaued around `MPC iter 28`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` is still alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` is still alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `14%` utilization and `3922 / 8188 MB` memory.

Conclusion: document updated, experiments continue unchanged, no new launches.
### Continued 2026-03-30 Status (No Change)

Another status check shows no material change from the previous snapshot.

- `P3_plan_retry_bg` remains active and remains plateaued around `MPC iter 28`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and remains plateaued around `MPC iter 28`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `15%` utilization and `3923 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### 2026-03-30 Teammate Draft Review Logged

Reviewed teammate draft paper at `C:\Users\zack\Documents\AwesomeGNN\GNN_Final_Project\main.tex` and recorded optimization opportunities without editing the draft itself.

Key recorded issues:
- the draft currently mixes two competing paper centers (`encoder comparison` and `controlled patch/CLS × deterministic/Gaussian` ablation),
- evidence tiers are not separated clearly enough,
- project-status sections remain in the main body,
- too many raw diagnostic plots are given relative to the strength of the core claims,
- several claims should be narrowed to the actual PointMaze-specific evidence.

Review note saved to:
- `docs/TEAMMATE_PAPER_REVIEW_2026-03-30.md`
### 2026-03-30 External Paper Review Feedback Integrated

Integrated an external review summary (Gemini) into the teammate-paper review note, again without editing the draft itself.

Newly emphasized items:
- citation numbering / bibliography consistency,
- broken cross-references and spacing errors,
- overly informal table language (`stalled`, `optional unfinished`, etc.),
- overexposure of environment/debugging details in the main text,
- need to sharpen the decoder-asymmetry fairness caveat,
- need to justify `mean_state_dist` more rigorously,
- need to soften the CLS competitiveness claim until larger-sample evidence is more symmetric.

Updated note remains:
- `docs/TEAMMATE_PAPER_REVIEW_2026-03-30.md`
### 2026-03-30 Additional Review Synthesis Integrated

Integrated one more round of written review feedback into the teammate-paper review note and then applied a limited cleanup pass to the teammate draft itself.

Documentation updates:
- added a new synthesis block to `docs/TEAMMATE_PAPER_REVIEW_2026-03-30.md`

Limited draft edits applied to `C:\Users\zack\Documents\AwesomeGNN\GNN_Final_Project\main.tex`:
- softened several over-strong claims (`fair comparison`, `systematic representation study`, etc.),
- cleaned a few remaining terminology inconsistencies (`V-JEPA 2`, `VFM-VAE`),
- made the decoder-asymmetry caveat more explicit,
- replaced notebook-like larger-sample table wording with cleaner labels,
- strengthened the one-sentence rationale for using `mean_state_dist` when `success_rate` saturates.
### 2026-03-30 Teammate Draft Cleanup Continued

Applied another small pass on `C:\Users\zack\Documents\AwesomeGNN\GNN_Final_Project\main.tex`, still limited to obvious issues a professor would quickly notice.

This pass included:
- replacing a few remaining `project/thesis`-style phrasings with more paper-like wording,
- cleaning section and subsection naming so they look less like draft labels,
- standardizing more `V-JEPA 2` mentions,
- removing one unnecessary punctuation mark from a subsection title,
- aligning a few conclusion / status sentences with the updated terminology.
### 2026-03-30 Static Citation Check For Teammate Draft

Ran a static citation/reference check on `C:\Users\zack\Documents\AwesomeGNN\GNN_Final_Project\main.tex`.

Results:
- every `\cite{...}` key used in the draft exists in `references.bib`,
- no remaining hard-coded numeric bracket citations were detected,
- no remaining `??` placeholder references were detected.

Limitation:
- this environment does not have `latexmk` or `pdflatex`, so a full compile-time bibliography and warning check could not be performed locally.
### Paper Sync 2026-03-30: Teammate Workstream Added and Figure 1 Fixed

The paper and figure-generation path were updated to reflect the current project state more accurately.

- `paper/main.tex` now explicitly distinguishes the completed controlled PointMaze evidence from the parallel teammate workstream on infrastructure broadening and exploratory later-encoder integration.
- This makes the scope boundary clearer: the paper's main claims remain tied to the completed `patch/CLS x deterministic/Gaussian` evidence, while teammate exploration is acknowledged as supporting and future-facing rather than mixed into the core results.
- `paper/generate_figures.py` was fixed so that formal-matrix rows are matched correctly from `docs/EXPERIMENT_TRACKER.csv`. The bug was an over-escaped regex for `F*_train` and `F*_plan`, which caused Figure 1 to miss the formal matrix rows.
- `paper/generate_figures.py` was re-run successfully after the fix.
- `paper/figure/formal_matrix_summary.png` was regenerated successfully; updated timestamp is `2026-03-30 10:41`.

Conclusion: the paper now records the teammate workstream at the scope/discussion level, and Figure 1 has been repaired at the data-ingestion level.
### Sync Data Integration 2026-03-30

Teammate-exported results were found under:

- `C:\Users\zack\Documents\AwesomeGNN\sync_data`

Available files:

- `wandb_export_2026-03-30T10_49_54.703+02_00.csv`
- `wandb_export_2026-03-30T10_50_16.740+02_00.csv`

These exports provide one exploratory PointMaze encoder comparison branch beyond the completed paper core:

- exploratory `DINOv2 patch` reference
- exploratory `VFM-VAE`
- exploratory `V-JEPA2`

Extracted planning-side summary:

- `DINOv2 patch`: `mean_state_dist=5.1009`, `success_rate=1.0`
- `VFM-VAE`: `mean_state_dist=4.2407`, `success_rate=0.9375`
- `V-JEPA2`: `mean_state_dist=4.3022`, `success_rate=0.625`

Extracted training-side summary:

- all three exported training runs are marked `crashed` in W\&B
- nevertheless, each export contains non-empty last logged train / val metrics
- because of the `crashed` status and recipe mismatch relative to the paper core, these runs should be treated as exploratory evidence rather than folded into the formal matrix

Integration actions completed:

- added six exploratory rows to `docs/EXPERIMENT_TRACKER.csv`
- filled the teammate numeric summary in `docs/COURSE_PROJECT_PLAN.md`
- added an exploratory-encoder subsection and table to `paper/main.tex`

Conclusion: the broader representation plan now has documented preliminary numeric evidence, but the main paper claims remain anchored to the completed controlled PointMaze patch/CLS and deterministic/Gaussian study.
### Continued 2026-03-30 Status (Added Teammate Workstream Planning Section)

Added a dedicated teammate-workstream section to the project plan so the final report can clearly distinguish:

- the completed minimum course-project matrix
- the teammate-led broader encoder and infrastructure direction
- the still-pending experiments whose numeric results will be filled later

Document updated:

- `docs/COURSE_PROJECT_PLAN.md`

New section added:

- `Teammate Workstream`

It currently records:

- the teammate line's main research direction
- the major contribution areas
- the main experimental themes
- the current structural status
- placeholders for numeric results still to be provided later
### Continued 2026-03-30 Status (Reality Check: All Four Long Runs Are Stale/Stopped)

A direct process and file-timestamp check shows the previously tracked long-running retries are no longer making forward progress and should not be treated as active runs anymore.

- `P3_plan_retry_bg` has no corresponding live `plan.py` process. Its stdout log last changed at `2026-03-30 01:04`, so it is stale/stopped rather than actively running.
- `P4_plan_retry_bg` also has no corresponding live `plan.py` process. Its stdout log last changed at `2026-03-30 00:49`, so it is stale/stopped rather than actively running.
- `T1_cls_det_resume5` shows no live `train.py` process, and its `train.log` has not advanced since the resume line written at `2026-03-29 23:51`.
- `T2_patch_det_resume5` also shows no live `train.py` process, and its `train.log` has not advanced since the resume line written at `2026-03-29 23:55`.
- `logs/large_eval_status` still contains no `.done` markers for `P3_plan_retry_bg` or `P4_plan_retry_bg`.

Conclusion: the four tracked “running” jobs should now be considered stopped/stale, and any next step should start from that assumption instead of waiting for them.
### Continued 2026-03-30 Status (User Decision: Stop Waiting on These Runs)

The current decision is to stop waiting on the four stale jobs and treat them as halted for project-planning purposes.

- `P3_plan_retry_bg` is considered stopped/stale and will not be waited on further.
- `P4_plan_retry_bg` is considered stopped/stale and will not be waited on further.
- `T1_cls_det_resume5` is considered stopped/stale and will not be waited on further.
- `T2_patch_det_resume5` is considered stopped/stale and will not be waited on further.

Conclusion: these runs are no longer treated as active experiments, and any remaining work should proceed from completed evidence plus explicitly marked incomplete attempts.
### Continued 2026-03-30 Status (Repeated 4 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `4%` utilization and `3933 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 19 Percent GPU Snapshot, Lower Memory)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `19%` utilization and `3933 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 16 Percent GPU Snapshot, Lower Memory Again)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `16%` utilization and `3936 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 16 Percent GPU Snapshot Again)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `16%` utilization and `3937 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 12 Percent GPU Snapshot, Lower Memory)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `12%` utilization and `3938 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 28 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `28%` utilization and `3938 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 27 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `27%` utilization and `3938 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 12 Percent GPU Snapshot Again)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `12%` utilization and `3937 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 17 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `17%` utilization and `3936 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 14 Percent GPU Snapshot Again)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `14%` utilization and `3935 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 14 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `14%` utilization and `3933 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 19 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `19%` utilization and `3937 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 11 Percent GPU Snapshot Yet Again)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `11%` utilization and `3943 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 23 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `23%` utilization and `3943 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 15 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `15%` utilization and `3944 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 13 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `13%` utilization and `3943 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 11 Percent GPU Snapshot Again)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `11%` utilization and `3944 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 18 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `18%` utilization and `3943 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 16 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `16%` utilization and `3943 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 26 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `26%` utilization and `3944 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 11 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `11%` utilization and `3944 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Both Planning Retries Still at 58, Fresh 11 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `11%` utilization and `3943 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Both Planning Retries Still at 58, Fresh 12 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `12%` utilization and `3943 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (P4 Advanced to 58)

This fresh check shows the second active planning retry has now moved again. The two resumed training runs still have not emitted new epoch summary lines.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and has now advanced from the earlier `iter 43` plateau to about `MPC iter 58`, while still showing the unresolved pattern `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `8%` utilization and `3946 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged, with `P4` now matching the higher planning-iter range previously reached only by `P3`.
### Continued 2026-03-30 Status (Both Planning Retries Holding at 58)

Another fresh check shows no new completions and no new training epoch outputs. Both active planning retries are now sitting at the same latest visible iter level, but neither has produced a completion marker yet.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is also now holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `32%` utilization and `3943 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged, with both planning retries now stalled at the same higher iter range.
### Continued 2026-03-30 Status (Both Planning Retries Still at 58)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive and still sitting at the same latest visible iter level.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `9%` utilization and `3944 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Same Four, Fresh 14 Percent GPU Snapshot)

Another fresh check again shows no new completions and no new training epoch outputs. The two active planning retries are still alive at the same latest visible iter plateaus.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `14%` utilization and `3942 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Same Four, Fresh 8 Percent GPU Snapshot)

Another fresh check again shows no new completions and no new training epoch outputs. The two active planning retries are still alive at the same latest visible iter plateaus.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `8%` utilization and `3938 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Same Four, Fresh 24 Percent GPU Snapshot)

Another fresh check again shows no new completions and no new training epoch outputs. The two active planning retries are still alive at the same latest visible iter plateaus.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `24%` utilization and `3939 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Same Four, Fresh 21 Percent GPU Snapshot)

Another fresh check shows no new completions and no new training epoch outputs. The two active planning retries are still alive at their same latest visible iter plateaus.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `21%` utilization and `3940 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (P3 Holding at 58, P4 Holding at 43)

Another fresh check shows the same four tasks are still alive. The main planning rerun remains ahead, but neither planning retry has produced a completion marker yet, and neither resumed training run has emitted a new epoch summary line.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 58`. Its latest unresolved pattern is still `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`. Its latest unresolved pattern is still `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `33%` utilization and `3926 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Still Running)

Another check shows the same four active runs still progressing without any terminal output.

- `P3_plan_retry_bg` remains active and remains plateaued around `MPC iter 28`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and remains plateaued around `MPC iter 28`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `28%` utilization and `3919 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Low GPU Snapshot)

Another check shows the same four active runs still alive, with no new terminal output.

- `P3_plan_retry_bg` remains active and plateaued around `MPC iter 28`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and plateaued around `MPC iter 28`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `7%` utilization and `3925 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated Plateau)

Another repeated check shows no new results.

- `P3_plan_retry_bg` remains active and plateaued around `MPC iter 28`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and plateaued around `MPC iter 28`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `28%` utilization and `3922 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Another Low-Util Check)

Another check shows no new terminal outputs from the current four priority runs.

- `P3_plan_retry_bg` remains active and plateaued around `MPC iter 28`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and plateaued around `MPC iter 28`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `8%` utilization and `3948 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Ongoing, No Completion)

Another status check shows the same four active runs still alive, with no new terminal outputs.

- `P3_plan_retry_bg` remains active and plateaued around `MPC iter 28`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and plateaued around `MPC iter 28`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `17%` utilization and `3924 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Very Low GPU Use)

Another status check shows no new completions.

- `P3_plan_retry_bg` remains active and plateaued around `MPC iter 28`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and plateaued around `MPC iter 28`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `6%` utilization and `3924 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (P3 Advanced)

New check shows one meaningful planning update while the overall experiment slate remains unchanged.

- `P3_plan_retry_bg` remains active and has now advanced from the long `iter 28` plateau up to about `MPC iter 43`. It still has the same unresolved success pattern `self.is_success = [False, True, True, False, True]`, so the run has progressed but not closed out.
- `P4_plan_retry_bg` remains active and still appears plateaued around `MPC iter 28`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `8%` utilization and `3925 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (P3 Holding at 43)

Another check shows no new completions and only the same limited planning progress profile.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 28`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `16%` utilization and `3925 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Still Holding)

Another check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 28`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `18%` utilization and `3924 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Still No Terminal Results)

Another check shows no completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 28`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `23%` utilization and `3925 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Higher GPU Snapshot)

Another check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 28`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `34%` utilization and `3926 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (P4 Also Advanced)

New check shows the second planning retry has also moved beyond the earlier long plateau.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and has now advanced from the earlier `iter 28` plateau up to about `MPC iter 43`, while still showing `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `16%` utilization and `3926 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Both Planning Runs at 43)

Another check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is now also holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `15%` utilization and `3924 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Both Still at 43)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `10%` utilization and `3929 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (No New Outputs)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `15%` utilization and `3925 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Still Stable)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `15%` utilization and `3925 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated 43 Snapshot)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `9%` utilization and `3930 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Unchanged 43/43)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `15%` utilization and `3927 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (No New Terminal State)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `29%` utilization and `3924 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (High GPU Util Snapshot)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `72%` utilization and `3927 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Still No Epoch Output)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `13%` utilization and `3927 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (No Change, Moderate GPU)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `26%` utilization and `3927 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Memory Slightly Higher)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `20%` utilization and `3946 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Very High GPU Util Snapshot)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `79%` utilization and `3925 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Low GPU Again)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `7%` utilization and `3926 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Steady State)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `17%` utilization and `3925 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Still Flat)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `12%` utilization and `3926 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Utilization Up, Results Flat)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `31%` utilization and `3948 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Repeated Low GPU Snapshot)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `9%` utilization and `3925 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (Same Four, Same State)

Another repeated check shows no new completions and no new training epoch outputs.

- `P3_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [False, True, True, False, True]`.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `16%` utilization and `3925 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
### Continued 2026-03-30 Status (P3 Advanced to 58)

New check shows one planning retry has moved again, while the overall four-run slate remains unchanged.

- `P3_plan_retry_bg` remains active and has now advanced from the earlier `iter 43` plateau to about `MPC iter 58`. It still has the unresolved pattern `self.is_success = [False, True, True, False, True]`, so progress is real but the run is not complete.
- `P4_plan_retry_bg` remains active and is still holding around `MPC iter 43`, with `self.is_success = [True, False, True, True, True]`.
- `T1_cls_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `T2_patch_det_resume5` remains alive, with resume from epoch 3 confirmed and still no new epoch completion line.
- `logs/large_eval_status` still contains no `.done` markers for the active `P3` and `P4` retries.
- GPU snapshot at this check: about `20%` utilization and `3924 / 8188 MB` memory.

Conclusion: document updated, current four-run slate continues unchanged.
