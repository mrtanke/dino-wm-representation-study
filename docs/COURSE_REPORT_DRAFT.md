# DINO-WM Course Project Report Draft

## Title

Representation and Uncertainty in Action-Conditioned World Models:
A DINO-WM Study on PointMaze

## 1. Introduction

World models aim to predict future states conditioned on actions and use those predictions for planning. DINO-WM is a strong recent baseline because it combines a frozen visual encoder, a latent transition model, and test-time planning. Its main claim is that strong pre-trained visual features can substantially improve downstream planning.

This course project studies two questions under the DINO-WM pipeline:

1. Representation study:
   How much does the choice of latent visual representation matter for action-conditioned prediction and planning?
2. Dynamics study:
   Can a lightweight stochastic extension improve the latent transition model compared with a purely deterministic predictor?

We keep the high-level pipeline fixed:

`image encoder -> latent transition model -> planning`

Then we change only one axis at a time.

## 2. Project Scope

The original project idea proposed a broad comparison across several representation families, including DINOv2 patch tokens, DINOv2 CLS, DINOv3, V-JEPA, DINO-Tok, and VFM-VAE. In practice, the engineering cost of integrating multiple new foundation models is high, so this implementation focuses on a minimal but defensible experimental set:

- `DINOv2 patch tokens` as the main baseline
- `DINOv2 CLS token` as the key representation ablation
- `Deterministic latent transition`
- `Gaussian latent transition`

This scope is sufficient to answer the two core questions of the project while keeping the implementation tractable.

## 3. Method

### 3.1 Baseline DINO-WM Pipeline

The repository follows the standard DINO-WM structure:

- Observation model:
  A frozen DINOv2 encoder maps each image to latent features.
- Transition model:
  A ViT-based dynamics model predicts the next latent state from past latent states and actions.
- Decoder:
  An image decoder is optional and mainly used for visualization.
- Planner:
  A CEM-based MPC planner searches action sequences in latent space.

### 3.2 Representation Study

We compare two encoders already supported by the repository:

- DINOv2 patch tokens
- DINOv2 CLS token

The intended contrast is straightforward:

- Patch tokens preserve spatial structure and local semantics.
- CLS compresses the full image into a single global token.

Our hypothesis is that CLS should be weaker for control because it discards spatial detail that may matter for action-conditioned prediction.

### 3.3 Dynamics Study

The original deterministic model predicts:

`(z_t, a_t) -> z_hat_(t+1)`

We extend it to a Gaussian latent model:

`(z_t, a_t) -> (mu_(t+1), logvar_(t+1))`

The stochastic predictor is implemented by changing the final predictor head to output both mean and log-variance. Training uses Gaussian negative log-likelihood with a small MSE auxiliary term:

```python
logvar = torch.clamp(logvar, -5, 5)
var = torch.exp(logvar)
gaussian_nll = ((target - mu) ** 2 / var + logvar).mean()
loss = gaussian_nll + 0.1 * mse(mu, target)
```

For rollout and planning, we use `mu` as the predicted latent state.

## 4. Implementation Details

### 4.1 Environment

The official repository targets Linux. The project was therefore run in WSL2 Ubuntu 24.04 with:

- Python 3.9
- MuJoCo 2.1
- CUDA-enabled PyTorch
- the repository environment created from `environment.yaml`

### 4.2 Local Compatibility Fixes

Several practical issues had to be resolved before experiments could run:

1. WSL-specific Hydra configs were added to replace the default Slurm launcher.
2. MuJoCo 2.1 was installed manually for `mujoco_py`.
3. DINOv2 loading had to be pinned to a Python-3.9-compatible ref.

The last point is important. The repository loads DINOv2 from `torch.hub`, which by default tracks the upstream `main` branch. The current upstream version uses newer Python syntax that is incompatible with Python 3.9. To stabilize the project, the encoder was pinned to the compatible DINOv2 ref `ebc1cba`.

### 4.3 CLS Limitation

A practical issue appeared for the DINOv2 CLS ablation:

- the current decoder assumes a spatial patch grid
- the CLS representation contains only a single token

As a result, CLS cannot directly reuse the existing image decoder. For this project, CLS experiments were run with the decoder disabled, which is still acceptable because the decoder is optional in the original DINO-WM design.

## 5. Experimental Setup

### 5.1 Environment

All current runs were performed on `PointMaze`, because it is the simplest environment to use for an end-to-end sanity check of:

- dataset loading
- training
- checkpoint saving
- planning

### 5.2 Sanity-Run Protocol

Since the main goal was to build a working course-project prototype first, experiments were run as small sanity checks:

- 32 rollouts from the dataset
- batch size 4
- 1 training epoch
- offline W&B logging

These runs are not final benchmark-quality results. They are meant to validate the system and provide early comparisons.

### 5.3 Configurations

- E1: `DINOv2 patch + deterministic`
- E2: `DINOv2 CLS + deterministic`
- E3: `DINOv2 patch + gaussian`
- E4: `DINOv2 CLS + gaussian`

## 6. Results

Table 1 summarizes the current sanity-scale results.

| Run | Encoder | Transition | Training or Planning Result |
|---|---|---|---|
| E1_plan | DINOv2 patch | deterministic | `success_rate = 1.0`, `mean_state_dist = 0.9909` |
| E1_train | DINOv2 patch | deterministic | `train_loss = 1.8771`, `val_loss = 1.3130` |
| E1_trained_plan | DINOv2 patch | deterministic | `success_rate = 1.0`, `mean_state_dist = 0.9120` |
| E2_train | DINOv2 CLS | deterministic | `train_loss = 1.7640`, `val_loss = 1.2005` |
| E2_plan | DINOv2 CLS | deterministic | `success_rate = 1.0`, `mean_state_dist = 0.5267` |
| E3_train | DINOv2 patch | gaussian | `train_loss = -1.1801`, `val_loss = -2.3765`, `val_z_mse_loss = 0.0818` |
| E3_plan | DINOv2 patch | gaussian | `success_rate = 1.0`, `mean_state_dist = 2.3579` |
| E4_train | DINOv2 CLS | gaussian | `train_loss = -2.0987`, `val_loss = -2.6678`, `val_z_mse_loss = 0.0329` |
| E4_plan | DINOv2 CLS | gaussian | `success_rate = 1.0`, `mean_state_dist = 1.1232` |

### 6.1 Pretrained Planning Baseline

Using the official pretrained PointMaze checkpoint, planning ran successfully end to end.

- `success_rate = 1.0`
- `mean_state_dist = 0.9909`

This confirmed that the environment, checkpoint loading, and planner were all working.

### 6.2 Deterministic Patch Training

Small-scale sanity run:

- `train_loss = 1.8771`
- `val_loss = 1.3130`

This establishes the main deterministic patch-token baseline.

### 6.3 Deterministic CLS Training

With the decoder disabled:

- `train_loss = 1.7640`
- `val_loss = 1.2005`

At the level of one-epoch latent loss, CLS does not look obviously worse than patch tokens. However, this should not yet be interpreted as evidence that CLS is equally good for planning. The main representation claim should be evaluated with larger runs and planning metrics, not only short training loss.

### 6.4 Gaussian Patch Training

Small-scale stochastic run:

- `train_loss = -1.1801`
- `val_loss = -2.3765`
- `val_z_mse_loss = 0.0818`
- `val_z_logvar_mean = -3.1655`

The negative total loss is expected because Gaussian NLL can be negative when the predicted variance becomes small and the fit is good enough.

### 6.5 Gaussian Planning

A sanity planning run using the Gaussian checkpoint also completed successfully:

- `success_rate = 1.0`
- `mean_state_dist = 2.3579`

This verifies that the stochastic transition model integrates correctly into checkpoint loading and planning.

### 6.6 Matrix View

The current PointMaze sanity matrix is now complete for the two implemented representation choices and the two implemented transition choices.

| Encoder | Deterministic planning | Gaussian planning |
|---|---|---|
| DINOv2 patch | `success_rate = 1.0`, `mean_state_dist = 0.9120` | `success_rate = 1.0`, `mean_state_dist = 2.3579` |
| DINOv2 CLS | `success_rate = 1.0`, `mean_state_dist = 0.5267` | `success_rate = 1.0`, `mean_state_dist = 1.1232` |

At this sanity scale, all four combinations can train and plan successfully. The most notable observation is that the CLS variants did not collapse in planning; in fact, under this very small setup, the deterministic CLS run achieved the lowest final state distance. This is interesting, but it is still too early to interpret as a reliable scientific conclusion because the matrix currently uses single-seed, one-epoch runs.

### 6.7 Formal Matrix View

To move beyond pure sanity checks, a first longer PointMaze batch was run with:

- `seed = 0`
- `epochs = 3`
- `n_rollout = 64`
- `n_evals = 3`

Table 2 summarizes the resulting formal matrix.

| Encoder | Transition | Final training loss | Final validation loss | Planning success rate | Mean state distance |
|---|---|---:|---:|---:|---:|
| DINOv2 patch | deterministic | 0.0800 | 0.0440 | 1.0 | 3.2298 |
| DINOv2 CLS | deterministic | 0.0414 | 0.0238 | 1.0 | 2.1996 |
| DINOv2 patch | gaussian | -2.9815 | -3.1822 | 1.0 | 3.6079 |
| DINOv2 CLS | gaussian | -3.0505 | -3.2116 | 1.0 | 1.7849 |

The formal batch preserves the same qualitative pattern seen in the earlier sanity runs: all four combinations remain trainable and plannable, and the CLS variants do not underperform by default on this PointMaze setup. Within this first seed, `CLS + gaussian` achieved the lowest mean state distance, followed by `CLS + deterministic`.

This result is notable because it runs against the original intuition that patch structure should clearly dominate CLS for planning. However, it would still be premature to make a strong claim from a single-seed batch. At this stage, the correct interpretation is narrower: the current PointMaze evidence does not support the claim that CLS is obviously inferior.

### 6.8 Three-Seed Summary

The formal matrix was then repeated for `seed = 1` and `seed = 2`, giving three total seeds for each of the four configurations. Table 3 summarizes the resulting averages.

| Encoder | Transition | Mean train loss | Mean val loss | Mean final state distance |
|---|---|---:|---:|---:|
| DINOv2 patch | deterministic | 0.0825 | 0.0429 | 3.2806 |
| DINOv2 CLS | deterministic | 0.0411 | 0.0240 | 3.3047 |
| DINOv2 patch | gaussian | -2.9815 | -3.1785 | 3.7772 |
| DINOv2 CLS | gaussian | -3.0751 | -3.2282 | 3.3277 |

Two observations stand out. First, the success rate again saturated at `1.0` in the final evaluation for all runs, which means it is not a sufficiently informative metric under the current `n_evals = 3` setup. Second, once we look at mean final state distance instead, the ranking becomes more nuanced. The best average result is `DINOv2 patch + deterministic`, but `DINOv2 CLS + deterministic` is extremely close, and `DINOv2 CLS + gaussian` remains competitive.

This changes the earlier single-seed interpretation. The single-seed result had suggested that CLS might be clearly better. The three-seed average no longer supports that stronger claim. Instead, the current evidence suggests a narrower conclusion:

- CLS is not obviously worse than patch tokens on PointMaze
- deterministic dynamics are currently more stable than Gaussian dynamics under this setup
- the Gaussian extension improves training likelihood objectives, but that does not translate into better planning quality here

### 6.9 Larger-Sample Planning Follow-Up

The main weakness of the three-seed formal matrix is that planning still used only `n_evals = 3`. That made `success_rate` saturate at `1.0` for all configurations, which limited how informative that metric could be.

To address this, larger-sample planning evaluation was added after the formal matrix. In principle, the natural next step was to run `n_evals = 10` for every checkpoint. In practice, WSL2 was not stable enough for every configuration under direct `n_evals = 10`. The deterministic patch checkpoint completed successfully, but other configurations sometimes stopped mid-rollout without a clean Python traceback, and WSL occasionally returned `Wsl/Service/E_UNEXPECTED`.

The practical workaround was to split larger evaluation into two shards:

- shard A: `n_evals = 5`, `seed = 0`, giving evaluation seeds `1..5`
- shard B: `n_evals = 5`, `seed = 5`, giving evaluation seeds `6..10`

Together, the two shards cover 10 evaluation episodes while keeping each run smaller and more stable.

Table 4 summarizes the currently completed larger-sample seed-0 results.

| Checkpoint | Encoder | Transition | Eval protocol | Final state distance |
|---|---|---|---|---:|
| `2026-03-18/19-12-06` | DINOv2 patch | deterministic | direct `n_evals=10` | 3.1738 |
| `2026-03-18/20-01-01` shard A | DINOv2 CLS | deterministic | `n_evals=5`, `seed=0` | 3.7136 |
| `2026-03-18/20-01-01` shard B | DINOv2 CLS | deterministic | `n_evals=5`, `seed=5` | 3.3445 |
| `2026-03-18/20-20-39` shard A | DINOv2 patch | gaussian | `n_evals=5`, `seed=0` | 4.4740 |
| `2026-03-18/20-20-39` shard B | DINOv2 patch | gaussian | `n_evals=5`, `seed=5` | 4.2024 |
| `2026-03-18/20-45-03` shard A | DINOv2 CLS | gaussian | `n_evals=5`, `seed=0` | 4.2965 |
| `2026-03-18/20-45-03` shard B | DINOv2 CLS | gaussian | `n_evals=5`, `seed=5` | 3.3520 |
| `2026-03-18/21-12-22` shard A | DINOv2 patch | deterministic | `n_evals=5`, `seed=1` | 4.0346 |
| `2026-03-18/21-12-22` shard B | DINOv2 patch | deterministic | `n_evals=5`, `seed=6` | 4.2033 |

If we average the two deterministic CLS shards, the seed-0 larger-sample estimate is:

- `mean_state_dist ~= 3.5291`

If we average the two Gaussian patch shards, the seed-0 larger-sample estimate is:

- `mean_state_dist ~= 4.3382`

If we average the two Gaussian CLS shards, the seed-0 larger-sample estimate is:

- `mean_state_dist ~= 3.8242`

If we average the two deterministic patch shards for seed 1, the current larger-sample estimate is:

- `mean_state_dist ~= 4.1189`

These larger-sample results are useful because they sharpen the earlier interpretation. At the seed-0 level:

- deterministic patch remains stronger than patch Gaussian
- deterministic CLS remains competitive with deterministic patch
- CLS Gaussian is better than patch Gaussian on seed 0, but neither Gaussian variant beats deterministic patch
- the Gaussian models do not improve planning quality despite better training likelihood metrics

This reinforces the earlier three-seed message: under the current PointMaze setup, the Gaussian dynamics extension improves the training objective more than it improves final planning performance.

This is a more defensible result for the course report because it is based on repeated runs rather than a single seed.

## 7. Discussion

### 7.1 What Has Been Demonstrated

The project now has working implementations for both requested axes:

- representation:
  patch tokens vs CLS
- dynamics:
  deterministic vs Gaussian

This is already enough to claim that the project goals have been converted from a proposal into an executable system.

### 7.2 What Cannot Yet Be Claimed

The current results are still sanity-scale experiments. Therefore, we should avoid making strong scientific claims such as:

- CLS is better than patch tokens
- Gaussian dynamics definitively improves planning

Those claims require:

- more training epochs
- more than one environment
- repeated planning runs
- more stable metrics across seeds

One concrete example is the larger-sample planning comparison. The project now has a meaningful seed-0 larger-sample follow-up, but it does not yet have the same larger-sample evaluation for all seeds and all four model combinations. Without that, the stronger statistical version of the representation comparison remains incomplete.

### 7.3 Main Engineering Lessons

Three practical lessons stood out:

1. WSL2 plus `/mnt/c` file access can cause training stalls when DataLoader workers are enabled.
2. Direct `torch.hub` loading from a moving upstream repository is fragile.
3. A decoder designed for patch grids does not automatically support global-token representations.

These observations are useful implementation takeaways for future world-model projects.

## 8. Limitations

- Only PointMaze has been validated so far.
- Current runs are small sanity experiments rather than full training runs.
- CLS uses no decoder, so decoder-side comparisons are not available.
- DINOv3 and V-JEPA integration was not implemented in this phase.
- The larger-sample planning reevaluation is still incomplete; it now covers all four seed-0 configurations and seed-1 deterministic patch, but not yet the full seed-1 to seed-2 matrix.

## 8.1 Current Progress

At the time of writing, the project is in a mixed completed/in-progress state.

Completed:

- implementation of the `patch/CLS x deterministic/Gaussian` experimental matrix
- full three-seed formal matrix with training and planning
- documentation of the main engineering changes and the three-seed summary

In progress:

- larger-sample planning reevaluation intended to reduce the saturation problem of `success_rate`

Not yet implemented:

- additional representation families such as DINOv3, V-JEPA, DINO-Tok, and VFM-VAE

Estimated remaining time for the current reevaluation plan:

- about `5` to `8` hours

This estimate assumes:

- about `1.5` to `2.5` hours for the remaining seed-1 configurations
- about `2.0` to `3.0` hours for seed 2
- about `0.5` to `1.0` hour for aggregation and final report cleanup

## 9. Future Work

The next steps are clear:

1. Finish the larger-sample planning follow-up for `CLS + gaussian` on seed 0.
2. Extend the same larger-sample follow-up to seeds 1 and 2.
3. Aggregate the larger-sample planning metrics before drawing stronger representation conclusions.
4. Add a third representation, preferably DINOv3 or V-JEPA.
5. Study uncertainty more directly by plotting predicted variance over rollout horizon.
6. If needed, design a CLS-compatible decoder for visualization.

## 10. Reproducibility

The repository now includes a compact reproducibility path for the implemented experiments:

- WSL-specific configs:
  `conf/train_wsl.yaml`, `conf/plan_point_maze_wsl.yaml`
- Gaussian predictor config:
  `conf/predictor/vit_gaussian.yaml`
- WSL sanity script:
  `scripts/wsl_pointmaze_sanity.sh`
- experiment table:
  `docs/EXPERIMENT_TRACKER.csv`

## 11. Conclusion

This project successfully turned the original DINO-WM course-project idea into a working prototype. We reproduced the pretrained PointMaze planning baseline, ran deterministic DINOv2 patch and CLS training sanity checks, implemented a Gaussian latent transition model, and verified that the stochastic checkpoint can also be used for planning.

The current stage should be viewed as a validated implementation milestone rather than a finished benchmark study. Still, the two central project questions are now represented by runnable code and initial experimental evidence, including a completed three-seed formal matrix and a partially completed larger-sample planning follow-up. This provides a solid base for final course-project reporting.

## Appendix: Current Experiment Table

| ID | Encoder | Transition | Result |
|---|---|---|---|
| E1_plan | DINOv2 patch | deterministic | pretrained planning sanity passed |
| E1_train | DINOv2 patch | deterministic | 1 epoch sanity passed |
| E1_trained_plan | DINOv2 patch | deterministic | trained-checkpoint planning sanity passed |
| E2_train | DINOv2 CLS | deterministic | 1 epoch sanity passed, decoder disabled |
| E2_plan | DINOv2 CLS | deterministic | trained-checkpoint planning sanity passed |
| E3_train | DINOv2 patch | gaussian | 1 epoch sanity passed |
| E3_plan | DINOv2 patch | gaussian | planning sanity passed |
| E4_train | DINOv2 CLS | gaussian | 1 epoch sanity passed, decoder disabled |
| E4_plan | DINOv2 CLS | gaussian | planning sanity passed |
