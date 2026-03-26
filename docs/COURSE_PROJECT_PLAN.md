# DINO-WM Course Project Plan

## 1. Big Picture

This project is not only about finishing a few PointMaze runs. The larger goal is to turn DINO-WM into a clean course-project study of:

1. representation quality for action-conditioned world models
2. uncertainty modeling for latent dynamics and planning

The fixed backbone is:

`image encoder -> latent transition model -> planning`

The project should eventually explain which part of DINO-WM's performance comes from:

- strong pre-trained semantics
- spatially structured patch features
- predictive pretraining
- reconstructive or tokenizer-style latent design
- uncertainty-aware transition modeling

## 2. Project Vision

The full vision has three layers, not one.

### Layer A: Minimum Defensible Course Project

This is the smallest scope that still answers the assignment well.

- representation study:
  `DINOv2 patch` vs `DINOv2 CLS`
- dynamics study:
  `deterministic` vs `gaussian`
- environment:
  `PointMaze`
- outcome:
  one complete ablation matrix with training and planning

This layer is already implemented and mostly executed.

### Layer B: Stronger Empirical Validation

This layer makes the conclusions more credible.

- replace saturated `n_evals=3` planning with larger-sample follow-up
- aggregate across multiple seeds
- rely more on `mean_state_dist` than on saturated `success_rate`
- tighten the report so conclusions match the evidence

This layer is in progress now.

### Layer C: Expanded Representation Study

This is the long-horizon research version of the project.

Possible additions:

- `DINOv3`
- `V-JEPA`
- `DINO-Tok`
- `VFM-VAE`

This layer is not implemented yet and should only be attempted after Layer B is complete.

Feasibility snapshot as of the current local pass:

- `DINOv3`:
  blocked by Python `3.9` compatibility in the validated WSL environment
- `V-JEPA`:
  the only branch that has progressed into real encoder-level integration work
- `DINO-Tok`:
  still only a paper-level target in the current workspace; no verified official local code or checkpoint path has been established
- `VFM-VAE`:
  also still only a paper-level target in the current workspace; no verified official local code or checkpoint path has been established

## 3. Research Questions

### Q1. Representation

How much of DINO-WM's planning ability depends on spatial patch structure, compared with a compressed global representation?

Immediate test:

- `DINOv2 patch` vs `DINOv2 CLS`

Future extension:

- compare foundation-model patch features against predictive or reconstructive latents

### Q2. Dynamics

Does a lightweight stochastic latent predictor improve downstream planning compared with a deterministic predictor?

Immediate test:

- `deterministic` vs `gaussian`

Future extension:

- study whether uncertainty helps more on harder environments or longer horizons

### Q3. Evaluation Quality

Can we trust planning conclusions drawn from very small `n_evals`?

Current answer:

- not fully
- `success_rate` saturates too easily at `n_evals=3`
- larger-sample reevaluation is therefore required

## 4. Scope Control

The original proposal listed many encoders. That is useful as a long-term map, but not all of it belongs in the first complete deliverable.

### Implemented scope

- `DINOv2 patch + deterministic`
- `DINOv2 CLS + deterministic`
- `DINOv2 patch + gaussian`
- `DINOv2 CLS + gaussian`

### Planned but not implemented

- `DINOv3`
- `V-JEPA`
- `DINO-Tok`
- `VFM-VAE`

### Decision rule

Do not expand to new encoders until:

1. the current larger-sample reevaluation is finished
2. the report tables are stable
3. the current conclusions are written clearly

## 5. Workstreams

The project should be managed as four parallel workstreams.

### Workstream A: Infrastructure and Reproducibility

Goal:

- make DINO-WM run reliably in the local WSL2 environment

Completed:

- WSL2 Ubuntu 24.04 setup
- `micromamba` environment
- Python `3.9.19`
- MuJoCo `2.1`
- CUDA-enabled PyTorch
- stable DINOv2 hub pin in `models/dino.py`
- local Hydra configs:
  - `conf/train_wsl.yaml`
  - `conf/plan_point_maze_wsl.yaml`

Status:

- completed for the current project scope

### Workstream B: Core Model Changes

Goal:

- implement the minimum changes needed for the two ablations

Completed:

- existing `DINOv2 patch` encoder used as baseline
- existing `DINOv2 CLS` encoder used as representation ablation
- Gaussian predictor config added:
  - `conf/predictor/vit_gaussian.yaml`
- Gaussian output head implemented in:
  - `models/vit.py`
- Gaussian NLL and rollout support implemented in:
  - `models/visual_world_model.py`
- backward compatibility fix for older deterministic checkpoints

Status:

- completed for the current project scope

### Workstream C: Experiments

Goal:

- produce a clean matrix of training and planning runs

Completed:

- pretrained PointMaze planning sanity check
- four-setting sanity matrix
- three-seed formal matrix:
  - seeds `0, 1, 2`
  - `epochs=3`
  - `n_rollout=64`
  - four settings

In progress:

- larger-sample planning reevaluation

Status:

- partially completed

### Workstream D: Report and Collaboration

Goal:

- keep all decisions, results, and remaining tasks visible to collaborators

Main files:

- [README.md](/C:/Users/zack/Documents/GNN3/README.md)
- [PROJECT_README.md](/C:/Users/zack/Documents/GNN3/docs/PROJECT_README.md)
- [EXECUTION_LOG.md](/C:/Users/zack/Documents/GNN3/docs/EXECUTION_LOG.md)
- [EXPERIMENT_TRACKER.csv](/C:/Users/zack/Documents/GNN3/docs/EXPERIMENT_TRACKER.csv)
- [COURSE_REPORT_DRAFT.md](/C:/Users/zack/Documents/GNN3/docs/COURSE_REPORT_DRAFT.md)

Status:

- active and ongoing

## 6. Experiment Roadmap

### Stage 0. Baseline Reproduction

Purpose:

- verify that the repository, data, checkpoints, and planner all work end to end

Deliverables:

- pretrained planning run
- first custom training run
- first custom checkpoint planning run

Status:

- completed

### Stage 1. Minimum Ablation Matrix

Purpose:

- answer the assignment with the smallest valid experiment set

Matrix:

- `patch + deterministic`
- `CLS + deterministic`
- `patch + gaussian`
- `CLS + gaussian`

Deliverables:

- training metrics
- planning metrics
- code support for all four settings

Status:

- completed

### Stage 2. Formal Multi-Seed Matrix

Purpose:

- move from single-run sanity checks to something more stable

Design:

- `3 seeds x 4 settings`
- fixed training recipe
- fixed planning recipe

Deliverables:

- 12 training runs
- 12 planning runs
- summary table

Status:

- completed

### Stage 3. Larger-Sample Planning Reevaluation

Purpose:

- fix the main weakness of the formal matrix:
  `n_evals=3` is too small

Design:

- target approximately `10` evaluation episodes per model
- use direct `n_evals=10` when stable
- otherwise use two `n_evals=5` shards

Completed:

- `seed0 patch + deterministic`
- `seed0 CLS + deterministic`
- `seed0 patch + gaussian`
- `seed0 CLS + gaussian`
- `seed1 patch + deterministic`
- `seed1 patch + gaussian`
- `seed1 CLS + gaussian`
- `seed2 patch + deterministic`
- `seed2 patch + gaussian`
- `seed1 CLS + deterministic`, shard B only

Pending:

- `seed2 CLS + gaussian`

Status:

- in progress

Important decision:

- `seed1 CLS + deterministic` was attempted twice on shard A and completed once on shard B
- because shard A repeatedly failed to finish cleanly, this group is now treated as dropped for the larger-sample follow-up
- do not spend more time on shard-A reruns for this setting
- `seed2 CLS + deterministic` shard A was also stopped after stalling without `final_eval`
- this seed-2 CLS branch is now optional unfinished work, not a blocker for closeout

### Stage 4. Encoder-3 and Encoder-4 Integration

Purpose:

- extend the project beyond the completed `DINOv2 patch / CLS` core without reopening the entire experimental matrix at once

Current local read:

- `DINOv3`:
  checked and blocked in Python `3.9`
- `V-JEPA`:
  checkpointed and partially validated, but still not exposed through an encoder-only wrapper compatible with the local DINO-WM training stack

Immediate next step:

- prefer `V-JEPA` encoder-only integration over any further work on `DINOv3`
- defer `DINO-Tok` and `VFM-VAE` until a real official code-and-weight path is identified
- `seed2 CLS + gaussian` shard A was attempted twice and also failed to reach `final_eval`
- do not continue the remaining seed-2 CLS branch in the main closeout path

### Stage 4. Final Analysis and Write-Up

Purpose:

- convert experiments into a careful final narrative

Deliverables:

- final result tables
- honest interpretation of what is supported
- explicit limits section
- collaborator-friendly README and report

Status:

- partially completed

### Stage 5. Extended Encoder Study

Purpose:

- expand beyond the minimum course-project scope

Candidate additions:

- `DINOv3`
- `V-JEPA`
- `DINO-Tok`
- `VFM-VAE`

Required before starting:

- Stage 3 complete
- Stage 4 stable
- enough time budget for integration and debugging

Status:

- deferred

## 6.1 Encoder-3 and Encoder-4 Ownership Plan

The original `DINOv2 patch` and `DINOv2 CLS` branches are already being handled elsewhere. The next local workstream is therefore focused on the third and fourth encoder slots only.

Assigned scope for this branch:

- encoder 3:
  `DINOv3 patch`
- encoder 4:
  `V-JEPA`
- tasks:
  `point_maze` and `wall_single`
- training target:
  `10 epochs`
- evaluation target:
  local planning plus shard-based larger-sample follow-up

Decision rule for this branch:

- start with `deterministic` only
- do not add Gaussian for the new encoders until deterministic training and planning both work on both tasks

Execution order:

1. feasibility check for `DINOv3`
2. feasibility check for `V-JEPA`
3. `point_maze` deterministic sanity for each new encoder
4. `wall_single` deterministic sanity for each new encoder
5. `10 epoch` deterministic runs on both tasks
6. local planning evaluation plus two-shard larger-sample follow-up

Minimum experiment matrix for encoder 3 and 4:

- `DINOv3 patch + deterministic + point_maze`
- `DINOv3 patch + deterministic + wall_single`
- `V-JEPA + deterministic + point_maze`
- `V-JEPA + deterministic + wall_single`

Estimated time for this branch:

- DINOv3 integration and sanity:
  about `2` to `4` hours
- V-JEPA integration and sanity:
  about `3` to `6` hours
- four deterministic `10 epoch` runs across two tasks:
  about `4` to `12` hours
- local planning evaluation:
  about `4` to `10` hours

Total estimate for encoder-3 and encoder-4 deterministic coverage:

- about `11` to `28` hours

Current feasibility snapshot:

- `DINOv3 patch`
  - status:
    `blocked`
  - blocker:
    the official `facebookresearch/dinov3` hub path currently fails under Python `3.9` because it uses Python `3.10+` union syntax such as `float | None`
  - near-term handling:
    do not start the `10 epoch` DINOv3 branch until there is a Python-version-compatible loading path

- `V-JEPA`
  - status:
    `active feasibility`
  - blocker:
    the official `facebookresearch/jepa-wms` hub path now reaches model construction, but stops because it expects the opensource V-JEPA visual checkpoint at `${JEPAWM_OSSCKPT}/vjepa2_opensource/vjepa2_vit_giant.pth`
  - checkpoint note:
    the official V-JEPA v2 opensource encoder files are large; the public `vitl.pt` checkpoint is about `5.1 GB`, and the `vitg.pt` checkpoint is about `16.5 GB`
  - near-term handling:
    decide whether to prepare the required opensource checkpoint locally; if yes, continue with latent-shape verification and a deterministic sanity run after the checkpoint is in place
  - current active step:
    local provisioning of the required V-JEPA v2 giant checkpoint is in progress so that the next experiment can move from dependency cleanup to actual encoder loading

## 7. Current Evidence

### Formal matrix summary

Across the current three-seed formal matrix:

- `patch + deterministic` is the best current baseline
- `CLS + deterministic` is unexpectedly close
- Gaussian models improve training likelihood-style losses
- Gaussian models do not currently improve planning quality

### Larger-sample follow-up summary

Current larger-sample results suggest:

- `mean_state_dist` is more informative than `success_rate`
- `patch + deterministic` remains strongest overall
- `CLS + gaussian` can beat `patch + gaussian` on at least some seed-0 follow-up
- the main open question is now whether spending more time on the optional seed-2 CLS follow-up would materially change the report conclusions

## 8. What Counts as Success

### Minimum success

The project is already a valid course project if it ends with:

- one complete implementation section
- one complete `patch/CLS x deterministic/gaussian` matrix
- one honest discussion of evaluation limitations

### Strong success

The project becomes strong if it also includes:

- larger-sample planning reevaluation across all three seeds
- a cleaned-up final report with stable tables

### Stretch success

The project becomes ambitious if it additionally includes:

- one third encoder family such as `DINOv3` or `V-JEPA`

## 9. Remaining Tasks

### Immediate

- aggregate larger-sample metrics into report-ready tables

### Near-term

- revise the report conclusions around saturated `success_rate`
- ensure the README, tracker, and report all say the same thing

### Optional

- choose one third representation family and test a minimum integration path

## 10. Estimated Remaining Time

For the current plan, not the long-horizon vision:

- optional seed-2 Gaussian CLS follow-up:
  about `0.5` to `1.0` hour
- aggregation and document cleanup:
  about `0.5` to `1.0` hour

Estimated total remaining time for the current deliverable:

- about `0.5` to `1.0` hour if the project closes after patch-focused reevaluation

For the extended vision:

- adding one new encoder family is likely a separate `4+` hour task even in the best case
- adding multiple new encoder families is a separate project phase, not a small follow-up

## 11. Recommended Next Decisions

1. Finish Stage 3 before starting any new encoder integration.
2. Treat the current deliverable as a complete study of:
   `patch/CLS x deterministic/gaussian`
3. Present `DINOv3`, `V-JEPA`, `DINO-Tok`, and `VFM-VAE` as future extensions unless they are actually implemented and run.
4. Use `mean_state_dist` as the primary planning comparison metric.
5. Keep the final write-up conservative about what the data currently supports.
6. Current operating mode: closeout-first.
7. Do not let optional CLS follow-up block report completion.

## 12. Collaboration Paths

The project is now at a stage where several different technical directions are all reasonable. A collaborator does not need to pick up the whole pipeline to make progress.

### Path A: Final Evaluation and Metric Quality

Examples:

- summarize planning outcomes with `mean_state_dist` as the main metric
- clearly label the seed-2 CLS branch as optional unfinished work
- turn the completed reevaluation results into final tables and concise report language

### Path B: Expanded Encoder Route

Examples:

- try one additional representation family, preferably `DINOv3` or `V-JEPA`
- focus on a minimum viable path:
  - encoder integration
  - latent compatibility
  - one sanity training run
  - one sanity planning run
- record what extra engineering work that encoder requires relative to DINOv2

### Path C: Results Consolidation and Final Writing

Examples:

- build one clean final table for the formal matrix
- build one clean final table for the larger-sample follow-up
- compute and verify shard-based averages
- align the README, tracker, execution log, and report wording
- refine the abstract, conclusion, and limitations

### Path D: Reproducibility and Sync

Examples:

- verify that scripts and commands still match the tracker
- prepare a short rerun guide for the main completed experiments
- sync the latest docs and status updates to the collaboration repository

## 13. One-Week Innovation Options Beyond Additional Encoders

If the project window is only about one week, the most useful innovation is not necessarily "add more encoder names". A better rule is to prefer small extensions that produce a clearer new conclusion.

### 13.1 Highest-Value Low-to-Medium-Risk Options

#### A. Uncertainty-Aware Planning

Current Gaussian work changes the training objective, but planning still mostly follows the mean prediction. A stronger extension would let uncertainty influence planning directly.

Examples:

- penalize high-variance rollouts during planning
- compare risk-neutral and risk-averse planning
- check whether uncertainty helps mainly at longer horizons

Why this is attractive:

- it builds directly on the implemented Gaussian predictor
- it adds a real methodological contribution rather than just another loss
- it is lighter than integrating a completely new representation family

#### B. Horizon-Sensitivity Analysis

Another good one-week contribution is to study how different representations or dynamics variants degrade as rollout horizon grows.

Examples:

- compare short-horizon and long-horizon latent prediction error
- compare planning quality as horizon increases
- test whether Gaussian uncertainty becomes more meaningful at larger horizon

Why this is attractive:

- low engineering overhead
- easy to explain in the final report
- directly relevant to world-model usefulness

#### C. Robustness Evaluation

Instead of only asking which representation is best under clean inputs, test which one is more stable under simple perturbations.

Examples:

- observation noise
- brightness or crop perturbations
- frame drop
- action noise

Why this is attractive:

- the implementation burden is modest
- results are often easier to interpret than adding one more large encoder
- it broadens the contribution without changing the core pipeline

### 13.2 Medium-Risk Representation-Focused Extensions

#### D. Spatial-Structure Middle Ablations

Between patch tokens and CLS token there is a useful middle ground.

Examples:

- pooled patch groups
- reduced patch grids
- masked or subsampled patches

Why this is attractive:

- it gives a cleaner answer to how much spatial structure planning actually needs
- it is usually cheaper than integrating a completely different external model

### 13.3 Higher-Risk Options

#### E. Additional Encoder Families

This includes:

- `DINOv3`
- `V-JEPA`
- `DINO-Tok`
- `VFM-VAE`

These can still be worth doing, but they are no longer the only or even the best innovation path for a short project. Their main cost is integration complexity, environment preparation, and checkpoint management.

### 13.4 Recommended Priority For A One-Week Project

If time is limited, the recommended innovation priority is:

1. `uncertainty-aware planning`
2. `horizon-sensitivity analysis`
3. `robustness evaluation`
4. `one new encoder family`

Practical interpretation:

- if the goal is a clear, defensible project contribution, the first three are often better than trying to integrate many new foundation models
- if the goal is a third and fourth encoder branch specifically, then `DINOv3` and `V-JEPA` remain the most reasonable choices, but they should be treated as heavier engineering tasks
