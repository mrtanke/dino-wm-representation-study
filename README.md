# DINO-WM Course Project

Course-project extension of DINO-WM focused on two questions:

1. Representation:
   `DINOv2 patch` vs `DINOv2 CLS`
2. Dynamics:
   `deterministic` vs `gaussian` latent transition models

This README is the collaboration homepage for the current project state: vision, progress, key results, remaining work, and ownership-ready next steps.

## Project Vision

Keep the original DINO-WM pipeline fixed:

`image encoder -> latent transition model -> planning`

Then answer two practical questions for action-conditioned world modeling on PointMaze:

- Does spatially structured `DINOv2 patch` still matter compared with the compressed `DINOv2 CLS` representation?
- Does a lightweight stochastic extension improve planning quality over the original deterministic latent dynamics?

The goal is not to rebuild the whole paper from scratch, but to produce a defensible course-project study with a clean ablation matrix, reproducible scripts, and conclusions that do not overclaim from weak evaluation.

For the full roadmap beyond the current experiment snapshot, see:

- [docs/COURSE_PROJECT_PLAN.md](/C:/Users/zack/Documents/GNN3/docs/COURSE_PROJECT_PLAN.md)

That document now separates:

- the minimum defensible course-project scope
- the stronger empirical validation phase
- the longer-horizon expanded representation study

## Current Status

### Completed

- End-to-end DINO-WM setup in WSL2 Ubuntu 24.04
- Stable local configs for training and planning:
  - `conf/train_wsl.yaml`
  - `conf/plan_point_maze_wsl.yaml`
- Stable DINOv2 loading pinned in `models/dino.py`
- Gaussian latent-transition extension implemented:
  - `conf/predictor/vit_gaussian.yaml`
  - `models/vit.py`
  - `models/visual_world_model.py`
- Full sanity matrix completed on PointMaze:
  - `patch + deterministic`
  - `CLS + deterministic`
  - `patch + gaussian`
  - `CLS + gaussian`
- Full formal matrix completed:
  - `3 seeds x 4 settings = 12 runs`
  - `epochs=3`
  - `n_rollout=64`
  - `n_evals=3`

### In Progress

Larger-sample planning reevaluation is being used to replace the saturated `n_evals=3` planning view.

Completed larger-sample follow-up:

- `seed0 patch + deterministic`
- `seed0 CLS + deterministic`
- `seed0 patch + gaussian`
- `seed0 CLS + gaussian`
- `seed1 patch + deterministic`
- `seed1 patch + gaussian`
- `seed1 CLS + gaussian`
- `seed2 patch + deterministic`
- `seed2 patch + gaussian`

Still pending:

- `seed1 CLS + deterministic`
- `seed2 CLS + deterministic` (optional unfinished)
- `seed2 CLS + gaussian` (optional unfinished)

Exploratory cross-environment sanity checks:

- `Wall` pretrained planning sanity completed successfully
- `PushT` pretrained planning sanity under a reduced budget stalled with poor results
- a stronger-budget `PushT` retry improved state distance substantially, but still stalled before `final_eval`

### Not Implemented Yet

These were part of the broader proposal, but are not part of the current implemented scope:

- `DINOv3`
- `V-JEPA`
- `DINO-Tok`
- `VFM-VAE`

## Key Results So Far

### Formal 3-Seed Matrix

The most useful planning metric so far is `mean_state_dist`, because `success_rate` saturates at `1.0` under `n_evals=3`.

| Setting | Mean train loss | Mean val loss | Mean state dist |
| --- | ---: | ---: | ---: |
| `DINOv2 patch + deterministic` | `0.0825` | `0.0429` | `3.2806` |
| `DINOv2 CLS + deterministic` | `0.0411` | `0.0240` | `3.3047` |
| `DINOv2 patch + gaussian` | `-2.9815` | `-3.1785` | `3.7772` |
| `DINOv2 CLS + gaussian` | `-3.0751` | `-3.2282` | `3.3277` |

Current interpretation:

- `patch + deterministic` remains the strongest current baseline
- `CLS + deterministic` is surprisingly competitive on PointMaze
- Gaussian models improve likelihood-style training losses
- Gaussian models have not yet shown a planning advantage

### Larger-Sample Follow-Up

The direct `n_evals=10` evaluation path was unstable in WSL/CUDA, so the current stable fallback is two shards of `n_evals=5`.

Completed larger-sample snapshots:

- `seed0 patch + deterministic`
  - direct `n_evals=10`
  - `mean_state_dist = 3.1738`
- `seed0 CLS + deterministic`
  - two-shard average `mean_state_dist ~= 3.5291`
- `seed0 patch + gaussian`
  - two-shard average `mean_state_dist ~= 4.3382`
- `seed0 CLS + gaussian`
  - two-shard average `mean_state_dist ~= 3.8242`
- `seed1 patch + deterministic`
  - two-shard average `mean_state_dist ~= 4.1189`
- `seed1 patch + gaussian`
  - two-shard average `mean_state_dist ~= 3.9366`
- `seed1 CLS + gaussian`
  - two-shard average `mean_state_dist ~= 3.9565`
- `seed2 patch + deterministic`
  - two-shard average `mean_state_dist ~= 3.4584`
- `seed2 patch + gaussian`
  - two-shard average `mean_state_dist ~= 4.3641`

Closed as optional unfinished:

- `seed2 CLS + deterministic`
  - stalled before `final_eval`
- `seed2 CLS + gaussian`
  - stalled twice before `final_eval`, including one overnight retry

Current interpretation:

- `mean_state_dist` is more informative than `success_rate`
- `patch + deterministic` still looks best overall
- `CLS + gaussian` looks better than `patch + gaussian` on seed 0
- the stochastic extension still does not clearly beat the deterministic baseline in planning
- on the patch side, the larger-sample follow-up now consistently favors deterministic over Gaussian on seeds 0 and 2
- the remaining seed-2 CLS branch should be treated as partial evidence, not missing required work

### Cross-Environment Sanity Checks

To test whether the current local setup generalizes beyond PointMaze, pretrained planning sanity checks were also started for `Wall` and `PushT`.

- `Wall`
  - config: `conf/plan_wall_wsl.yaml`
  - output: `plan_outputs/20260322164743_wall_single_gH5`
  - result: `success_rate = 1.0`, `mean_state_dist = 1.7964`, `mean_visual_dist = 0.5799`
- `PushT`
  - config: `conf/plan_pusht_wsl.yaml`
  - reduced-budget output: `plan_outputs/20260322165419_pusht_gH5`
  - stronger-budget retry output: `plan_outputs/20260322190605_pusht_gH5`
  - reduced-budget status: stalled with `mpc/success_rate = 0.0` through step `74`
  - stronger-budget retry status: improved `mean_state_dist` into the `60-70` range by step `52`, but still stalled before `final_eval`

## Estimated Remaining Time

For the current closeout path, the estimated remaining time is:

- metric aggregation and final doc cleanup: about `0.5` to `1.0` hour

Estimated total remaining time:

- about `0.5` to `1.0` hour if the project closes after the patch-focused reevaluation

This estimate assumes no new model integrations and no major WSL/CUDA instability.

## Collaboration Notes

If you are joining the project now, start here:

- status and runnable commands:
  [docs/PROJECT_README.md](/C:/Users/zack/Documents/GNN3/docs/PROJECT_README.md)
- exact execution history and engineering fixes:
  [docs/EXECUTION_LOG.md](/C:/Users/zack/Documents/GNN3/docs/EXECUTION_LOG.md)
- experiment table:
  [docs/EXPERIMENT_TRACKER.csv](/C:/Users/zack/Documents/GNN3/docs/EXPERIMENT_TRACKER.csv)
- report draft:
  [docs/COURSE_REPORT_DRAFT.md](/C:/Users/zack/Documents/GNN3/docs/COURSE_REPORT_DRAFT.md)
- near-term plan:
  [docs/COURSE_PROJECT_PLAN.md](/C:/Users/zack/Documents/GNN3/docs/COURSE_PROJECT_PLAN.md)

Recommended working rule:

- treat the current project scope as `patch/CLS x deterministic/gaussian`
- do not claim anything from `success_rate=1.0` alone
- use `mean_state_dist` and larger-sample follow-up as the main planning evidence

## Possible Next Tracks

At this point, several technical directions are all reasonable. They can move in parallel depending on what people want to pick up next.

### Track A: Final Evaluation and Metric Cleanup

Useful work here:

- summarize the larger-sample outcomes using `mean_state_dist` rather than relying on saturated `success_rate`
- mark the remaining seed-2 CLS reevaluation attempts as partial optional evidence
- finish the final tables and report wording cleanup
- if time allows, test one slightly harder planning setting, such as a stricter evaluation budget or one additional environment like `PushT` or `Wall`

### Track B: Expanded Representation Path

Useful work here:

- prototype one additional encoder family, preferably `DINOv3` or `V-JEPA`
- aim for the minimum viable integration:
  - encoder loading
  - latent shape compatibility
  - one sanity training run
  - one sanity planning run
- document what changes are needed compared with the current DINOv2-based path

### Track C: Results Consolidation and Final Write-Up

Useful work here:

- build one clean final table for the formal 3-seed matrix
- build one clean final table for the larger-sample follow-up
- make sure `README`, `PROJECT_README`, `EXECUTION_LOG`, and the report use the same numbers and conclusions
- tighten the wording around:
  - `success_rate` saturation
  - why `mean_state_dist` is the primary metric
  - what is completed versus what remains optional
- add a concise abstract, limitations section, and final conclusion paragraph

### Track D: Reproducibility and Repo Handoff

Useful work here:

- verify that the documented commands still match the current scripts
- prepare a short rerun guide for the main completed experiments
- make checkpoint names, output folders, and tracker rows easier to cross-reference
- sync the most important docs and status summaries to the collaboration repositories

## Checklist

- [x] WSL2 environment setup
- [x] Pretrained PointMaze planning sanity check
- [x] Deterministic patch training and planning
- [x] Deterministic CLS training and planning
- [x] Gaussian patch training and planning
- [x] Gaussian CLS training and planning
- [x] Full 3-seed formal matrix
- [x] Seed-0 larger-sample follow-up
- [x] Seed-1 patch deterministic larger-sample follow-up
- [x] Seed-1 patch gaussian larger-sample follow-up
- [x] Seed-1 CLS gaussian larger-sample follow-up
- [~] Seed-1 CLS deterministic larger-sample follow-up
  partial result only; shard B completed, shard A dropped after repeated instability
- [x] Seed-2 patch-focused larger-sample follow-up
- [ ] Seed-2 CLS larger-sample follow-up
- [ ] Final larger-sample aggregation table
- [ ] Final report wording cleanup

### Target Completion Window

If the current reevaluation plan continues without new blockers:

- remaining optional CLS experiments: within `1` to `2` hours
- final aggregation and document cleanup: within `0.5` to `1.0` hour
- full current-scope wrap-up: within `1.5` to `3.0` hours

If the team decides not to pursue the optional seed-2 CLS follow-up, a polished submission-ready version is much closer:

- final documentation and report cleanup: within `0.5` to `1.0` hour
- stable current-scope submission package: within the same session

## Acknowledgment

This project builds directly on the original DINO-WM work by:

- Gaoyue Zhou
- Hengkai Pan
- Yann LeCun
- Lerrel Pinto

Affiliations:

- New York University
- Meta AI

Original project links:

- Paper: <https://arxiv.org/abs/2411.04983>
- Homepage: <https://dino-wm.github.io/>
- Original code: <https://github.com/gaoyuezhou/dino_wm>

Please retain attribution to the original authors when sharing derived materials from this course-project extension.

## Citation

```bibtex
@article{zhou2024dino,
  title={DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning},
  author={Zhou, Gaoyue and Pan, Hengkai and LeCun, Yann and Pinto, Lerrel},
  journal={arXiv preprint arXiv:2411.04983},
  year={2024}
}
```
