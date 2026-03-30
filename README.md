# DINO-WM Course Project

Course-project extension of DINO-WM on PointMaze, centered on two questions:

1. Does `DINOv2 patch` remain stronger than `DINOv2 CLS` for action-conditioned world modeling?
2. Does a lightweight `gaussian` latent transition improve planning over the original `deterministic` predictor?

This repository contains the engineering code, experiment artifacts, and paper materials for our final project study.

## Overview

We keep the original DINO-WM pipeline fixed:

`image encoder -> latent transition model -> planning`

Within that pipeline, the completed project is organized as two complementary studies:

- `Experiment I`: encoder comparison under a shared PointMaze recipe
- `Experiment II`: within-DINOv2 ablation on `patch/CLS` and `deterministic/gaussian`

The final paper reports both studies while treating `mean_state_dist` as the primary planning metric under the current evaluation regime.

## Team Contributions

This repository reflects joint work by:

- `Ke Tan`
- `Weiguo Li`
- `Zhaokun Wang`

At a high level, the completed project combines:

- engineering adaptation of DINO-WM for local WSL2-based execution
- implementation of the Gaussian latent-transition variant
- PointMaze training and planning experiments
- encoder-comparison and ablation analysis
- paper writing, figures, and bibliography curation

## Final Project Scope

Implemented and used in the final report:

- DINO-WM setup and local execution pipeline
- `DINOv2 patch`
- `DINOv2 CLS`
- deterministic latent transition model
- Gaussian latent transition model
- PointMaze evaluation

Also included in the final paper as a completed comparison study:

- `V-JEPA 2`
- `VFM-VAE / SigLIP2`

Not part of the completed main evidence base:

- `DINOv3`
- `DINO-Tok`

## Main Results

### Formal Three-Seed Matrix

| Setting | Mean train loss | Mean val loss | Mean state dist |
| --- | ---: | ---: | ---: |
| `DINOv2 patch + deterministic` | `0.0825` | `0.0429` | `3.2806` |
| `DINOv2 CLS + deterministic` | `0.0411` | `0.0240` | `3.3047` |
| `DINOv2 patch + gaussian` | `-2.9815` | `-3.1785` | `3.7772` |
| `DINOv2 CLS + gaussian` | `-3.0751` | `-3.2282` | `3.3277` |

Main takeaways:

- `DINOv2 patch + deterministic` remains the strongest completed baseline
- `DINOv2 CLS + deterministic` is more competitive than initially expected
- Gaussian variants improve likelihood-style training losses
- those training improvements do not translate into a stable planning gain

### Larger-Sample Follow-Up

The larger-sample PointMaze follow-up was used to go beyond the saturated `n_evals=3` planning view. Under this follow-up:

- deterministic patch remains the most reliable patch-side baseline
- `mean_state_dist` is more informative than saturated `success_rate`
- CLS remains competitive, but coverage is less complete than for the patch branch

## Repository Layout

- [paper/](/C:/Users/zack/Documents/GNN3/paper): paper source, figures, and report materials
- [docs/](/C:/Users/zack/Documents/GNN3/docs): project notes, execution history, and planning documents
- [scripts/](/C:/Users/zack/Documents/GNN3/scripts): helper scripts for experiments and analysis
- [course_project/](/C:/Users/zack/Documents/GNN3/course_project): project-specific code and utilities

Important supporting files:

- [plan.py](/C:/Users/zack/Documents/GNN3/plan.py)
- [paper/main.tex](/C:/Users/zack/Documents/GNN3/paper/main.tex)

## Running the Project

The local project setup was developed in `WSL2 Ubuntu 24.04` with Python 3.9 and MuJoCo 2.1 compatibility fixes for the original DINO-WM stack.

Project-specific local configs referenced during the study include:

- `conf/train_wsl.yaml`
- `conf/plan_point_maze_wsl.yaml`
- `conf/predictor/vit_gaussian.yaml`

Additional execution details are documented in:

- [docs/PROJECT_README.md](/C:/Users/zack/Documents/GNN3/docs/PROJECT_README.md)
- [docs/EXECUTION_LOG.md](/C:/Users/zack/Documents/GNN3/docs/EXECUTION_LOG.md)

## Acknowledgment

This project builds directly on the original DINO-WM work by:

- Gaoyue Zhou
- Hengkai Pan
- Yann LeCun
- Lerrel Pinto

Original project links:

- Paper: <https://arxiv.org/abs/2411.04983>
- Homepage: <https://dino-wm.github.io/>
- Original code: <https://github.com/gaoyuezhou/dino_wm>

Please retain attribution to the original authors when sharing derived materials from this course-project extension.

## Citation

If you want to cite the original DINO-WM work:

```bibtex
@article{zhou2024dino,
  title={DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning},
  author={Zhou, Gaoyue and Pan, Hengkai and LeCun, Yann and Pinto, Lerrel},
  journal={arXiv preprint arXiv:2411.04983},
  year={2024}
}
```
