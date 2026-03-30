# Course Project Layer

This folder is the project-owned layer that sits on top of the upstream-style
`dino_wm` repository layout.

## Goal

Keep the repository root as close as possible to the original upstream
`dino_wm` structure, while moving course-project assets, merged collaborator
artifacts, and paper materials into a separate namespace.

## Design Rule

- Root level:
  upstream-compatible code and configs
- `course_project/`:
  project-owned docs, paper assets, experiment exports, trackers, scripts, and
  manifests

## Recommended Ownership Split

- Upstream-like root:
  `assets/`, `conf/`, `datasets/`, `distributed_fn/`, `env/`, `metrics/`,
  `models/`, `planning/`, `train.py`, `plan.py`, `utils.py`,
  `custom_resolvers.py`, `preprocessor.py`, `environment.yaml`
- Course project layer:
  `course_project/docs/`
  `course_project/paper/`
  `course_project/scripts/`
  `course_project/experiments/`
  `course_project/manifests/`

## Current Status

This folder is the first consolidation step. It does not yet move the active
paper and docs out of their current paths. Instead, it establishes the target
layout and starts consolidating external teammate data under the project-owned
namespace.
