# Migration Map

This map shows where the current active project-owned files should move once the
repository is fully reorganized.

## Current -> Target

- `docs/COURSE_PROJECT_PLAN.md`
  -> `course_project/docs/COURSE_PROJECT_PLAN.md`
- `docs/EXECUTION_LOG.md`
  -> `course_project/docs/EXECUTION_LOG.md`
- `docs/EXPERIMENT_TRACKER.csv`
  -> `course_project/experiments/trackers/EXPERIMENT_TRACKER.csv`
- `docs/PROJECT_README.md`
  -> `course_project/docs/PROJECT_README.md`
- `paper/main.tex`
  -> `course_project/paper/main.tex`
- `paper/references.bib`
  -> `course_project/paper/references.bib`
- `paper/generate_figures.py`
  -> `course_project/paper/generate_figures.py`
- `paper/figure/*`
  -> `course_project/paper/figure/*`
- `scripts/start_wsl_pointmaze_large_eval_bg.ps1`
  -> `course_project/scripts/start_wsl_pointmaze_large_eval_bg.ps1`
- `scripts/wsl_resume_train_in_dir.sh`
  -> `course_project/scripts/wsl_resume_train_in_dir.sh`
- `scripts/run_overnight_closeout.ps1`
  -> `course_project/scripts/run_overnight_closeout.ps1`
- `scripts/run_overnight_encoder_feasibility.ps1`
  -> `course_project/scripts/run_overnight_encoder_feasibility.ps1`
- `Sync/`
  -> `course_project/experiments/exports/external_snapshots/`
- `logs/`
  -> `course_project/experiments/runtime_logs/`
- `plan_outputs/`
  -> `course_project/experiments/plan_outputs/`
- `course_runs/`
  -> `course_project/experiments/course_runs/`

## Keep At Root

These should remain at the root unless there is a clean wrapper-based refactor:

- `train.py`
- `plan.py`
- `models/`
- `planning/`
- `conf/`
- `datasets/`
- `assets/`
- `env/`
- `metrics/`

## Recommended Merge Order

1. move docs and paper
2. move project-owned scripts
3. move runtime artifacts and sync snapshots
4. update references and paths
5. only then reduce root-level patches if needed
