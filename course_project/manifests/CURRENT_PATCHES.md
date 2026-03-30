# Current Root-Level Patches

These files currently differ from a clean upstream-style layout and should be
treated as documented exceptions until they can be wrapped or migrated.

## Actively Modified Root/Upstream-Like Files

- `plan.py`
- `scripts/wsl_pointmaze_large_eval_single.sh`

## Active Project-Owned Content Still Outside `course_project/`

- `docs/`
- `paper/`
- `logs/`
- `Sync/`
- `plan_outputs/`
- `course_runs/`

## Post-Deadline Refactor Direction

- Move paper assets into `course_project/paper/`
- Move experiment tracking into `course_project/docs/` and
  `course_project/experiments/trackers/`
- Move custom automation into `course_project/scripts/`
- Keep root scripts only when they are required to preserve upstream-compatible
  entrypoints
