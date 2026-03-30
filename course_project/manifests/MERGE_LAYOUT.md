# Merge Layout Plan

## Canonical Repository Choice

Use this repository as the canonical merged repo because it already contains:

- the active paper draft
- the experiment tracker
- the course-project execution log
- the generated figures
- the controlled PointMaze study

Teammate repository assets should be imported into this repo selectively.

## Structural Rule

Preserve the upstream `dino_wm` shape at the root whenever possible.

That means:

- do not scatter course-specific notes and exports across root-level upstream
  folders
- keep project-specific additions under `course_project/`
- keep any unavoidable root-level patches documented in
  `course_project/manifests/CURRENT_PATCHES.md`

## Target Layout

```text
repo_root/
  assets/
  conf/
  datasets/
  distributed_fn/
  env/
  metrics/
  models/
  planning/
  train.py
  plan.py
  utils.py
  ...
  course_project/
    docs/
    paper/
    scripts/
    experiments/
      trackers/
      exports/
        teammate/
    manifests/
```

## Migration Priority

1. Keep root runnable.
2. Move project-owned artifacts before touching core upstream-like code.
3. Document every remaining root-level patch.
4. Only refactor root code after the paper deadline if the patch can be moved
   into wrappers or extension modules cleanly.
