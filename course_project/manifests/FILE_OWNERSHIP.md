# File Ownership Rules

## Upstream-Owned or Upstream-Shaped

These should remain stable and close to the original `dino_wm` repository
structure:

- core code in `models/`, `planning/`, `metrics/`, `env/`
- Hydra configs in `conf/`
- top-level training and planning entrypoints

## Project-Owned

These should live under `course_project/`:

- paper writing
- experiment summaries
- collaborator sync exports
- custom scripts for closeout and reporting
- manifests describing local decisions and patches

## External or Imported

These should be stored as imports rather than mixed with hand-maintained source:

- teammate W\&B exports
- cloned external repos
- one-off sync snapshots
