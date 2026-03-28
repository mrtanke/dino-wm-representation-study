import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def make_run_dir(base_dir: Path, run_name: str | None) -> Path:
    if run_name:
        run_dir = base_dir / run_name
    else:
        run_dir = base_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_command(args: argparse.Namespace, repo_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(repo_root / "train.py"),
        "--config-name",
        args.config_name,
        f"training.epochs={args.epochs}",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
    ]
    if args.extra_override:
        cmd.extend(args.extra_override)
    return cmd


def latest_checkpoint(run_dir: Path) -> Path:
    return run_dir / "checkpoints" / "model_latest.pth"


def tail_log_lines(run_dir: Path, num_lines: int = 20) -> str:
    hydra_yaml = run_dir / "hydra.yaml"
    if not hydra_yaml.exists():
        return "hydra.yaml not found yet."

    log_candidates = sorted(run_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not log_candidates:
        return "No log file found in run directory yet."

    log_path = log_candidates[-1]
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        return f"Failed to read log: {exc}"
    return "\n".join(lines[-num_lines:])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one or more epochs of DINO-WM training in a fixed run directory."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Persistent directory for run folders, e.g. /content/drive/MyDrive/dino_wm_runs/vjepa",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Existing run folder name to resume, or a new name to create.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="train_vjepa_wsl",
        help="Hydra config name to use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="How many additional epochs to run this time.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to the repo root. Defaults to the parent of this script.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Optional DATASET_DIR env var.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        help="WANDB_MODE env var. Default disables wandb in Colab.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Extra Hydra override. Repeat this flag for multiple overrides.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve() if args.repo_root else infer_repo_root()
    base_dir = args.base_dir.resolve()
    run_dir = make_run_dir(base_dir, args.run_name)

    env = os.environ.copy()
    env["WANDB_MODE"] = args.wandb_mode
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(repo_root)
    )
    if args.dataset_dir:
        env["DATASET_DIR"] = args.dataset_dir

    ckpt = latest_checkpoint(run_dir)
    mode = "resume" if ckpt.exists() else "fresh"

    cmd = build_command(args, repo_root)

    print(f"[colab_epoch_runner] repo_root={repo_root}")
    print(f"[colab_epoch_runner] run_dir={run_dir}")
    print(f"[colab_epoch_runner] mode={mode}")
    if ckpt.exists():
        print(f"[colab_epoch_runner] resume_ckpt={ckpt}")
    print(f"[colab_epoch_runner] command={' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=run_dir, env=env)
    print(f"[colab_epoch_runner] exit_code={result.returncode}")

    if latest_checkpoint(run_dir).exists():
        print(f"[colab_epoch_runner] latest_ckpt={latest_checkpoint(run_dir)}")
    print("[colab_epoch_runner] log_tail:")
    print(tail_log_lines(run_dir))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
