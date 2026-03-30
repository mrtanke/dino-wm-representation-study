$ErrorActionPreference = "Stop"

$RepoDir = "C:\Users\zack\Documents\GNN3"
$LogDir = Join-Path $RepoDir "logs\overnight_closeout"
$StatusDir = Join-Path $RepoDir "logs\large_eval_status"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $StatusDir | Out-Null

function Invoke-WslStep {
    param(
        [string]$Label,
        [string]$BashCommand,
        [int]$TimeoutMinutes = 120
    )

    $stdout = Join-Path $LogDir "$Label.out.log"
    $stderr = Join-Path $LogDir "$Label.err.log"
    $doneFlag = Join-Path $StatusDir "$Label.done"
    $failFlag = Join-Path $StatusDir "$Label.fail"

    Remove-Item $doneFlag -ErrorAction SilentlyContinue
    Remove-Item $failFlag -ErrorAction SilentlyContinue

    $proc = Start-Process `
        -FilePath "wsl.exe" `
        -ArgumentList @("-d", "Ubuntu-24.04", "--", "bash", "-lc", $BashCommand) `
        -PassThru `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden

    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)

    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 20
        if ((Get-Date) -gt $deadline) {
            try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
            & wsl --shutdown | Out-Null
            Add-Content -Path $stderr -Value "Timed out after $TimeoutMinutes minutes."
            return $false
        }
    }

    if (Test-Path $doneFlag) {
        return $true
    }

    if ($proc.ExitCode -eq 0 -and -not (Test-Path $doneFlag)) {
        Add-Content -Path $stderr -Value "Process exited without done flag."
    }

    return $false
}

function Invoke-WslPlanSanity {
    param(
        [string]$Label,
        [string]$ConfigName,
        [string]$ModelName,
        [int]$TimeoutMinutes = 60
    )

    $stdout = Join-Path $LogDir "$Label.out.log"
    $stderr = Join-Path $LogDir "$Label.err.log"

    $bashCommand = @"
cd /mnt/c/Users/zack/Documents/GNN3
export DATASET_DIR="\$HOME/dino_wm_data"
export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}:\$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export MUJOCO_PY_MUJOCO_PATH="\$HOME/.mujoco/mujoco210"
export MUJOCO_GL="egl"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export WANDB_MODE=offline
\$HOME/.local/bin/micromamba run -n dino_wm python plan.py \
  --config-name $ConfigName \
  hydra/launcher=basic \
  ckpt_base_path=\$HOME/dino_wm_ckpts \
  model_name=$ModelName \
  n_evals=3 \
  n_plot_samples=0 \
  plot_rollouts=False \
  save_video=False \
  planner.sub_planner.num_samples=32 \
  planner.sub_planner.topk=8 \
  planner.sub_planner.opt_steps=3 \
  planner.n_taken_actions=1
"@

    $proc = Start-Process `
        -FilePath "wsl.exe" `
        -ArgumentList @("-d", "Ubuntu-24.04", "--", "bash", "-lc", $bashCommand) `
        -PassThru `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden

    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 20
        if ((Get-Date) -gt $deadline) {
            try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
            & wsl --shutdown | Out-Null
            Add-Content -Path $stderr -Value "Timed out after $TimeoutMinutes minutes."
            return $false
        }
    }

    return ($proc.ExitCode -eq 0)
}

$steps = @(
    @{
        label = "seed2_cls_gauss_n5a_retry"
        type = "large_eval"
        timeout = 120
        bash = "cd /mnt/c/Users/zack/Documents/GNN3 && env N_EVALS=5 PLAN_SAMPLES=16 PLAN_TOPK=4 PLAN_OPT_STEPS=2 PLOT_ROLLOUTS=False SAVE_VIDEO=False N_PLOT_SAMPLES=0 bash scripts/wsl_pointmaze_large_eval_single.sh seed2_cls_gauss_n5a_retry 2026-03-19/00-16-03 2"
    }
    @{
        label = "seed2_cls_gauss_n5b"
        type = "large_eval"
        timeout = 120
        bash = "cd /mnt/c/Users/zack/Documents/GNN3 && env N_EVALS=5 PLAN_SAMPLES=16 PLAN_TOPK=4 PLAN_OPT_STEPS=2 PLOT_ROLLOUTS=False SAVE_VIDEO=False N_PLOT_SAMPLES=0 bash scripts/wsl_pointmaze_large_eval_single.sh seed2_cls_gauss_n5b 2026-03-19/00-16-03 7"
    }
    @{
        label = "pusht_pretrained_sanity"
        type = "sanity"
        timeout = 60
        config = "plan_pusht.yaml"
        model = "pusht"
    }
    @{
        label = "wall_pretrained_sanity"
        type = "sanity"
        timeout = 60
        config = "plan_wall.yaml"
        model = "wall_single"
    }
)

$summary = Join-Path $LogDir "overnight_summary.log"
"Overnight closeout queue started at $(Get-Date -Format s)" | Set-Content $summary

foreach ($step in $steps) {
    Add-Content $summary "Starting $($step.label) at $(Get-Date -Format s)"

    if ($step.type -eq "large_eval") {
        $ok = Invoke-WslStep -Label $step.label -BashCommand $step.bash -TimeoutMinutes $step.timeout
        if (-not $ok) {
            Add-Content $summary "$($step.label): stopped or stalled"
            if ($step.label -like "*n5a*") {
                Add-Content $summary "Stopping queue after shard-A failure."
                break
            }
            continue
        }
        Add-Content $summary "$($step.label): completed"
        continue
    }

    if ($step.type -eq "sanity") {
        $ok = Invoke-WslPlanSanity -Label $step.label -ConfigName $step.config -ModelName $step.model -TimeoutMinutes $step.timeout
        if ($ok) {
            Add-Content $summary "$($step.label): completed"
        } else {
            Add-Content $summary "$($step.label): failed or timed out"
        }
    }
}

Add-Content $summary "Overnight closeout queue finished at $(Get-Date -Format s)"
