$ErrorActionPreference = "Stop"

if ($args.Count -lt 3) {
    throw "usage: .\scripts\start_wsl_pointmaze_large_eval_bg.ps1 <label> <model_name> <seed> [n_evals] [plan_samples] [plan_topk] [plan_opt_steps]"
}

$Label = $args[0]
$ModelName = $args[1]
$Seed = [int]$args[2]
$NEvals = if ($args.Count -ge 4) { [int]$args[3] } else { 5 }
$PlanSamples = if ($args.Count -ge 5) { [int]$args[4] } else { 16 }
$PlanTopK = if ($args.Count -ge 6) { [int]$args[5] } else { 4 }
$PlanOptSteps = if ($args.Count -ge 7) { [int]$args[6] } else { 2 }

$repo = "C:\Users\zack\Documents\GNN3"
$stdoutWin = Join-Path $repo ("logs\{0}_stdout.log" -f $Label)
$stderrWin = Join-Path $repo ("logs\{0}_stderr.log" -f $Label)
$pidWin = Join-Path $repo ("logs\{0}.pid" -f $Label)

$stdoutWsl = "/mnt/c/Users/zack/Documents/GNN3/logs/${Label}_stdout.log"
$stderrWsl = "/mnt/c/Users/zack/Documents/GNN3/logs/${Label}_stderr.log"
$pidWsl = "/mnt/c/Users/zack/Documents/GNN3/logs/${Label}.pid"
$scriptWsl = "/mnt/c/Users/zack/Documents/GNN3/scripts/wsl_pointmaze_large_eval_single.sh"

$bashCmd = @"
cd /mnt/c/Users/zack/Documents/GNN3
nohup env \
  N_EVALS=$NEvals \
  PLAN_SAMPLES=$PlanSamples \
  PLAN_TOPK=$PlanTopK \
  PLAN_OPT_STEPS=$PlanOptSteps \
  PLOT_ROLLOUTS=False \
  SAVE_VIDEO=False \
  N_PLOT_SAMPLES=0 \
  $scriptWsl $Label '$ModelName' $Seed \
  > '$stdoutWsl' 2> '$stderrWsl' < /dev/null &
echo $! > '$pidWsl'
cat '$pidWsl'
"@

$bgPid = wsl -e bash -lc $bashCmd

[pscustomobject]@{
    Label = $Label
    PID = ($bgPid | Select-Object -Last 1).Trim()
    StdoutLog = $stdoutWin
    StderrLog = $stderrWin
    PidFile = $pidWin
}
