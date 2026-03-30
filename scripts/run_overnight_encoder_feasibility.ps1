$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$PSNativeCommandUseErrorActionPreference = $false

$Root = "C:\Users\zack\Documents\GNN3"
$LogDir = Join-Path $Root "logs\overnight_encoder_feasibility"
$CacheRoot = "C:\Users\zack\ModelCache"
$RepoDir = Join-Path $CacheRoot "VFM-VAE"
$VfmDir = Join-Path $CacheRoot "VFM-VAE-model"
$VjepaDir = Join-Path $CacheRoot "VJEPA2-vitl"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null
New-Item -ItemType Directory -Force -Path $VfmDir | Out-Null
New-Item -ItemType Directory -Force -Path $VjepaDir | Out-Null

$SummaryLog = Join-Path $LogDir "summary.log"

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$stamp] $Message"
    $line | Tee-Object -FilePath $SummaryLog -Append
}

Write-Log "Overnight encoder feasibility run started."

if (-not (Test-Path $RepoDir)) {
    Write-Log "Cloning official VFM-VAE repository."
    git clone https://github.com/tianciB/VFM-VAE $RepoDir 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "git_clone_vfm_vae.log") -Append
} else {
    Write-Log "Updating existing VFM-VAE repository clone."
    git -C $RepoDir pull 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "git_pull_vfm_vae.log") -Append
}

$DownloadScript = Join-Path $LogDir "download_checkpoints.py"
@'
from huggingface_hub import hf_hub_download

jobs = [
    (
        "tiancibi/VFM-VAE",
        "checkpoints_imagenet256/vfm_vae/vfm_vae_f16d32_siglip2_44m_after_stage_3_patchgan_fine_tuning_legacy.pth",
        r"C:\Users\zack\ModelCache\VFM-VAE-model",
    ),
    (
        "facebook/vjepa2-vitl-fpc64-256",
        "original/model.pth",
        r"C:\Users\zack\ModelCache\VJEPA2-vitl",
    ),
]

for repo_id, filename, local_dir in jobs:
    print(f"DOWNLOAD_START {repo_id} {filename}")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"DOWNLOAD_DONE {repo_id} -> {path}")
'@ | Set-Content -Path $DownloadScript

Write-Log "Downloading VFM-VAE stage-3 checkpoint and V-JEPA vitl original checkpoint."
python -X utf8 $DownloadScript 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "download_checkpoints.log") -Append

$InspectScript = Join-Path $LogDir "inspect_checkpoints.py"
@'
import os
import torch

targets = [
    (
        "vfm_vae_stage3",
        r"C:\Users\zack\ModelCache\VFM-VAE-model\checkpoints_imagenet256\vfm_vae\vfm_vae_f16d32_siglip2_44m_after_stage_3_patchgan_fine_tuning_legacy.pth",
    ),
    (
        "vjepa2_vitl_original",
        r"C:\Users\zack\ModelCache\VJEPA2-vitl\original\model.pth",
    ),
]

for name, path in targets:
    print(f"INSPECT_START {name} {path}")
    if not os.path.exists(path):
        print("MISSING", path)
        continue
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print("TOP_LEVEL_KEYS", keys[:30])
        for key in keys[:10]:
            val = obj[key]
            if isinstance(val, dict):
                nested = list(val.keys())[:20]
                print(f"NESTED_KEYS[{key}]", nested)
            else:
                print(f"VALUE_TYPE[{key}]", type(val).__name__)
    else:
        print("OBJECT_TYPE", type(obj).__name__)
    print(f"INSPECT_DONE {name}")
'@ | Set-Content -Path $InspectScript

Write-Log "Inspecting downloaded checkpoint structures."
python -X utf8 $InspectScript 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "inspect_checkpoints.log") -Append

$WslLinkScript = Join-Path $LogDir "link_vjepa_vitl.sh"
@'
mkdir -p /home/zack/jepawm_ossckpt/vjepa2_hf_vitl
cp -f /mnt/c/Users/zack/ModelCache/VJEPA2-vitl/original/model.pth /home/zack/jepawm_ossckpt/vjepa2_hf_vitl/model.pth
ls -lh /home/zack/jepawm_ossckpt/vjepa2_hf_vitl/model.pth
'@ | Set-Content -Path $WslLinkScript

Write-Log "Copying V-JEPA vitl checkpoint into the WSL checkpoint area."
wsl bash -lc "sed -i 's/\r$//' /mnt/c/Users/zack/Documents/GNN3/logs/overnight_encoder_feasibility/link_vjepa_vitl.sh && bash /mnt/c/Users/zack/Documents/GNN3/logs/overnight_encoder_feasibility/link_vjepa_vitl.sh" 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "wsl_link_vjepa_vitl.log") -Append

Write-Log "Overnight encoder feasibility run completed."
