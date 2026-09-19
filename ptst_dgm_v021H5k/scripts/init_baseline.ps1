# init_baseline.ps1
# Train PatchTST baseline with v0.2.1H5k configuration, then initialize the DGM archive.
#
# v0.2.1H5k baseline uses:
#   - Fixed Architecture: patch_len=26, stride=16 (v0.2.3 best)
#   - Fixed LoRA: rank=16, alpha=47 (v0.2.3 best)
#   - Focal Loss: horizon-specific defaults (alpha=0.75, gamma=1.0, w_normal=1.0, w_anomal=1.0 for all horizons)
#
# Run once before starting the DGM loop:
#   .\ptst_dgm_v021H5k\scripts\init_baseline.ps1

param(
    [string]$DataPath  = "data\golden_testset",
    [string]$Archive   = "ptst_dgm_v021H5k\results\ptst_archive_v021H5k.jsonl",
    [int]   $Epochs    = 100
)

$TRAIN_PYTHON = ".venv-ptstf\Scripts\python.exe"
$DGM_PYTHON   = ".venv-codagt\Scripts\python.exe"

# Derive paths from Archive parameter
$ArchiveDir = Split-Path $Archive -Parent
$TRAIN_SCRIPT = "ptst_dgm_v021H5k\training\train_patchtst_dgm.py"
$INIT_SCRIPT  = "ptst_dgm_v021H5k\scripts\init_baseline_archive.py"
$EVAL_JSON    = Join-Path $ArchiveDir "baseline_eval.json"
$OUTPUT_DIR   = Join-Path $ArchiveDir "baseline_model"

Write-Host "===== PatchTST DGM v0.2.1H5k - Baseline Initialisation =====" -ForegroundColor Cyan
Write-Host "Architecture: v0.2.3 best (patch=26, stride=16)" -ForegroundColor Yellow
Write-Host "LoRA: v0.2.3 best (rank=16, alpha=47)" -ForegroundColor Yellow
Write-Host "Focal Loss: horizon-specific (all horizons: alpha=0.75, gamma=1.0, w_n=1.0, w_a=1.0)" -ForegroundColor Yellow
Write-Host ""

foreach ($exe in @($TRAIN_PYTHON, $DGM_PYTHON)) {
    if (-not (Test-Path $exe)) {
        Write-Host "[Error] Not found: $exe" -ForegroundColor Red
        exit 1
    }
}

# -- Step 1: Train baseline ----------------------------------------------------
Write-Host "[Step 1] Training PatchTST baseline..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null

& $TRAIN_PYTHON $TRAIN_SCRIPT `
    --focal-alpha-30d 0.75 `
    --focal-gamma-30d 1.0 `
    --w-normal-30d 1.0 `
    --w-anomal-30d 1.0 `
    --focal-alpha-60d 0.75 `
    --focal-gamma-60d 1.0 `
    --w-normal-60d 1.0 `
    --w-anomal-60d 1.0 `
    --focal-alpha-90d 0.75 `
    --focal-gamma-90d 1.0 `
    --w-normal-90d 1.0 `
    --w-anomal-90d 1.0 `
    --patch-len 26 `
    --stride 16 `
    --lora-rank 16 `
    --lora-alpha 47 `
    --epochs $Epochs `
    --data-path $DataPath `
    --output-dir $OUTPUT_DIR `
    --output-json $EVAL_JSON

if ($LASTEXITCODE -ne 0) {
    Write-Host "[Error] Baseline training failed" -ForegroundColor Red; exit 1
}
Write-Host "[Step 1] Done - metrics saved to $EVAL_JSON" -ForegroundColor Green

# -- Step 2: Initialise archive ------------------------------------------------
Write-Host ""
Write-Host "[Step 2] Initialising DGM archive..." -ForegroundColor Yellow

& $DGM_PYTHON $INIT_SCRIPT `
    --eval-json $EVAL_JSON `
    --archive $Archive

if ($LASTEXITCODE -ne 0) {
    Write-Host "[Error] Archive initialisation failed" -ForegroundColor Red; exit 1
}
Write-Host "[Step 2] Done - archive created: $Archive" -ForegroundColor Green

Write-Host ""
Write-Host "===== Baseline Initialisation Complete =====" -ForegroundColor Green
Write-Host "You can now run the DGM loop:" -ForegroundColor White
Write-Host "  .\ptst_dgm_v021H\scripts\run_v021H.ps1" -ForegroundColor Cyan
Write-Host ""
