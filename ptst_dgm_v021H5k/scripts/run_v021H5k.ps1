# run_v021H5k.ps1
# Multi-Objective Pareto DGM for PatchTST v0.2.1H5k - Horizon-specific 9D optimization (5000 iterations).
#
# Experiment Configuration:
#   - Control: 9D Horizon-specific Focal Loss (3 params × 3 horizons)
#     * 30d: alpha, gamma, w_normal
#     * 60d: alpha, gamma, w_normal  
#     * 90d: alpha, gamma, w_normal
#   - Constraint: w_normal_XXd + w_anomal_XXd = 5.88 (per-horizon)
#   - Fixed Architecture: patch_len=26, stride=16 (v0.2.3 best from 8D optimization)
#   - Fixed LoRA: rank=16, alpha=47 (v0.2.3 best from 8D optimization)
#   - Budget: 5000 iterations (extended from v0.2.1H's 2000 iterations)
#   - Checkpoint: every 100 iterations
#   - Goal: macro F1 > 0.655
#
# Background:
#   v0.2.1H (2000 iters): Achieved significant results
#   v0.2.1H5k extends exploration with 5000 iterations to further optimize
#   
#   This 9D approach may discover horizon-specific optimal loss configurations
#   with deeper exploration.
#
# Usage:
#   .\ptst_dgm_v021H5k\scripts\run_v021H5k.ps1
#   .\ptst_dgm_v021H5k\scripts\run_v021H5k.ps1 -TotalBudget 5000 -CheckpointInterval 100

param(
    [string]$Model              = "codestral:latest",
    [int]   $TotalBudget        = 5000,
    [int]   $PopulationSize     = 20,
    [int]   $Epochs             = 100,
    [int]   $CheckpointInterval = 100,
    [string]$DataPath           = "data\golden_testset",
    [string]$Archive            = "ptst_dgm_v021H5k\results\ptst_archive_v021H5k.jsonl",
    [string]$OutputDir          = "ptst_dgm_v021H5k\results\temp_model",
    [switch]$DryRun
)

$env:OLLAMA_NUM_GPU       = "1"
$env:CUDA_VISIBLE_DEVICES = "0"

$DGM_PYTHON   = ".venv-codagt\Scripts\python.exe"
$TRAIN_PYTHON = ".venv-ptstf\Scripts\python.exe"
$LOOP_SCRIPT  = "ptst_dgm_v021H5k\multi_objective_agent\ptst_loop.py"
$TRAIN_SCRIPT = "ptst_dgm_v021H5k\training\train_patchtst_dgm.py"

Write-Host "===== Multi-Objective Pareto DGM v0.2.1H5k - Horizon-specific 9D Optimization (5000 iters) =====" -ForegroundColor Cyan
Write-Host "Control: 9D Horizon-specific Focal Loss (3 params × 3 horizons)" -ForegroundColor Yellow
Write-Host "  30d/60d/90d: each has independent alpha, gamma, w_normal" -ForegroundColor Yellow
Write-Host "Constraint: w_normal_XXd + w_anomal_XXd = 5.88 (per-horizon)" -ForegroundColor Yellow
Write-Host "Fixed: Architecture (patch=26, stride=16), LoRA (rank=16, alpha=47)" -ForegroundColor Yellow
Write-Host "Budget: $TotalBudget iters | Checkpoint: every $CheckpointInterval iters" -ForegroundColor Yellow
Write-Host "Goal: macro F1 > 0.655" -ForegroundColor Yellow
Write-Host ""

foreach ($exe in @($DGM_PYTHON, $TRAIN_PYTHON)) {
    if (-not (Test-Path $exe)) {
        Write-Host "[Error] Virtual environment not found: $exe" -ForegroundColor Red
        exit 1
    }
}

# ── Archive bootstrap ─────────────────────────────────────────────────────────
if (-not (Test-Path $Archive)) {
    Write-Host "[Init] Archive not found - running baseline init first..." -ForegroundColor Yellow
    & .\ptst_dgm_v021H5k\scripts\init_baseline.ps1 -DataPath $DataPath -Archive $Archive -Epochs $Epochs
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

# ── DGM loop ──────────────────────────────────────────────────────────────────
$dgm_args = @(
    "-m", "ptst_dgm_v021H5k.multi_objective_agent.ptst_loop",
    "--model", $Model,
    "--total-budget", $TotalBudget,
    "--population-size", $PopulationSize,
    "--epochs", $Epochs,
    "--checkpoint-interval", $CheckpointInterval,
    "--data-path", $DataPath,
    "--archive", $Archive,
    "--output-dir", $OutputDir,
    "--python-exe", $TRAIN_PYTHON,
    "--script", $TRAIN_SCRIPT
)
if ($DryRun) { $dgm_args += "--dry-run" }

Write-Host "[DGM] Starting Multi-Objective Pareto optimization loop (9D, 5000 iters)..." -ForegroundColor Green
Write-Host "  Model: $Model" -ForegroundColor White
Write-Host "  Total Budget: $TotalBudget iterations" -ForegroundColor White
Write-Host "  Population Size: $PopulationSize" -ForegroundColor White
Write-Host "  Checkpoint Interval: $CheckpointInterval iterations" -ForegroundColor White
Write-Host "  Training Epochs: $Epochs" -ForegroundColor White
Write-Host "  Archive: $Archive" -ForegroundColor White
Write-Host "  Output Dir: $OutputDir" -ForegroundColor White
Write-Host ""

& $DGM_PYTHON @dgm_args
if ($LASTEXITCODE -ne 0) {
    Write-Host "[Error] DGM loop failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===== DGM v0.2.1H5k Completed =====" -ForegroundColor Green
Write-Host "Results saved to: $Archive" -ForegroundColor White
Write-Host "Check models\v021H5k_checkpoints for saved models" -ForegroundColor White
