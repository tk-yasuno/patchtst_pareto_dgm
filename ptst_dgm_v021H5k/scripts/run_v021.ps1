# run_v021.ps1
# Multi-Objective Pareto DGM for PatchTST v0.2.1 - Focal Loss optimization.
#
# Experiment Configuration:
#   - Control: 4D Focal Loss (alpha, gamma, w_normal, w_anomal)
#   - Fixed Architecture: patch_len=29, stride=16 (v0.3.6 best)
#   - Fixed LoRA: rank=19, alpha=53 (v0.3.6.3 best)
#   - Budget: 1000 iterations
#   - Checkpoint: every 100 iterations
#
# Usage:
#   .\ptst_dgm_v021\scripts\run_v021.ps1
#   .\ptst_dgm_v021\scripts\run_v021.ps1 -TotalBudget 1000 -CheckpointInterval 100

param(
    [string]$Model              = "codestral:latest",
    [int]   $TotalBudget        = 1000,
    [int]   $PopulationSize     = 20,
    [int]   $Epochs             = 100,
    [int]   $CheckpointInterval = 100,
    [string]$DataPath           = "data\golden_testset",
    [string]$Archive            = "ptst_dgm_v021\results\ptst_archive_v021.jsonl",
    [string]$OutputDir          = "ptst_dgm_v021\results\temp_model",
    [switch]$DryRun
)

$env:OLLAMA_NUM_GPU       = "1"
$env:CUDA_VISIBLE_DEVICES = "0"

$DGM_PYTHON   = ".venv-codagt\Scripts\python.exe"
$TRAIN_PYTHON = ".venv-ptstf\Scripts\python.exe"
$LOOP_SCRIPT  = "ptst_dgm_v021\multi_objective_agent\ptst_loop.py"
$TRAIN_SCRIPT = "ptst_dgm_v021\training\train_patchtst_dgm.py"

Write-Host "===== Multi-Objective Pareto DGM v0.2.1 - Focal Loss Optimization =====" -ForegroundColor Cyan
Write-Host "Control: Focal Loss 4D (alpha, gamma, w_normal, w_anomal)" -ForegroundColor Yellow
Write-Host "Fixed: Architecture (patch=29, stride=16), LoRA (rank=19, alpha=53)" -ForegroundColor Yellow
Write-Host "Budget: $TotalBudget iters | Checkpoint: every $CheckpointInterval iters" -ForegroundColor Yellow
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
    & .\ptst_dgm_v021\scripts\init_baseline.ps1 -DataPath $DataPath -Archive $Archive -Epochs $Epochs
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

# ── DGM loop ──────────────────────────────────────────────────────────────────
$dgm_args = @(
    "-m", "ptst_dgm_v021.multi_objective_agent.ptst_loop",
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

Write-Host "[DGM] Starting Multi-Objective Pareto optimization loop..." -ForegroundColor Green
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
Write-Host "===== DGM v0.2.1 Completed =====" -ForegroundColor Green
Write-Host "Results saved to: $Archive" -ForegroundColor White
Write-Host ""
