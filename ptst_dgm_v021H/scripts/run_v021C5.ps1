# run_v021C5.ps1
# Multi-Objective Pareto DGM for PatchTST v0.2.1C5 - Focal Loss optimization with relaxed constraint.
#
# Experiment Configuration (based on v0.2 best results):
#   - Control: 3D Focal Loss (alpha, gamma, w_normal)
#   - Constraint: w_normal + w_anomal = 5.88 (v0.2 best: w_n=1.85, w_a=4.03)
#   - Fixed Architecture: patch_len=26, stride=16 (v0.2.3 best from 8D optimization)
#   - Fixed LoRA: rank=16, alpha=47 (v0.2.3 best from 8D optimization)
#   - Budget: 1000 iterations
#   - Checkpoint: every 100 iterations
#   - Goal: macro F1 > 0.655
#
# Background:
#   v0.2.1C achieved macro F1=0.637 with constraint w_n+w_a=2.0, but failed to reach
#   the target 0.655. Analysis showed the constraint was too restrictive. v0.2's best
#   solution had w_n=1.85, w_a=4.03 (sum=5.88, ratio=2.18). This experiment relaxes
#   the constraint to match v0.2's optimal weight sum.
#
# Usage:
#   .\ptst_dgm_v021C5\scripts\run_v021C5.ps1
#   .\ptst_dgm_v021C5\scripts\run_v021C5.ps1 -TotalBudget 1000 -CheckpointInterval 100

param(
    [string]$Model              = "codestral:latest",
    [int]   $TotalBudget        = 1000,
    [int]   $PopulationSize     = 20,
    [int]   $Epochs             = 100,
    [int]   $CheckpointInterval = 100,
    [string]$DataPath           = "data\golden_testset",
    [string]$Archive            = "ptst_dgm_v021C5\results\ptst_archive_v021C5.jsonl",
    [string]$OutputDir          = "ptst_dgm_v021C5\results\temp_model",
    [switch]$DryRun
)

$env:OLLAMA_NUM_GPU       = "1"
$env:CUDA_VISIBLE_DEVICES = "0"

$DGM_PYTHON   = ".venv-codagt\Scripts\python.exe"
$TRAIN_PYTHON = ".venv-ptstf\Scripts\python.exe"
$LOOP_SCRIPT  = "ptst_dgm_v021C5\multi_objective_agent\ptst_loop.py"
$TRAIN_SCRIPT = "ptst_dgm_v021C5\training\train_patchtst_dgm.py"

Write-Host "===== Multi-Objective Pareto DGM v0.2.1C5 - Focal Loss + Relaxed Constraint =====" -ForegroundColor Cyan
Write-Host "Control: Focal Loss 3D (alpha, gamma, w_normal)" -ForegroundColor Yellow
Write-Host "Constraint: w_normal + w_anomal = 5.88 (v0.2 best sum)" -ForegroundColor Yellow
Write-Host "Fixed: Architecture (patch=26, stride=16), LoRA (rank=16, alpha=47)" -ForegroundColor Yellow
Write-Host "Budget: $TotalBudget iters | Checkpoint: every $CheckpointInterval iters" -ForegroundColor Yellow
Write-Host "Goal: macro F1 > 0.655 (v0.2.1C achieved: 0.637)" -ForegroundColor Yellow
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
    & .\ptst_dgm_v021C5\scripts\init_baseline.ps1 -DataPath $DataPath -Archive $Archive -Epochs $Epochs
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

# ── DGM loop ──────────────────────────────────────────────────────────────────
$dgm_args = @(
    "-m", "ptst_dgm_v021C5.multi_objective_agent.ptst_loop",
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
Write-Host "===== DGM v0.2.1C5 Completed =====" -ForegroundColor Green
Write-Host "Results saved to: $Archive" -ForegroundColor White
Write-Host "Check models\v021C5_checkpoints for saved models" -ForegroundColor White
Write-Host ""
