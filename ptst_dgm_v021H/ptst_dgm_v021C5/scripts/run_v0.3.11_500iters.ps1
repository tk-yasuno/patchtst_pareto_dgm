# run_v0.3.11_500iters.ps1
# Phase 4 Architecture Optimization (v0.3.11)
#
# Key changes from v0.3.10:
#   - Reverted to v0.2 seed strategy: seed=None (random initialization each time)
#   - Purpose: Increase exploration diversity, avoid local optima
#   - Same as v0.2's successful approach → F1=0.776 @ 500 iters
#
# v0.3.9 issue analysis:
#   - Random seed (0-9999) as optimization parameter prevented NSGA-II learning
#   - Same architecture with different seeds gave different results
#   - NSGA-II couldn't learn "which architecture is better"
#
# v0.3.10 issue analysis:
#   - Fixed seed=42 for complete reproducibility
#   - BUT: Less diversity than v0.2, may need more iterations
#
# v0.3.11 solution:
#   - Seed=None (no fixing) → random initialization like v0.2
#   - NSGA-II seed=42 (fixed) → consistent Pareto optimization
#   - Best of both: reproducible optimization + diverse exploration

param(
    [int]$TotalBudget = 500,
    [int]$CheckpointEvery = 60
)

$ErrorActionPreference = "Stop"
$env:OLLAMA_NUM_GPU = "1"
$env:CUDA_VISIBLE_DEVICES = "0"

$DGM_PYTHON = ".venv-codagt\Scripts\python.exe"
$TRAIN_PYTHON = ".venv-ptstf\Scripts\python.exe"
$ARCHIVE = "ptst_dgm\results\ptst_archive_v0.3.11_500iters.jsonl"
$DATA_PATH = "data\golden_testset"
$OUTPUT_DIR = "ptst_dgm\results\temp_model"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " PatchTST DGM - Phase 4 (v0.3.11)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Version        : v0.3.11 (v0.2 seed strategy)"
Write-Host "  Control vars   : patch_len [20-32], stride [10-20]"
Write-Host "  Fixed params   : Focal Loss (v0.2 best), LoRA r=16/α=32"
Write-Host "  Seed strategy  : None (random init like v0.2)"
Write-Host "  Optimizer      : NSGA-II (seed=42, 15 objectives)"
Write-Host "  Budget         : $TotalBudget iterations"
Write-Host "  Checkpoint     : Every $CheckpointEvery iterations"
Write-Host "  Archive        : $ARCHIVE"
Write-Host ""

# Initialize archive from v0.3.6 baseline (seed=42 baseline evaluation)
if (-not (Test-Path $ARCHIVE)) {
    Write-Host "[Init] Creating initial archive from v0.3.6 baseline..." -ForegroundColor Yellow
    & $DGM_PYTHON ptst_dgm\scripts\init_baseline_archive.py `
        --eval-json ptst_dgm\results\baseline_eval.json `
        --archive $ARCHIVE
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[Error] Baseline initialization failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "[Init] Initial archive created (F1=0.4797 baseline)`n" -ForegroundColor Green
} else {
    $lineCount = (Get-Content $ARCHIVE | Measure-Object -Line).Lines
    Write-Host "[Init] Resuming from existing archive ($lineCount agents)`n" -ForegroundColor Green
}

Write-Host "Starting optimization loop...`n" -ForegroundColor Green

& $DGM_PYTHON -m ptst_dgm.multi_objective_agent.ptst_loop `
    --model "codestral:latest" `
    --archive $ARCHIVE `
    --total-budget $TotalBudget `
    --population-size 20 `
    --python-exe $TRAIN_PYTHON `
    --script "ptst_dgm\training\train_patchtst_dgm.py" `
    --data-path $DATA_PATH `
    --output-dir $OUTPUT_DIR `
    --epochs 100 `
    --checkpoint-interval $CheckpointEvery

$exit_code = $LASTEXITCODE
Write-Host ""

if ($exit_code -eq 0) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " v0.3.11 Complete" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Results:" -ForegroundColor Yellow
    Write-Host "  Archive : $ARCHIVE"
    $logPath = $ARCHIVE -replace '\.jsonl$', '_log.jsonl'
    $paretoPath = $ARCHIVE -replace '\.jsonl$', '_pareto.jsonl'
    if (Test-Path $logPath) { Write-Host "  Log     : $logPath" }
    if (Test-Path $paretoPath) { Write-Host "  Pareto  : $paretoPath" }
    Write-Host ""
    
    # Display best result
    $lastEntry = Get-Content $ARCHIVE | Select-Object -Last 1 | ConvertFrom-Json
    if ($lastEntry) {
        Write-Host "Latest agent:" -ForegroundColor Yellow
        Write-Host "  macro_F1 = $($lastEntry.macro_f1)"
        Write-Host "  patch_len = $($lastEntry.patch_len), stride = $($lastEntry.stride)"
    }
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host " Execution failed (exit code $exit_code)" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
}

exit $exit_code
