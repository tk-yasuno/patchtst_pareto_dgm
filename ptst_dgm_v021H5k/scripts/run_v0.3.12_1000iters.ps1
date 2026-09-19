# run_v0.3.12_1000iters.ps1
# Phase 4 Architecture Optimization (v0.3.12)
#
# Experiment Goal: Test Fixed Seed Strategy for Scientific Reproducibility
#
# Key changes from v0.3.11:
#   - FIXED seed=42 (not random) → Complete reproducibility
#   - Purpose: Enable NSGA-II to learn true architecture quality without seed noise
#   - Extended budget: 1000 iterations (vs 500 in v0.3.11)
#
# Rationale from LESSON_v0.3.11_seed_UPDATED.md:
#   - v0.3.11 with seed=None failed: F1=0.5775 (same params explored 14x → best F1=0.5312)
#   - Phase 4 is HIGH VARIANCE problem: seed noise >> architecture signal
#   - Fixed seed eliminates seed lottery, lets NSGA-II compare architectures fairly
#
# Expected Outcome:
#   - Slower convergence (single initialization path)
#   - But stable, monotonic improvement
#   - Should find architecture quality patterns (e.g., "patch=26 > patch=30")
#
# Trade-offs:
#   - Pros: Reproducibility, NSGA-II can learn, scientific validity
#   - Cons: Less diversity, may miss optimal regions, requires many iterations

param(
    [int]$TotalBudget = 1000,
    [int]$CheckpointEvery = 100
)

$ErrorActionPreference = "Stop"
$env:OLLAMA_NUM_GPU = "1"
$env:CUDA_VISIBLE_DEVICES = "0"

$DGM_PYTHON = ".venv-codagt\Scripts\python.exe"
$TRAIN_PYTHON = ".venv-ptstf\Scripts\python.exe"
$ARCHIVE = "ptst_dgm\results\ptst_archive_v0.3.12_1000iters.jsonl"
$DATA_PATH = "data\golden_testset"
$OUTPUT_DIR = "ptst_dgm\results\temp_model"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " PatchTST DGM - Phase 4 (v0.3.12)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "🔬 Scientific Reproducibility Experiment" -ForegroundColor Magenta
Write-Host "   Testing: Fixed Seed=42 Strategy`n" -ForegroundColor Magenta

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Version        : v0.3.12 (Fixed Seed Strategy)"
Write-Host "  Control vars   : patch_len [20-32], stride [10-20]"
Write-Host "  Fixed params   : Focal Loss (v0.2 best), LoRA r=16/α=32"
Write-Host "  Seed strategy  : 42 (FIXED for reproducibility)" -ForegroundColor Green
Write-Host "  Optimizer      : NSGA-II (seed=42, 15 objectives)"
Write-Host "  Budget         : $TotalBudget iterations (extended)" -ForegroundColor Green
Write-Host "  Checkpoint     : Every $CheckpointEvery iterations"
Write-Host "  Archive        : $ARCHIVE"
Write-Host ""

Write-Host "Hypothesis:" -ForegroundColor Yellow
Write-Host "  • Eliminating seed variance → NSGA-II can learn architecture quality" -ForegroundColor White
Write-Host "  • 1000 iters with fixed seed > 500 iters with random seed" -ForegroundColor White
Write-Host "  • Expect monotonic improvement, not lucky discoveries`n" -ForegroundColor White

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
Write-Host "⏱️  Estimated runtime: ~24-48 hours (1000 iterations)`n" -ForegroundColor Cyan

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
    Write-Host " v0.3.12 Complete" -ForegroundColor Green
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
    Write-Host "Best Result:" -ForegroundColor Cyan
    $best = Get-Content $ARCHIVE | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.macro_f1 } | Sort-Object -Property macro_f1 -Descending | Select-Object -First 1
    if ($best) {
        Write-Host "  Trial #     : $($best.trial_number)" -ForegroundColor White
        Write-Host "  Macro F1    : $($best.macro_f1)" -ForegroundColor Yellow
        Write-Host "  Mean FPR    : $($best.mean_fpr)" -ForegroundColor White
        Write-Host "  patch_len   : $($best.patch_len)" -ForegroundColor Cyan
        Write-Host "  stride      : $($best.stride)" -ForegroundColor Cyan
        Write-Host ""
        
        # Compare with v0.3.11
        Write-Host "Comparison with v0.3.11 (seed=None, 500 iters):" -ForegroundColor Yellow
        Write-Host "  v0.3.11 best: F1=0.5775 (patch=30, stride=17)" -ForegroundColor Red
        Write-Host "  v0.3.12 best: F1=$($best.macro_f1) (patch=$($best.patch_len), stride=$($best.stride))" -ForegroundColor Green
        $improvement = (($best.macro_f1 - 0.5775) / 0.5775) * 100
        if ($improvement -gt 0) {
            Write-Host "  Improvement : +$([Math]::Round($improvement, 2))%`n" -ForegroundColor Green
        } else {
            Write-Host "  Change      : $([Math]::Round($improvement, 2))%`n" -ForegroundColor Red
        }
    }
    
    Write-Host "========================================`n" -ForegroundColor Green
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host " v0.3.12 Failed (exit code: $exit_code)" -ForegroundColor Red
    Write-Host "========================================`n" -ForegroundColor Red
    exit $exit_code
}
