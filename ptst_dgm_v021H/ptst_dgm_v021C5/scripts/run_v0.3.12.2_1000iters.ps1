# run_v0.3.12.2_1000iters.ps1
# Phase 4 Architecture Optimization (v0.3.12.2)
#
# Final Configuration: v0.3 Success Pattern Reproduction
#
# Key settings:
#   - Seed: None (random init like v0.3 - diversity over reproducibility)
#   - Golden Ratio Guidance: ENABLED (overlap ≈ 61.8%)
#   - NSGA-II: seed=42 (consistent Pareto optimization)
#   - Budget: 1000 iterations
#
# Why v0.3.12.2 (not continuing v0.3.12):
#   - v0.3.12 (73 iters) used seed=42 (fixed) → low diversity
#   - v0.3.12.2 uses seed=None → matches v0.3 successful strategy
#   - Fresh start with correct configuration from iteration 0
#
# Expected Outcome:
#   - "Lucky discovery" pattern like v0.3 Trial #1 (F1=0.7726)
#   - Golden ratio guidance + seed diversity → explore good overlap regions
#   - Higher variance but potentially higher peaks

param(
    [int]$TotalBudget = 1000,
    [int]$CheckpointEvery = 100
)

$ErrorActionPreference = "Stop"
$env:OLLAMA_NUM_GPU = "1"
$env:CUDA_VISIBLE_DEVICES = "0"

$DGM_PYTHON = ".venv-codagt\Scripts\python.exe"
$TRAIN_PYTHON = ".venv-ptstf\Scripts\python.exe"
$ARCHIVE = "ptst_dgm\results\ptst_archive_v0.3.12.2_1000iters.jsonl"
$DATA_PATH = "data\golden_testset"
$OUTPUT_DIR = "ptst_dgm\results\temp_model"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " PatchTST DGM - Phase 4 (v0.3.12.2)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "🎯 v0.3 Success Pattern Reproduction" -ForegroundColor Magenta
Write-Host "   Strategy: seed=None + Golden Ratio Guidance`n" -ForegroundColor Magenta

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Version        : v0.3.12.2 (v0.3 strategy + Golden Ratio + wider range)"
Write-Host "  Control vars   : patch_len [15-40], stride [5-25]" -ForegroundColor Cyan
Write-Host "  Fixed params   : Focal Loss (v0.2 best), LoRA r=16/α=32"
Write-Host "  Seed strategy  : None (random init like v0.3)" -ForegroundColor Green
Write-Host "  LLM guidance   : Golden Ratio (overlap ≈ 61.8%)" -ForegroundColor Green
Write-Host "  Optimizer      : NSGA-II (seed=42, 15 objectives)"
Write-Host "  Budget         : $TotalBudget iterations"
Write-Host "  Checkpoint     : Every $CheckpointEvery iterations"
Write-Host "  Archive        : $ARCHIVE"
Write-Host ""

Write-Host "Expected Behavior:" -ForegroundColor Yellow
Write-Host "  • Each trial uses different random seed → high diversity" -ForegroundColor White
Write-Host "  • Golden ratio guidance → prefers overlap ≈ 38-62%" -ForegroundColor White
Write-Host "  • Looking for 'Lucky Trial' pattern (like v0.3 Trial #1)" -ForegroundColor White
Write-Host "  • Higher variance, potentially higher peaks`n" -ForegroundColor White

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
Write-Host "⏱️  Estimated runtime: 24-48 hours (1000 iterations)`n" -ForegroundColor Cyan

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
    Write-Host " v0.3.12.2 Complete" -ForegroundColor Green
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
        Write-Host "  overlap     : $([Math]::Round((1 - $best.stride/$best.patch_len)*100, 1))%" -ForegroundColor Cyan
        Write-Host ""
        
        # Compare with v0.3
        Write-Host "Comparison with v0.3 (50 iters, seed=None):" -ForegroundColor Yellow
        Write-Host "  v0.3 Trial #1 : F1=0.7726 (patch=26, stride=16, overlap=38.5%)" -ForegroundColor Green
        Write-Host "  v0.3.12.2 best: F1=$($best.macro_f1) (patch=$($best.patch_len), stride=$($best.stride), overlap=$([Math]::Round((1 - $best.stride/$best.patch_len)*100, 1))%)" -ForegroundColor Cyan
        $improvement = (($best.macro_f1 - 0.7726) / 0.7726) * 100
        if ($improvement -gt 0) {
            Write-Host "  Improvement : +$([Math]::Round($improvement, 2))%`n" -ForegroundColor Green
        } else {
            Write-Host "  Change      : $([Math]::Round($improvement, 2))%`n" -ForegroundColor Red
        }
    }
    
    Write-Host "========================================`n" -ForegroundColor Green
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host " v0.3.12.2 Failed (exit code: $exit_code)" -ForegroundColor Red
    Write-Host "========================================`n" -ForegroundColor Red
    exit $exit_code
}
