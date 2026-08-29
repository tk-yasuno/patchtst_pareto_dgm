# v0.3.8: Architecture Optimization WITHOUT Constraint (500 iterations)
# - Same strategy as v0.3.7 BUT: NO constraint (patch_len > stride)
# - 2D Architecture (patch_len, stride) with full exploration
# - seed=42 fixed (deterministic mode)
# - Save best model every 60 iterations
# - Budget: 500 iterations (comprehensive search)
# - Checkpoint interval: 60 iterations

Write-Host "=" -ForegroundColor Cyan
Write-Host "v0.3.8: Architecture Optimization WITHOUT Constraint (500 iters)" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  - Search space: 2D (patch_len, stride)" -ForegroundColor White
Write-Host "  - Constraint: REMOVED (v0.3.7 constraint was too restrictive)" -ForegroundColor Red
Write-Host "  - Fixed params: Focal Loss (v0.2 best), LoRA (baseline), seed=42" -ForegroundColor White
Write-Host "  - Budget: 500 iterations" -ForegroundColor White
Write-Host "  - Checkpoint: Every 60 iterations" -ForegroundColor White
Write-Host "  - Optimizer: NSGA-II (Pareto frontier)" -ForegroundColor White
Write-Host "  - Objectives: 15 (5 metrics × 3 horizons)" -ForegroundColor White
Write-Host ""
Write-Host "Hypothesis:" -ForegroundColor Green
Write-Host "  - v0.3.7 failed due to overly restrictive constraint" -ForegroundColor White
Write-Host "  - Removing constraint should recover v0.3 performance" -ForegroundColor White
Write-Host "  - 500 iters provides comprehensive exploration" -ForegroundColor White
Write-Host ""

# Activate DGM environment
& .\.venv-codagt\Scripts\Activate.ps1

# Verify ptst_sampler.py is configured for v0.3.8
Write-Host "Verifying ptst_sampler.py configuration..." -ForegroundColor Yellow
$samplerContent = Get-Content ptst_dgm\multi_objective_agent\ptst_sampler.py -Raw
if ($samplerContent -match "FIXED_SEED\s*=\s*42") {
    Write-Host "  ✓ FIXED_SEED = 42" -ForegroundColor Green
} else {
    Write-Host "  ✗ ERROR: FIXED_SEED is not 42" -ForegroundColor Red
    exit 1
}

if ($samplerContent -match '"patch_len":\s*\(20,\s*32\)') {
    Write-Host "  ✓ patch_len range: (20, 32)" -ForegroundColor Green
} else {
    Write-Host "  ✗ ERROR: patch_len range is incorrect" -ForegroundColor Red
    exit 1
}

if ($samplerContent -match '"stride":\s*\(10,\s*20\)') {
    Write-Host "  ✓ stride range: (10, 20)" -ForegroundColor Green
} else {
    Write-Host "  ✗ ERROR: stride range is incorrect" -ForegroundColor Red
    exit 1
}

# Verify NO constraint in suggest() method
if ($samplerContent -match 'max_stride\s*=\s*min\(20,\s*params\["patch_len"\]\s*-\s*1\)') {
    Write-Host "  ✗ ERROR: Constraint still present in suggest() method!" -ForegroundColor Red
    Write-Host "    Please update ptst_sampler.py to remove constraint logic" -ForegroundColor Red
    exit 1
} else {
    Write-Host "  ✓ Constraint REMOVED (full exploration enabled)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting v0.3.8 run..." -ForegroundColor Cyan
Write-Host ""

# Run optimization
$startTime = Get-Date
python -m ptst_dgm.multi_objective_agent.ptst_loop `
    --total-budget 500 `
    --archive ptst_dgm/results/ptst_archive_v0.3.8_500iters.jsonl `
    --checkpoint-interval 60

$endTime = Get-Date
$duration = $endTime - $startTime
$durationStr = "{0:hh\:mm\:ss}" -f ([datetime]$duration.Ticks)

Write-Host ""
Write-Host "=" -ForegroundColor Cyan
Write-Host "v0.3.8 (500 iters) Complete!" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan
Write-Host "Duration: $durationStr" -ForegroundColor Yellow
Write-Host ""

# Extract and display checkpoint summary
Write-Host "Checkpoint Summary:" -ForegroundColor Yellow
if (Test-Path "models\v0.3.8_500iters_checkpoints") {
    $checkpoints = Get-ChildItem "models\v0.3.8_500iters_checkpoints\*.pt" | Sort-Object Name
    foreach ($ckpt in $checkpoints) {
        if ($ckpt.Name -match "trial_(\d+)_iter_(\d+)_f1_([0-9.]+)\.pt") {
            $trial = $matches[1]
            $iter = $matches[2]
            $f1 = $matches[3]
            Write-Host "  [$($iter.PadLeft(3))] Trial # $($trial.PadLeft(3)) | F1=$f1 | $($ckpt.Name)" -ForegroundColor White
        }
    }
    Write-Host ""
    
    # Find best checkpoint
    $bestCkpt = $checkpoints | Where-Object { $_.Name -match "_f1_" } | 
                Sort-Object { [double]($_.Name -replace '.*_f1_([0-9.]+)\.pt', '$1') } -Descending | 
                Select-Object -First 1
    if ($bestCkpt) {
        if ($bestCkpt.Name -match "trial_(\d+)_iter_(\d+)_f1_([0-9.]+)\.pt") {
            Write-Host "Best Checkpoint:" -ForegroundColor Green
            Write-Host "  Trial #$($matches[1]) | F1=$($matches[3]) | Iteration $($matches[2])" -ForegroundColor White
            Write-Host "  Model: $($bestCkpt.Name)" -ForegroundColor White
        }
    }
}
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Compare with v0.3 and v0.3.7: python ptst_dgm/scripts/compare_versions.py" -ForegroundColor White
Write-Host "  2. Visualize Pareto: python ptst_dgm/scripts/visualize_pareto_frontier.py --pareto-file ptst_dgm/results/ptst_archive_v0.3.8_500iters_pareto.jsonl --output-dir ptst_dgm/results/visualizations_v0.3.8_500iters" -ForegroundColor White
Write-Host "  3. Test best model: Load checkpoint and run inference" -ForegroundColor White
Write-Host "  4. Create result report: RESULT_v0.3.8_500iters.md" -ForegroundColor White
Write-Host ""
