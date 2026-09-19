# v0.3.7: Architecture Optimization with Model Checkpointing (300 iterations)
# - Same strategy as v0.3.6 (2D Architecture, seed=42 fixed)
# - NEW: Save best model every 60 iterations
# - NEW CONSTRAINT: patch_len > stride (enforced in ptst_sampler.py)
# - Budget: 300 iterations (more efficient than 500)
# - Checkpoint interval: 60 iterations (5 checkpoints total)

Write-Host "=" -ForegroundColor Cyan
Write-Host "v0.3.7: Architecture Optimization + Checkpoint Strategy (300 iters)" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  - Search space: 2D (patch_len, stride)" -ForegroundColor White
Write-Host "  - Constraint: patch_len > stride (improved search)" -ForegroundColor Magenta
Write-Host "  - Fixed params: Focal Loss (v0.2 best), LoRA (baseline), seed=42" -ForegroundColor White
Write-Host "  - Budget: 300 iterations" -ForegroundColor White
Write-Host "  - Checkpoint: Every 60 iterations" -ForegroundColor White
Write-Host "  - Optimizer: NSGA-II (Pareto frontier)" -ForegroundColor White
Write-Host "  - Objectives: 15 (5 metrics × 3 horizons)" -ForegroundColor White
Write-Host ""
Write-Host "Innovation:" -ForegroundColor Green
Write-Host "  - Saves best model every 60 iterations" -ForegroundColor White
Write-Host "  - Constraint ensures patch_len > stride for better overlap" -ForegroundColor White
Write-Host "  - No more 'lost best model' incidents" -ForegroundColor White
Write-Host "  - 5 checkpoints → models/v0.3.7_300iters_checkpoints/" -ForegroundColor White
Write-Host ""

# Activate DGM environment
& .\.venv-codagt\Scripts\Activate.ps1

# Verify ptst_sampler.py is configured for v0.3.7
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

# Verify constraint: patch_len > stride
if ($samplerContent -match "max_stride\s*=\s*min\(20,\s*params\[`"patch_len`"\]\s*-\s*1\)") {
    Write-Host "  ✓ Constraint: patch_len > stride" -ForegroundColor Green
} else {
    Write-Host "  ✗ WARNING: Constraint 'patch_len > stride' not detected" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting v0.3.7 DGM run (300 iterations)..." -ForegroundColor Cyan
Write-Host "Expected duration: ~2.5-3 hours" -ForegroundColor Gray
Write-Host ""

$startTime = Get-Date

# Run DGM loop with checkpoint enabled
python -m ptst_dgm.multi_objective_agent.ptst_loop `
    --model codestral:latest `
    --archive ptst_dgm/results/ptst_archive_v0.3.7_300iters.jsonl `
    --total-budget 300 `
    --checkpoint-interval 60 `
    --population-size 20 `
    --python-exe .venv-ptstf\Scripts\python.exe `
    --script ptst_dgm\training\train_patchtst_dgm.py `
    --data-path data\golden_testset `
    --output-dir ptst_dgm\results\temp_model `
    --epochs 100

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "=" -ForegroundColor Cyan
Write-Host "v0.3.7 (300 iters) Complete!" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan
Write-Host "Duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
Write-Host ""

# Display checkpoint summary
Write-Host "Checkpoint Summary:" -ForegroundColor Cyan
if (Test-Path "models\v0.3.7_300iters_checkpoints\*.pt") {
    Get-ChildItem models\v0.3.7_300iters_checkpoints\*.pt | Sort-Object Name | ForEach-Object {
        $json = Get-Content $_.FullName.Replace(".pt", ".json") | ConvertFrom-Json
        Write-Host "  [$($json.iteration.ToString().PadLeft(3))] Trial #$($json.trial_number.ToString().PadLeft(4)) | F1=$($json.macro_f1.ToString('0.0000')) | $($_.Name)" -ForegroundColor White
    }
    
    # Find overall best
    $allCheckpoints = Get-ChildItem models\v0.3.7_300iters_checkpoints\*.json | ForEach-Object {
        $json = Get-Content $_.FullName | ConvertFrom-Json
        [PSCustomObject]@{
            Path = $_.FullName.Replace(".json", ".pt")
            F1 = $json.macro_f1
            Trial = $json.trial_number
            Iteration = $json.iteration
        }
    }
    $best = $allCheckpoints | Sort-Object F1 -Descending | Select-Object -First 1
    
    Write-Host ""
    Write-Host "Best Checkpoint:" -ForegroundColor Green
    Write-Host "  Trial #$($best.Trial) | F1=$($best.F1.ToString('0.0000')) | Iteration $($best.Iteration)" -ForegroundColor Yellow
    Write-Host "  Model: $(Split-Path $best.Path -Leaf)" -ForegroundColor White
} else {
    Write-Host "  No checkpoints found (dry run or error?)" -ForegroundColor Red
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review results: ptst_dgm/results/ptst_archive_v0.3.7_300iters_log.jsonl" -ForegroundColor White
Write-Host "  2. Visualize Pareto: python ptst_dgm/scripts/visualize_pareto.py" -ForegroundColor White
Write-Host "  3. Test best model: Load checkpoint and run inference" -ForegroundColor White
Write-Host "  4. Create result report: RESULT_v0.3.7_300iters.md" -ForegroundColor White
Write-Host ""
