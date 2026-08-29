# run_v0.3.6_500iters.ps1
# v0.3.6: Seed-fixed Architecture optimization (2D: patch_len, stride)
# Seed: 42 (reproducibility guaranteed)
# Fixed: Focal Loss (v0.2 best), LoRA (baseline rank=16, alpha=32)
# Budget: 500 iterations

param(
    [int]$TotalBudget = 500,
    [switch]$DryRun
)

$PYTHON = ".venv-codagt\Scripts\python.exe"
$ARCHIVE = "ptst_dgm/results/ptst_archive_v0.3.6.jsonl"

if (-not (Test-Path $PYTHON)) {
    Write-Host "[Error] Python not found: $PYTHON" -ForegroundColor Red
    exit 1
}

Write-Host "===== v0.3.6: Seed-Fixed Architecture Optimization (500 iters) =====" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Experiment   : v0.3.6 (Seed-fixed Architecture optimization)"
Write-Host "  Budget       : $TotalBudget iterations"
Write-Host "  Archive      : $ARCHIVE"
Write-Host "  Search space : 2D Architecture (patch_len [20,32], stride [10,20])"
Write-Host "  Fixed params : Focal Loss (v0.2 best), LoRA (rank=16, alpha=32)"
Write-Host "  Seed         : 42 (FIXED - reproducibility guaranteed)"
Write-Host ""
Write-Host "Previous experiments:" -ForegroundColor Yellow
Write-Host "  v0.3 (seed=unknown): F1=0.7726, patch=26, stride=16, 50 iters"
Write-Host "  v0.3.5 Quick Test (seeds 0-9): Best F1=0.5338 (seed 5)"
Write-Host ""
Write-Host "Expected outcome:" -ForegroundColor Yellow
Write-Host "  With seed=42 fixed:"
Write-Host "    - Results are fully reproducible"
Write-Host "    - Baseline quality depends on seed 42 initialization"
Write-Host "    - 500 iters gives 10x more exploration than original v0.3"
Write-Host "    - Goal: Find best Architecture with reproducible training"
Write-Host ""

# Confirm execution
if (-not $DryRun) {
    $confirm = Read-Host "Start v0.3.6 (500 iters, ~21 hours)? (y/n)"
    if ($confirm -ne "y") {
        Write-Host "Aborted by user." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host "Starting v0.3.6 optimization..." -ForegroundColor Green
Write-Host ""

# Build command
$args = @(
    "-m", "ptst_dgm.multi_objective_agent.ptst_loop",
    "--total-budget", $TotalBudget,
    "--archive", $ARCHIVE
)

if ($DryRun) {
    $args += "--dry-run"
    Write-Host "[DRY RUN MODE] No actual training" -ForegroundColor Magenta
}

# Execute
$startTime = Get-Date
& $PYTHON $args
$exitCode = $LASTEXITCODE
$endTime = Get-Date
$duration = $endTime - $startTime

if ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "[Success] v0.3.6 optimization completed!" -ForegroundColor Green
    Write-Host "Duration: $($duration.ToString())" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Analyze results:"
    Write-Host "     Get-Content $ARCHIVE | ConvertFrom-Json | Sort-Object macro_f1 -Descending | Select-Object -First 10"
    Write-Host "  2. Compare with original v0.3:"
    Write-Host "     v0.3 (seed unknown): F1=0.7726"
    Write-Host "     v0.3.6 (seed=42): Check best F1 from archive"
    Write-Host "  3. Visualize Pareto frontier:"
    Write-Host "     python ptst_dgm/scripts/visualize_pareto.py --archive $ARCHIVE"
} else {
    Write-Host ""
    Write-Host "[Error] v0.3.6 optimization failed with exit code $exitCode" -ForegroundColor Red
}

exit $exitCode
