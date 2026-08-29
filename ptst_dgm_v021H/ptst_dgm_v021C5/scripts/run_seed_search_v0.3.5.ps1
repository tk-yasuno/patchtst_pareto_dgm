# run_seed_search_v0.3.5.ps1
# Wrapper script for v0.3.5 seed search
#
# Usage:
#   .\ptst_dgm\scripts\run_seed_search_v0.3.5.ps1 -Start 0 -End 100
#   .\ptst_dgm\scripts\run_seed_search_v0.3.5.ps1 -Start 100 -End 200 -Resume
#   .\ptst_dgm\scripts\run_seed_search_v0.3.5.ps1 -QuickTest  # Test first 10 seeds

param(
    [int]$Start = 0,
    [int]$End = 1000,
    [int]$Epochs = 100,
    [double]$TargetF1 = 0.77,
    [switch]$Resume,
    [switch]$QuickTest
)

$PYTHON = ".venv-codagt\Scripts\python.exe"
$SCRIPT = "ptst_dgm\scripts\seed_search_v0.3.5.py"

if (-not (Test-Path $PYTHON)) {
    Write-Host "[Error] Python not found: $PYTHON" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $SCRIPT)) {
    Write-Host "[Error] Script not found: $SCRIPT" -ForegroundColor Red
    exit 1
}

Write-Host "===== v0.3.5 Seed Search: Reproducing v0.3 Best Performance =====" -ForegroundColor Cyan
Write-Host ""

# Quick test mode: only test first 10 seeds
if ($QuickTest) {
    $Start = 0
    $End = 10
    Write-Host "[Quick Test Mode] Testing only seeds 0-9" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Search range   : seed $Start to $($End - 1) ($($End - $Start) seeds)"
Write-Host "  Epochs per seed: $Epochs"
Write-Host "  Target F1      : $TargetF1 (v0.3 best: 0.7726)"
Write-Host "  Resume mode    : $Resume"
Write-Host ""
Write-Host "Fixed parameters (v0.3 best):" -ForegroundColor Yellow
Write-Host "  Architecture   : patch_len=26, stride=16"
Write-Host "  Focal Loss     : α=0.866, γ=1.156, w_n=1.851, w_a=4.035"
Write-Host "  LoRA           : rank=16, alpha=32"
Write-Host ""

# Estimate time
$estimated_time_per_seed = 2.5  # minutes (conservative estimate)
$total_seeds = $End - $Start
$estimated_hours = ($total_seeds * $estimated_time_per_seed) / 60

Write-Host "Estimated time: $([math]::Round($estimated_hours, 1)) hours" -ForegroundColor Cyan
Write-Host "  (assuming ~$estimated_time_per_seed min per seed, $total_seeds seeds total)" -ForegroundColor DarkGray
Write-Host ""

# Confirm execution
if (-not $QuickTest -and $total_seeds -gt 100) {
    $confirm = Read-Host "This will take significant time. Continue? (y/n)"
    if ($confirm -ne "y") {
        Write-Host "Aborted by user." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host "Starting seed search..." -ForegroundColor Green
Write-Host ""

# Build command
$args = @(
    $SCRIPT,
    "--start", $Start,
    "--end", $End,
    "--epochs", $Epochs,
    "--target-f1", $TargetF1
)

if ($Resume) {
    $args += "--resume"
}

# Execute
& $PYTHON $args

$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "[Success] Seed search completed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Check results: ptst_dgm\results\seed_search_v0.3.5.jsonl"
    Write-Host "  2. Find best seed with: Get-Content ptst_dgm\results\seed_search_v0.3.5.jsonl | ConvertFrom-Json | Sort-Object macro_f1 -Descending | Select-Object -First 10"
    Write-Host "  3. Retrain with best seed for production use"
} else {
    Write-Host ""
    Write-Host "[Error] Seed search failed with exit code $exitCode" -ForegroundColor Red
}

exit $exitCode
