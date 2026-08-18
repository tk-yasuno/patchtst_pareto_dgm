# run_ptst_dgm.ps1
# Multi-Objective Pareto DGM for PatchTST anomaly detector.
#
# Usage:
#   .\ptst_dgm\scripts\run_ptst_dgm.ps1 -TotalBudget 50
#   .\ptst_dgm\scripts\run_ptst_dgm.ps1 -TotalBudget 50 -DryRun

param(
    [string]$Model          = "codestral:latest",
    [int]   $TotalBudget    = 50,
    [int]   $PopulationSize = 20,
    [int]   $Epochs         = 100,
    [string]$DataPath       = "data\golden_testset",
    [string]$Archive        = "ptst_dgm\results\ptst_archive.jsonl",
    [string]$OutputDir      = "ptst_dgm\results\temp_model",
    [switch]$DryRun
)

$env:OLLAMA_NUM_GPU       = "1"
$env:CUDA_VISIBLE_DEVICES = "0"

$DGM_PYTHON   = ".venv-codagt\Scripts\python.exe"
$TRAIN_PYTHON = ".venv-ptstf\Scripts\python.exe"
$LOOP_SCRIPT  = "ptst_dgm\multi_objective_agent\ptst_loop.py"

Write-Host "===== Multi-Objective Pareto DGM — PatchTST Anomaly Detector =====" -ForegroundColor Cyan
Write-Host ""

foreach ($exe in @($DGM_PYTHON, $TRAIN_PYTHON)) {
    if (-not (Test-Path $exe)) {
        Write-Host "[Error] Virtual environment not found: $exe" -ForegroundColor Red
        exit 1
    }
}

# ── Archive bootstrap ─────────────────────────────────────────────────────────
if (-not (Test-Path $Archive)) {
    Write-Host "[Init] Archive not found — running baseline init first..." -ForegroundColor Yellow
    & .\ptst_dgm\scripts\init_baseline.ps1 -DataPath $DataPath -Archive $Archive -Epochs $Epochs
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Model          : $Model"
Write-Host "  Optimizer      : NSGA-II (Pareto, 15 objectives)"
Write-Host "  Control vars   : focal_alpha, focal_gamma, w_normal, w_anomal"
Write-Host "  Objectives     : AUC/Precision/Recall/F1 (maximize) × FPR (minimize) × 3 horizons"
Write-Host "  Population     : $PopulationSize"
Write-Host "  Budget         : $TotalBudget iterations"
Write-Host "  Epochs/iter    : $Epochs"
Write-Host "  Archive        : $Archive"
if ($DryRun) { Write-Host "  Mode           : DRY RUN" -ForegroundColor Magenta }
Write-Host ""

$argList = @(
    "-m", "ptst_dgm.multi_objective_agent.ptst_loop",
    "--model",           $Model,
    "--archive",         $Archive,
    "--total-budget",    $TotalBudget,
    "--population-size", $PopulationSize,
    "--python-exe",      $TRAIN_PYTHON,
    "--script",          "ptst_dgm\training\train_patchtst_dgm.py",
    "--data-path",       $DataPath,
    "--output-dir",      $OutputDir,
    "--epochs",          $Epochs
)
if ($DryRun) { $argList += "--dry-run" }

& $DGM_PYTHON @argList

$exit_code = $LASTEXITCODE
Write-Host ""
if ($exit_code -eq 0) {
    Write-Host ("=" * 70) -ForegroundColor Green
    Write-Host "PatchTST DGM Complete" -ForegroundColor Green
    $archiveItem = Get-Item $Archive -ErrorAction SilentlyContinue
    $paretoPath  = $Archive -replace '\.jsonl$', '_pareto.jsonl'
    if ($archiveItem) { Write-Host "Archive : $($archiveItem.FullName)" }
    if (Test-Path $paretoPath) { Write-Host "Pareto  : $(Resolve-Path $paretoPath)" }
    Write-Host ("=" * 70) -ForegroundColor Green
} else {
    Write-Host "Execution failed (exit code $exit_code)" -ForegroundColor Red
}

exit $exit_code
