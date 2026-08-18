# init_baseline.ps1
# Train PatchTST baseline (focal_alpha=0.5, gamma=1.0, w=1.0) and
# initialise the DGM archive with its results.
#
# Run once before starting the DGM loop:
#   .\ptst_dgm\scripts\init_baseline.ps1

param(
    [string]$DataPath  = "data\golden_testset",
    [string]$Archive   = "ptst_dgm\results\ptst_archive.jsonl",
    [int]   $Epochs    = 100
)

$TRAIN_PYTHON = ".venv-ptstf\Scripts\python.exe"
$DGM_PYTHON   = ".venv-codagt\Scripts\python.exe"
$TRAIN_SCRIPT = "ptst_dgm\training\train_patchtst_dgm.py"
$INIT_SCRIPT  = "ptst_dgm\scripts\init_baseline_archive.py"
$EVAL_JSON    = "ptst_dgm\results\baseline_eval.json"
$OUTPUT_DIR   = "ptst_dgm\results\baseline_model"

Write-Host "===== PatchTST DGM — Baseline Initialisation =====" -ForegroundColor Cyan
Write-Host ""

foreach ($exe in @($TRAIN_PYTHON, $DGM_PYTHON)) {
    if (-not (Test-Path $exe)) {
        Write-Host "[Error] Not found: $exe" -ForegroundColor Red
        exit 1
    }
}

# ── Step 1: Train baseline ────────────────────────────────────────────────────
Write-Host "[Step 1] Training PatchTST baseline (alpha=0.5, gamma=1.0, w=1.0)..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path (Split-Path $EVAL_JSON) | Out-Null

& $TRAIN_PYTHON $TRAIN_SCRIPT `
    --focal-alpha 0.5 `
    --focal-gamma 1.0 `
    --w-normal 1.0 `
    --w-anomal 1.0 `
    --epochs $Epochs `
    --data-path $DataPath `
    --output-dir $OUTPUT_DIR `
    --output-json $EVAL_JSON

if ($LASTEXITCODE -ne 0) {
    Write-Host "[Error] Baseline training failed" -ForegroundColor Red; exit 1
}
Write-Host "[Step 1] Done — metrics saved to $EVAL_JSON" -ForegroundColor Green

# ── Step 2: Initialise archive ────────────────────────────────────────────────
Write-Host ""
Write-Host "[Step 2] Initialising DGM archive..." -ForegroundColor Yellow

& $DGM_PYTHON $INIT_SCRIPT `
    --eval-json $EVAL_JSON `
    --archive $Archive

if ($LASTEXITCODE -ne 0) {
    Write-Host "[Error] Archive initialisation failed" -ForegroundColor Red; exit 1
}

Write-Host ""
Write-Host "Baseline initialisation complete." -ForegroundColor Green
Write-Host "Next: run '.\ptst_dgm\scripts\run_ptst_dgm.ps1' to start DGM loop." -ForegroundColor Cyan
