# init_baseline_v0.3.4.ps1
# Initialize v0.3.4 DGM archive with v0.3 best solution as baseline.
#
# v0.3.4 Strategy:
#   - Start from v0.3's best solution (F1=0.7726, patch=26, stride=16)
#   - Fix Architecture at v0.3 best (patch_len=26, stride=16)
#   - Fix Focal Loss at v0.2 best (α=0.866, γ=1.156, w_n=1.851, w_a=4.035)
#   - Optimize LoRA parameters (rank, alpha) to improve beyond baseline
#
# Usage:
#   .\ptst_dgm\scripts\init_baseline_v0.3.4.ps1 -Archive "ptst_dgm\results\ptst_archive_v0.3.4.jsonl"

param(
    [string]$DataPath = "data\golden_testset",
    [string]$Archive  = "ptst_dgm\results\ptst_archive_v0.3.4.jsonl",
    [int]   $Epochs   = 100
)

$PYTHON = ".venv-ptstf\Scripts\python.exe"
$SCRIPT = "ptst_dgm\training\train_patchtst_dgm.py"

if (-not (Test-Path $PYTHON)) {
    Write-Host "[Error] Python not found: $PYTHON" -ForegroundColor Red
    exit 1
}

Write-Host "===== v0.3.4 Baseline Initialization =====" -ForegroundColor Cyan
Write-Host "  Strategy: Start from v0.3 best (F1=0.7726, patch=26, stride=16)" -ForegroundColor Yellow
Write-Host "  Control:  LoRA rank/alpha (2D)" -ForegroundColor Yellow
Write-Host "  Fixed:    Architecture (v0.3 best) + Focal Loss (v0.2 best)" -ForegroundColor Yellow
Write-Host ""

# v0.3 best solution parameters (trial #1):
# - Architecture: patch_len=26, stride=16
# - Focal Loss: α=0.866, γ=1.156, w_n=1.851, w_a=4.035
# - LoRA: rank=16, alpha=32 (baseline)
# - macro_F1: 0.7726

Write-Host "[Baseline] Training with v0.3 best parameters..." -ForegroundColor Green
Write-Host "  Architecture: patch_len=26, stride=16 (v0.3 best - FIXED)"
Write-Host "  Focal Loss: α=0.866, γ=1.156, w_n=1.851, w_a=4.035 (v0.2 best - FIXED)"
Write-Host "  LoRA: rank=16, alpha=32 (baseline - starting point for optimization)"
Write-Host "  Epochs: $Epochs"
Write-Host ""

& $PYTHON $SCRIPT `
    --data-path          $DataPath `
    --focal-alpha        0.866 `
    --focal-gamma        1.156 `
    --w-normal           1.851 `
    --w-anomal           4.035 `
    --patch-len          26 `
    --stride             16 `
    --lora-rank          16 `
    --lora-alpha         32 `
    --epochs             $Epochs `
    --output-json        $Archive

if ($LASTEXITCODE -ne 0) {
    Write-Host "[Error] Baseline training failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[Success] v0.3.4 baseline initialized → $Archive" -ForegroundColor Green
Write-Host ""
Write-Host "Next step:" -ForegroundColor Yellow
Write-Host "  .\ptst_dgm\scripts\run_ptst_dgm.ps1 -TotalBudget 100 -Archive ""$Archive"" -PopulationSize 20 -Epochs $Epochs"
Write-Host ""
Write-Host "Goal: Improve v0.3 best (F1=0.7726) by optimizing LoRA parameters (rank, alpha)" -ForegroundColor Cyan
