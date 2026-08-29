# run_ensemble_v0.5.ps1
# Execute Pareto Top-3 Ensemble Evaluation (v0.5)

param(
    [string]$ParetoJsonl = "ptst_dgm\results\ptst_archive_v0.3_pareto.jsonl",
    [string]$DataPath = "data\golden_testset",
    [int]$TopK = 3,
    [int]$Epochs = 100,
    [string]$Output = "ptst_dgm\results\ensemble_v0.5_result.json"
)

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " PatchTST v0.5: Pareto Top-3 Ensemble Evaluation" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Pareto Frontier: $ParetoJsonl"
Write-Host "  Data Path:       $DataPath"
Write-Host "  Top-K:           $TopK"
Write-Host "  Epochs/Model:    $Epochs"
Write-Host "  Output:          $Output"
Write-Host ""

# Activate .venv-codagt environment
Write-Host "[1/3] Activating .venv-codagt environment..." -ForegroundColor Green
& .\.venv-codagt\Scripts\Activate.ps1

# Run ensemble evaluation
Write-Host "[2/3] Running ensemble evaluation..." -ForegroundColor Green
python ptst_dgm\scripts\evaluate_ensemble.py `
    --pareto-jsonl $ParetoJsonl `
    --data-path $DataPath `
    --top-k $TopK `
    --epochs $Epochs `
    --output $Output

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Ensemble evaluation failed!" -ForegroundColor Red
    exit 1
}

# Create markdown report
Write-Host "[3/3] Creating markdown report..." -ForegroundColor Green
$resultJson = Get-Content $Output | ConvertFrom-Json

$report = @"
# PatchTST v0.5: Pareto Top-3 Ensemble Results

**Date**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
**Strategy**: Soft voting ensemble of v0.3's top-3 Pareto solutions
**Total Models**: $TopK
**Epochs per Model**: $Epochs

---

## Executive Summary

### Individual Model Performance (v0.3 Pareto Frontier)

| Rank | Trial | Macro F1 | Mean FPR | patch_len | stride |
|------|-------|----------|----------|-----------|--------|
$(for ($i = 0; $i -lt $TopK; $i++) {
    $sol = $resultJson.pareto_solutions[$i]
    $trialNum = $sol.trial_number
    $f1 = [math]::Round($sol.macro_f1, 4)
    $fpr = [math]::Round($sol.mean_fpr, 4)
    $patch = $sol.params.patch_len
    $stride = $sol.params.stride
    "| $($i+1) | #$trialNum | $f1 | $fpr | $patch | $stride |"
})

### Ensemble Performance (v0.5)

**Macro F1**: $([math]::Round($resultJson.ensemble_metrics.macro_f1, 4))
**Mean FPR**: $([math]::Round($resultJson.ensemble_metrics.mean_fpr, 4))

**Per-Horizon Metrics**:
- **30d**: F1=$([math]::Round($resultJson.ensemble_metrics.f1_30d, 4)), FPR=$([math]::Round($resultJson.ensemble_metrics.fpr_30d, 4))
- **60d**: F1=$([math]::Round($resultJson.ensemble_metrics.f1_60d, 4)), FPR=$([math]::Round($resultJson.ensemble_metrics.fpr_60d, 4))
- **90d**: F1=$([math]::Round($resultJson.ensemble_metrics.f1_90d, 4)), FPR=$([math]::Round($resultJson.ensemble_metrics.fpr_90d, 4))

### Comparison

| Metric | v0.3 Best (Single) | v0.5 Ensemble | Improvement |
|--------|-------------------|---------------|-------------|
| **Macro F1** | $([math]::Round($resultJson.v03_best_f1, 4)) | $([math]::Round($resultJson.ensemble_metrics.macro_f1, 4)) | $([math]::Round($resultJson.improvement_pct, 2))% |

---

## Analysis

$(if ($resultJson.improvement_pct -gt 0) {
    "✅ **Ensemble improved performance by $([math]::Round($resultJson.improvement_pct, 2))%**"
    ""
    "**Key Insight**: Soft voting successfully leveraged diversity among Pareto solutions, demonstrating that ensemble methods can extract additional value from multi-objective optimization frontiers."
} else {
    "⚠️ **Ensemble did not improve over single best model**"
    ""
    "**Possible Reasons**:"
    "- Models too similar (low diversity)"
    "- Best model already near optimal"
    "- Overfitting in ensemble members"
    ""
    "**Recommendation**: Consider hard voting, weighted averaging, or stacking ensemble methods."
})

---

## Conclusion

v0.5's Pareto top-3 ensemble strategy $(if ($resultJson.improvement_pct -gt 0) { "successfully demonstrated" } else { "explored" }) the potential of leveraging multi-objective optimization diversity. $(if ($resultJson.improvement_pct -gt 5) { "The significant improvement validates ensemble methods as a promising direction for future work." } elseif ($resultJson.improvement_pct -gt 0) { "While improvement is modest, the approach shows potential with further refinement." } else { "Future work should explore alternative ensemble strategies or focus on improving individual model quality." })

---

## Files

- **Result JSON**: ``$Output``
- **Source Pareto Frontier**: ``$ParetoJsonl``
- **Model Checkpoints**: ``ptst_dgm/results/ensemble_models/``
"@

$reportPath = "ptst_dgm\results\RESULT_v0.5_ensemble.md"
$report | Out-File -FilePath $reportPath -Encoding utf8

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Green
Write-Host " Ensemble evaluation complete!" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results:" -ForegroundColor Yellow
Write-Host "  JSON:     $Output"
Write-Host "  Markdown: $reportPath"
Write-Host ""
Write-Host "Summary:" -ForegroundColor Yellow
Write-Host "  v0.3 Best:     F1=$([math]::Round($resultJson.v03_best_f1, 4))"
Write-Host "  v0.5 Ensemble: F1=$([math]::Round($resultJson.ensemble_metrics.macro_f1, 4))"
Write-Host "  Improvement:   $([math]::Round($resultJson.improvement_pct, 2))%"
