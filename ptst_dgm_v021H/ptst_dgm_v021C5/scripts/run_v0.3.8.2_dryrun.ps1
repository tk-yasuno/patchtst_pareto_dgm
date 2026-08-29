# v0.3.8.2 Dry Run: 3 iterations to verify constraint-free operation
# Test: LLM prompt with golden ratio guidance removed
# Expected: stride >= patch_len configurations should be accepted

.\.venv-codagt\Scripts\Activate.ps1

Write-Host "=== v0.3.8.2 Dry Run (3 iterations, constraint-free) ===" -ForegroundColor Cyan

python -m ptst_dgm.multi_objective_agent.ptst_loop `
    --total-budget 3 `
    --archive ptst_dgm/results/ptst_archive_v0.3.8.2_dryrun.jsonl `
    --checkpoint-interval 60 `
    --dry-run

Write-Host "`n=== Dry Run Complete ===" -ForegroundColor Green
Write-Host "Checking for constraint violations (stride >= patch_len)..." -ForegroundColor Cyan

$results = Get-Content "ptst_dgm\results\ptst_archive_v0.3.8.2_dryrun.jsonl" | ForEach-Object { $_ | ConvertFrom-Json }
$violations = $results | Where-Object { $_.stride -ge $_.patch_len }

if ($violations.Count -gt 0) {
    Write-Host "SUCCESS: Found $($violations.Count) configurations with stride >= patch_len" -ForegroundColor Green
    $violations | ForEach-Object {
        Write-Host "  patch_len=$($_.patch_len), stride=$($_.stride) (overlap=$([math]::Round(1-$_.stride/$_.patch_len, 3)))" -ForegroundColor Yellow
    }
} else {
    Write-Host "WARNING: No configurations with stride >= patch_len found" -ForegroundColor Yellow
    Write-Host "All configurations:" -ForegroundColor Cyan
    $results | ForEach-Object {
        Write-Host "  patch_len=$($_.patch_len), stride=$($_.stride) (overlap=$([math]::Round(1-$_.stride/$_.patch_len, 3)))"
    }
}
