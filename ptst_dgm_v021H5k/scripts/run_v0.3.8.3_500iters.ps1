# v0.3.8.3: Architecture optimization with PROPER initial archive
# Fixed: Use v0.3.6's good initial parent (F1=0.6898) instead of poor baseline (F1=0.48)
# Configuration: seed=42, Focal Loss from v0.2, no constraint, full 2D grid

.\.venv-codagt\Scripts\Activate.ps1

Write-Host "=== Preparing v0.3.8.3 with v0.3.6 initial archive ===" -ForegroundColor Cyan

# Copy v0.3.6's first parent as initial archive for v0.3.8.3
$v036_first = Get-Content "ptst_dgm\results\ptst_archive_v0.3.6.jsonl" | Select-Object -First 1
$v036_first | Set-Content "ptst_dgm\results\ptst_archive_v0.3.8.3_init.jsonl"

Write-Host "Initial archive created from v0.3.6 parent (F1=0.6898)" -ForegroundColor Green
Write-Host "This matches v0.3.6's successful configuration" -ForegroundColor Yellow

python -m ptst_dgm.multi_objective_agent.ptst_loop `
    --total-budget 500 `
    --archive ptst_dgm/results/ptst_archive_v0.3.8.3_500iters.jsonl `
    --parent-archive ptst_dgm/results/ptst_archive_v0.3.8.3_init.jsonl `
    --checkpoint-interval 60

Write-Host "`nv0.3.8.3 (500 iters, proper initialization) complete!" -ForegroundColor Green
Write-Host "Expected: Should reproduce v0.3.6 success (F1 > 0.7)" -ForegroundColor Cyan
