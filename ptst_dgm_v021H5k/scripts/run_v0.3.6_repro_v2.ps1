# v0.3.6 Reproduction v2: seed=42 fixed, no constraint, with v0.3.6 initial parent
# Goal: Reproduce v0.3.6 success (F1=0.818) with proper initialization
# Configuration: v0.3.8.2 code (no constraint) + v0.3.6 initial parent (F1=0.6898)

.\.venv-codagt\Scripts\Activate.ps1

Write-Host "=== v0.3.6 Reproduction v2 ===" -ForegroundColor Cyan
Write-Host "Initial parent: F1=0.6898 (v0.3.6 original)" -ForegroundColor Green
Write-Host "Code: v0.3.8.2 (no constraint, seed=42 fixed)" -ForegroundColor Yellow
Write-Host "Expected: F1 > 0.7 (hopefully approaching 0.8)" -ForegroundColor Magenta
Write-Host ""

python -m ptst_dgm.multi_objective_agent.ptst_loop `
    --total-budget 500 `
    --archive ptst_dgm/results/ptst_archive_v0.3.6_repro_v2.jsonl `
    --checkpoint-interval 60

Write-Host "`nv0.3.6 Reproduction v2 complete!" -ForegroundColor Green
Write-Host "Check if F1 improved from 0.6898" -ForegroundColor Cyan
