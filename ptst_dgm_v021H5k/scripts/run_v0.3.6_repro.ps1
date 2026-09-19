# v0.3.6 Reproduction: Exact reproduction with seed=42
# Goal: Verify if v0.3.6's F1=0.818 result is reproducible with seed=42
# Configuration: Same as v0.3.6 but with clean start

.\.venv-codagt\Scripts\Activate.ps1

Write-Host "=== v0.3.6 Reproduction with seed=42 ===" -ForegroundColor Cyan
Write-Host "Expected result: F1 ≈ 0.818 at iteration 2" -ForegroundColor Yellow
Write-Host "Architecture: patch=29, stride=16 (44.8% overlap)" -ForegroundColor Yellow

python -m ptst_dgm.multi_objective_agent.ptst_loop `
    --total-budget 500 `
    --archive ptst_dgm/results/ptst_archive_v0.3.6_repro.jsonl `
    --checkpoint-interval 60 `
    --seed 42

Write-Host "`nv0.3.6 reproduction complete!" -ForegroundColor Green
Write-Host "Check if Trial #1 achieves F1 ≈ 0.818 (reproducibility test)" -ForegroundColor Cyan
