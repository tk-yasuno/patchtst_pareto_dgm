# v0.3.7 Dry-Run Test (13 iterations)
# Test checkpoint mechanism without real training

Write-Host "v0.3.7 Dry-Run Test" -ForegroundColor Cyan
Write-Host "Testing checkpoint mechanism with 13 iterations..." -ForegroundColor Yellow
Write-Host ""

& .\.venv-codagt\Scripts\Activate.ps1

python -m ptst_dgm.multi_objective_agent.ptst_loop `
    --model codestral:latest `
    --archive ptst_dgm/results/ptst_archive_v0.3.7_dryrun.jsonl `
    --total-budget 13 `
    --population-size 20 `
    --dry-run `
    --python-exe .venv-ptstf\Scripts\python.exe `
    --script ptst_dgm\training\train_patchtst_dgm.py `
    --data-path data\golden_testset `
    --output-dir ptst_dgm\results\temp_model `
    --epochs 100

Write-Host ""
Write-Host "Dry-run complete!" -ForegroundColor Green
Write-Host "Note: Checkpoints are NOT saved in dry-run mode" -ForegroundColor Yellow
