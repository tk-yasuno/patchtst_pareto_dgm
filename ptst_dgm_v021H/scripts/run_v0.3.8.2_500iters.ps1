# v0.3.8.2: Architecture optimization WITHOUT constraint (LLM prompt also fixed)
# Removed golden ratio guidance from LLM validator prompt
# Full 2D grid exploration: patch_len [20,32], stride [10,20]
# No overlap requirement, no constraint logic

.\.venv-codagt\Scripts\Activate.ps1

python -m ptst_dgm.multi_objective_agent.ptst_loop `
    --total-budget 500 `
    --archive ptst_dgm/results/ptst_archive_v0.3.8.2_500iters.jsonl `
    --checkpoint-interval 60

Write-Host "v0.3.8.2 (500 iters, no constraint, LLM prompt fixed) complete!"
