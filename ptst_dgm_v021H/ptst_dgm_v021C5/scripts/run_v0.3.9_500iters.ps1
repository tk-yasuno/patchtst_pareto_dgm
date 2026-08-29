# v0.3.9: DGM with RANDOM seeds for enhanced exploration
# Philosophy: DGM evolution is non-reproducible by nature
#   - LLM responses vary even with same prompt
#   - Multiple random seeds increase diversity of search space
#   - Checkpoint saves best models during evolution (not after)
#
# Configuration:
#   - Seeds: RANDOM [0, 9999] (no fixed seed)
#   - Architecture: patch_len [20,32], stride [10,20] (no constraint)
#   - Focal Loss: FIXED at v0.2 best (α=0.866, γ=1.156, w_n=1.851, w_a=4.035)
#   - LoRA: FIXED at baseline (rank=16, alpha=32)
#   - Checkpoint: Every 60 iters (save best in each interval)
#   - Budget: 500 iterations
#
# Expected outcome:
#   - More diverse exploration than seed=42 fixed version
#   - Better chance of finding high-performance regions
#   - Checkpoints preserve best models even if unreproducible

.\.venv-codagt\Scripts\Activate.ps1

Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "v0.3.9: DGM with RANDOM Seeds - Enhanced Exploration Strategy" -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Philosophy:" -ForegroundColor Yellow
Write-Host "  DGM evolution is inherently NON-REPRODUCIBLE" -ForegroundColor White
Write-Host "  → LLM varies, environment changes, random states differ" -ForegroundColor White
Write-Host "  → Fixed seed=42 LIMITS exploration to single trajectory" -ForegroundColor White
Write-Host "  → Random seeds EXPAND search space through diversity" -ForegroundColor White
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Seeds:        RANDOM [0-9999] (no reproducibility, max diversity)" -ForegroundColor Green
Write-Host "  Architecture: patch_len [20,32], stride [10,20] (no constraint)" -ForegroundColor Green
Write-Host "  Focal Loss:   FIXED α=0.866 γ=1.156 (v0.2 best)" -ForegroundColor Green
Write-Host "  LoRA:         FIXED rank=16 alpha=32 (baseline)" -ForegroundColor Green
Write-Host "  Checkpoint:   Every 60 iters → models/v0.3.9_500iters_checkpoints/" -ForegroundColor Green
Write-Host "  Budget:       500 iterations" -ForegroundColor Green
Write-Host ""
Write-Host "Initial parent: F1=0.6898 (from v0.3.6 success baseline)" -ForegroundColor Magenta
Write-Host ""
Write-Host "Press Ctrl+C to stop (checkpoints will be saved)" -ForegroundColor Red
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host ""

python -m ptst_dgm.multi_objective_agent.ptst_loop `
    --total-budget 500 `
    --archive ptst_dgm/results/ptst_archive_v0.3.9_500iters.jsonl `
    --checkpoint-interval 60

Write-Host ""
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "v0.3.9 Complete!" -ForegroundColor Green
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Check results:" -ForegroundColor Yellow
Write-Host "  1. Log:        ptst_dgm/results/ptst_archive_v0.3.9_500iters_log.jsonl"
Write-Host "  2. Pareto:     ptst_dgm/results/ptst_archive_v0.3.9_500iters_pareto.jsonl"
Write-Host "  3. Checkpoints: models/v0.3.9_500iters_checkpoints/"
Write-Host ""
Write-Host "Best models are saved in checkpoints (every 60 iters)" -ForegroundColor Cyan
Write-Host "No reproducibility - each run explores different trajectories" -ForegroundColor Magenta
