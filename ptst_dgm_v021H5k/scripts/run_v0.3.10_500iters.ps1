# v0.3.10: DGM with FIXED seed (revert v0.3.9 random seed strategy)
# Issue diagnosed: Random seeds prevented NSGA-II from learning effectively
#   - v0.3.9: F1=0.5779 @ 224 iters (worse than initial F1=0.6898)
#   - Root cause: Different seeds → unstable evaluation → no convergence
#
# Configuration:
#   - Seeds: FIXED at 42 (reproducibility restored)
#   - Architecture: patch_len [20,32], stride [10,20]
#   - Focal Loss: FIXED at v0.2 best (α=0.866, γ=1.156, w_n=1.851, w_a=4.035)
#   - LoRA: FIXED at baseline (rank=16, alpha=32)
#   - Initial archive: F1=0.6898 from v0.3.9 (best baseline)
#   - Checkpoint: Every 60 iters
#   - Budget: 500 iterations
#
# Expected outcome:
#   - Stable NSGA-II optimization with reproducible results
#   - Improvement from F1=0.6898 baseline
#   - Similar success to v0.3.6 (F1=0.773 achieved)

.\.venv-codagt\Scripts\Activate.ps1

Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "v0.3.10: DGM with FIXED Seed - Revert Random Seed Strategy" -ForegroundColor Cyan
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Issue diagnosed in v0.3.9:" -ForegroundColor Red
Write-Host "  Random seeds → unstable evaluation → NSGA-II cannot learn" -ForegroundColor White
Write-Host "  Result: F1=0.5779 @ 224 iters (WORSE than initial F1=0.6898)" -ForegroundColor White
Write-Host ""
Write-Host "v0.3.10 Fix:" -ForegroundColor Yellow
Write-Host "  Seeds:        FIXED at 42 (reproducibility restored)" -ForegroundColor Green
Write-Host "  Architecture: patch_len [20,32], stride [10,20]" -ForegroundColor Green
Write-Host "  Focal Loss:   FIXED α=0.866 γ=1.156 (v0.2 best)" -ForegroundColor Green
Write-Host "  LoRA:         FIXED rank=16 alpha=32 (baseline)" -ForegroundColor Green
Write-Host "  Initial:      F1=0.6898 (v0.3.9 best baseline)" -ForegroundColor Green
Write-Host "  Checkpoint:   Every 60 iters → models/v0.3.10_500iters_checkpoints/" -ForegroundColor Green
Write-Host "  Budget:       500 iterations" -ForegroundColor Green
Write-Host ""
Write-Host "Expected: Stable optimization, improvement from F1=0.6898" -ForegroundColor Magenta
Write-Host ""
Write-Host "Press Ctrl+C to stop (checkpoints will be saved)" -ForegroundColor Red
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host ""

python -m ptst_dgm.multi_objective_agent.ptst_loop `
    --total-budget 500 `
    --archive ptst_dgm/results/ptst_archive_v0.3.10_500iters.jsonl `
    --checkpoint-interval 60

Write-Host ""
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host "v0.3.10 Complete!" -ForegroundColor Green
Write-Host "=================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Check results:" -ForegroundColor Yellow
Write-Host "  1. Log:        ptst_dgm/results/ptst_archive_v0.3.10_500iters_log.jsonl"
Write-Host "  2. Pareto:     ptst_dgm/results/ptst_archive_v0.3.10_500iters_pareto.jsonl"
Write-Host "  3. Checkpoints: models/v0.3.10_500iters_checkpoints/"
Write-Host ""
Write-Host "Seed FIXED at 42 - reproducible results" -ForegroundColor Cyan
Write-Host "Initial archive: F1=0.6898 (v0.3.9 best)" -ForegroundColor Magenta
