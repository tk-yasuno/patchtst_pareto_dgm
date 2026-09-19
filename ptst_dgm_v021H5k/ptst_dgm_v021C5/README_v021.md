# PatchTST DGM v0.2.1 - Focal Loss 4D Optimization

## Experiment Overview

v0.2.1は、v0.2のFocal Loss 4D探索に戻り、v0.3.6/v0.3.6.3で得られた最適なArchitectureとLoRAパラメータを固定して実行する実験です。

### Control Variables (4D)
| Parameter | Range | Description | v0.2 Phase 3 best |
|-----------|-------|-------------|-------------------|
| `focal_alpha` | [0.5, 0.9] | Focal Loss alpha (focus on hard examples) | 0.866 |
| `focal_gamma` | [0.6, 2.0] | Focal Loss gamma (modulation factor) | 1.156 |
| `w_normal` | [1.0, 3.0] | Class weight for normal samples | 1.851 |
| `w_anomal` | [2.0, 6.0] | Class weight for anomalous samples | 4.035 |

**Search Strategy**: Refined search near v0.2 Phase 3 best (±40-50% range)

### Fixed Parameters
| Parameter | Value | Source |
|-----------|-------|--------|
| `patch_len` | 29 | v0.3.6 best |
| `stride` | 16 | v0.3.6 best |
| `lora_rank` | 19 | v0.3.6.3 best (Iter 269) |
| `lora_alpha` | 53 | v0.3.6.3 best (Iter 269) |
| `seed` | None (random) | For diversity |

### Baseline (v0.2 Phase 3 best)
- **Focal Loss**: α=0.866, γ=1.156, w_normal=1.851, w_anomal=4.035
- **Architecture**: patch_len=29, stride=16 (v0.3.6 best)
- **LoRA**: rank=19, alpha=53 (v0.3.6.3 best)

### Experiment Configuration
- **Budget**: 1000 iterations
- **Checkpoint**: Every 100 iterations (save adapter model)
- **Population**: 20 (NSGA-II)
- **Epochs**: 100 per trial
- **Objectives**: 15 (5 metrics × 3 horizons: 30d, 60d, 90d)
  - Maximize: AUC, Precision, Recall, F1
  - Minimize: FPR

## Usage

### 1. Initialize Baseline
```powershell
.\ptst_dgm_v021\scripts\init_baseline.ps1
```
- Trains PatchTST baseline with v0.2 Phase 3 best Focal Loss + v0.3.6/v0.3.6.3 Architecture/LoRA
- Creates archive: `ptst_dgm_v021\results\ptst_archive_v021.jsonl`

### 2. Run DGM Loop
```powershell
.\ptst_dgm_v021\scripts\run_v021.ps1
```
- Executes 1000 iterations of Multi-Objective Pareto optimization
- Saves checkpoint every 100 iterations

## Expected Outcomes
1. **Optimal Focal Loss parameters** for v0.3.6/v0.3.6.3 Architecture/LoRA configuration
2. **Pareto frontier** across 15 objectives
3. **Checkpoint models** at Iter 100, 200, ..., 1000

## Rationale
- **v0.3.6.2**: LoRA optimization showed 27% performance degradation (F1=0.5949 vs baseline 0.8183)
- **v0.3.6.3**: Refined LoRA search with random seeds achieved F1=0.6188 (r=19, α=53)
- **v0.2.1**: Return to Focal Loss optimization using optimal Architecture/LoRA from v0.3.6.x
- **Hypothesis**: Loss function optimization may be more impactful than LoRA fine-tuning for this task

## Directory Structure
```
ptst_dgm_v021/
├── agent/
│   ├── archive.py              # Archive management
│   └── evaluator.py            # PatchTST training subprocess evaluator
├── multi_objective_agent/
│   ├── ptst_agent.py           # LLM validator for NSGA-II proposals
│   ├── ptst_loop.py            # Main DGM loop with checkpoint mechanism
│   ├── ptst_sampler.py         # 4D Focal Loss sampler
│   └── pareto_archive.py       # Pareto frontier tracking
├── scripts/
│   ├── init_baseline.ps1       # Baseline initialization
│   ├── init_baseline_archive.py # Archive creation from baseline
│   └── run_v021.ps1            # Main execution script
├── training/
│   └── train_patchtst_dgm.py   # PatchTST training script
└── results/
    ├── ptst_archive_v021.jsonl # Main archive
    ├── baseline_eval.json      # Baseline metrics
    └── temp_model/             # Checkpoint models
```

## Implementation Notes
- **Import paths**: All `ptst_dgm` imports changed to `ptst_dgm_v021`
- **Python cache**: Cleared before execution
- **NSGA-II**: Population size 20, multi-objective optimization
- **LLM validator**: codestral:latest at localhost:11434
- **Seed strategy**: Random (not fixed) for diversity in DGM exploration
