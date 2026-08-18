# Multi-Objective DGM: PatchTST Time-Series Anomaly Detector

**Version:** 0.1.0  
**License:** MIT  
**Date:** 2026-08-18

---

## Overview

This project applies the **Darwin-Gödel Machine (DGM)** self-improvement framework to automate hyperparameter tuning for a **PatchTST**-based time-series anomaly detector. The system optimizes four Focal Loss and class-weight parameters across **15 objectives** (AUC, Precision, Recall, F1, FPR × 3 forecast horizons) using **NSGA-II** Pareto multi-objective optimization with **Codestral 22B** LLM-based validation.

### Problem Statement

Pump facility time-series anomaly detection suffers from:
- **Severe class imbalance**: Fault events occur in <13% of time windows
- **Multi-horizon trade-offs**: 30d/60d/90d forecasts have different optimal parameters
- **4D hyperparameter space**: Focal Loss (α, γ) × class weights (w_Normal, w_Anomal) require expensive full fine-tuning runs to evaluate

### Solution

**DGM + NSGA-II Pareto Optimization:**
1. NSGA-II proposes 4-parameter configurations exploring the Pareto frontier
2. Codestral LLM validates proposals in anomaly detection context
3. PatchTST (LoRA r=16, α=32) fine-tunes for 100 epochs per iteration
4. 15 objectives guide the search toward "super-forecast" configurations

---

## Architecture

```
ptst_dgm/
├── agent/
│   ├── archive.py          # JSONL-backed archive (PatchTSTAgentEntry)
│   └── evaluator.py        # Subprocess evaluator → 15-metric JSON
├── multi_objective_agent/
│   ├── ptst_sampler.py     # NSGA-II (4 params × 15 objectives)
│   ├── pareto_archive.py   # Pareto frontier with mixed directions
│   ├── ptst_agent.py       # Codestral LLM validator
│   └── ptst_loop.py        # Main DGM loop + JSONL logging
├── training/
│   └── train_patchtst_dgm.py  # WeightedFocalLoss + 15-metric eval
└── scripts/
    ├── init_baseline.ps1       # Baseline training + archive init
    ├── init_baseline_archive.py
    └── run_ptst_dgm.ps1        # Main execution script
```

### Control Variables (4D Search Space)

| Parameter      | Range         | Description                          |
|----------------|---------------|--------------------------------------|
| `focal_alpha`  | [0.10, 0.90]  | Focal Loss class balancing factor    |
| `focal_gamma`  | [0.50, 5.00]  | Focal Loss focusing exponent         |
| `w_normal`     | [0.10, 5.00]  | Per-sample weight for Normal class   |
| `w_anomal`     | [0.50, 10.0]  | Per-sample weight for Anomaly class  |

### Objectives (15 Total)

| Metric         | Horizons        | Direction  |
|----------------|-----------------|------------|
| AUC            | 30d, 60d, 90d   | Maximize   |
| Precision      | 30d, 60d, 90d   | Maximize   |
| Recall         | 30d, 60d, 90d   | Maximize   |
| F1             | 30d, 60d, 90d   | Maximize   |
| FPR            | 30d, 60d, 90d   | Minimize   |

---

## Baseline Performance

**PatchTST v4.1.2** (LoRA r=16, α=32, Focal Loss α=0.5/γ=1.0, equal class weights):

| Horizon | AUC   | Precision | Recall | F1    | FPR   |
|---------|-------|-----------|--------|-------|-------|
| 30d     | 0.955 | 0.832     | 0.690  | 0.754 | 0.021 |
| 60d     | 0.965 | 0.923     | 0.730  | 0.815 | 0.011 |
| 90d     | 0.921 | 0.744     | 0.584  | 0.654 | 0.035 |

**Dataset:** Golden Testset (1000 samples, 70%/15%/15% train/val/test split, stratified on label_30d)

---

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 16GB VRAM (recommended)
- Ollama with `codestral:latest` model

### Virtual Environments

This project uses two separate virtual environments:

1. **`.venv-codagt`** (DGM loop, Codestral agent):
   ```powershell
   python -m venv .venv-codagt
   .\.venv-codagt\Scripts\Activate.ps1
   pip install optuna requests numpy pandas
   ```

2. **`.venv-ptstf`** (PatchTST training):
   ```powershell
   python -m venv .venv-ptstf
   .\.venv-ptstf\Scripts\Activate.ps1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers peft pandas scikit-learn pyarrow
   ```

---

## Usage

### Step 1: Initialize Baseline Archive

Train the baseline configuration and create the initial DGM archive:

```powershell
.\ptst_dgm\scripts\init_baseline.ps1
```

**Expected runtime:** ~20-30 minutes (100 epochs with early stopping)

**Output:**
- `ptst_dgm/results/baseline_eval.json` (15 metrics)
- `ptst_dgm/results/ptst_archive.jsonl` (1 baseline entry)

### Step 2: Dry-Run Verification

Test the DGM loop without actual training (uses random objectives):

```powershell
.\.venv-codagt\Scripts\python.exe -m ptst_dgm.multi_objective_agent.ptst_loop `
    --dry-run --total-budget 3
```

**Expected output:**
```
[PatchTSTDGM] Complete!  Pareto-optimal: 3/3 (100.0%)
[Pareto] Pareto frontier: 3 solutions  best_macroF1=...  best_maxFPR=...
```

### Step 3: Production DGM Loop

Run the full self-improvement loop:

```powershell
.\ptst_dgm\scripts\run_ptst_dgm.ps1 -TotalBudget 50
```

**Expected runtime:** ~17-25 hours (50 iterations × ~20-30 min/iteration)

**Output:**
- `ptst_dgm/results/ptst_archive.jsonl` (all evaluated configurations)
- `ptst_dgm/results/ptst_archive_log.jsonl` (iteration-by-iteration log)
- `ptst_dgm/results/ptst_archive_pareto.jsonl` (Pareto-optimal solutions)

### Command-Line Options

```powershell
.\ptst_dgm\scripts\run_ptst_dgm.ps1 `
    -TotalBudget 50 `
    -PopulationSize 20 `
    -Epochs 100 `
    -Model "codestral:latest" `
    -DryRun  # Optional: skip actual training
```

---

## Verification Results (v0.1.0)

### Module Import Test

```powershell
.\.venv-codagt\Scripts\python.exe -c "
from ptst_dgm.agent.archive import PatchTSTArchive
from ptst_dgm.multi_objective_agent.ptst_sampler import PatchTSTSampler
from ptst_dgm.multi_objective_agent.pareto_archive import ParetoArchive
print('All modules OK')
"
```

**Status:** ✅ Pass

### NSGA-II Sampler Test

```python
from ptst_dgm.multi_objective_agent.ptst_sampler import PatchTSTSampler
sampler = PatchTSTSampler()
trial_num, params = sampler.suggest()
print(f"trial={trial_num} params={params}")
```

**Output:**
```
trial=0 params={'focal_alpha': 0.400, 'focal_gamma': 4.778, 
                'w_normal': 3.687, 'w_anomal': 6.187}
```

**Status:** ✅ Pass

### DGM Loop Dry-Run (2 iterations)

```
[PatchTSTDGM] Complete!  Pareto-optimal: 2/2 (100.0%)
[Pareto] Pareto frontier: 2 solutions  best_macroF1=0.7608  best_maxFPR=0.0553
```

**Status:** ✅ Pass

### PatchTST Model Test

```python
from src.models.patch_tst_lora import build_patch_tst, PatchTSTWrapper
from src.models.lora_config import LoRAParams
import torch

lora = LoRAParams(r=16, lora_alpha=32, lora_dropout=0.05, bias='none')
base = build_patch_tst(lora_params=lora)
model = PatchTSTWrapper(base, dropout=0.1)
dummy = torch.randn(2, 90)
out = model(dummy)
print(f"Output shape: {out.shape}")  # torch.Size([2, 3])
```

**Trainable parameters:** 49,152 / 656,131 (7.49%)

**Status:** ✅ Pass

---

## Implementation Details

### WeightedFocalLoss

Combines Focal Loss with per-sample class weighting:

```python
class WeightedFocalLoss(nn.Module):
    def forward(self, logits, targets):
        focal_loss = FocalLoss(alpha, gamma, reduction="none")(logits, targets)
        class_weights = w_anomal * targets + w_normal * (1 - targets)
        return (class_weights * focal_loss).mean()
```

### Threshold Selection

For each horizon, we find the threshold on the validation set that maximizes F1:

```python
def find_best_threshold(y_true, y_score):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.91, 0.01):
        pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t
```

This threshold is then applied to the test set to compute final metrics.

### Pareto Dominance with Mixed Directions

Objectives 0-11 (AUC/Precision/Recall/F1 × 3) are maximized.  
Objectives 12-14 (FPR × 3) are minimized.

Internally, FPR values are negated so all comparisons use "higher is better":

```python
def _to_internal(objectives: Dict[str, float]) -> List[float]:
    vals = [objectives[k] for k in OBJECTIVE_KEYS]
    for i in [12, 13, 14]:  # FPR indices
        vals[i] = -vals[i]
    return vals
```

---

## Roadmap

### v0.1.0 (Current)
- [x] DGM architecture implementation
- [x] NSGA-II multi-objective sampler
- [x] Codestral LLM validator
- [x] WeightedFocalLoss training pipeline
- [x] 15-objective evaluation
- [x] Pareto archive with mixed directions
- [x] Dry-run verification

### v0.2.0 (Planned)
- [ ] Run baseline comparison (50 iterations)
- [ ] Pareto frontier analysis
- [ ] Super-forecast achievement validation
- [ ] Hyperparameter sensitivity plots
- [ ] Paper results section completion

### v0.3.0 (Future)
- [ ] Multi-GPU parallel evaluation
- [ ] Resume from checkpoint
- [ ] Interactive Pareto front visualization
- [ ] Transfer learning to new facilities
- [ ] Online learning mode

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yasuno2026patchtst_dgm,
  title   = {Multi-Objective {DGM}: Finetuning Anomaly Detector
             with Focal Loss and Class Weights
             Using {PatchTST} Time Series Transformer},
  author  = {Yasuno, Takato},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## References

- **Darwin-Gödel Machine**: Liu et al. (2025) [arXiv:2505.22954](https://arxiv.org/abs/2505.22954)
- **NSGA-II**: Deb et al. (2002) "A fast and elitist multiobjective genetic algorithm"
- **PatchTST**: Nie et al. (2023) "A Time Series is Worth 64 Words" [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
- **Focal Loss**: Lin et al. (2017) "Focal Loss for Dense Object Detection" (ICCV 2017)
- **LoRA**: Hu et al. (2022) "LoRA: Low-Rank Adaptation of Large Language Models" [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

---

## License

```
MIT License

Copyright (c) 2026 Takato Yasuno

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact

For questions or collaboration inquiries, please open an issue on the repository.

**Maintainer:** Takato Yasuno  
**Project Homepage:** [GitHub Repository URL TBD]
