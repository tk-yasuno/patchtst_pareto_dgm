"""
ptst_sampler.py
NSGA-II multi-objective sampler for PatchTST DGM.

2 control variables (v0.3 - architecture optimization):
  patch_len    [8, 32]   ← Patch length for time series segmentation
  stride       [4, 24]   ← Stride between patches (overlap control)

Fixed (v0.2 best): focal_alpha=0.866, gamma=1.156, w_normal=1.851, w_anomal=4.035
Fixed: LoRA r=16, alpha=32

15 objectives (5 metrics × 3 horizons):
  Maximize: auc×3, precision×3, recall×3, f1×3
  Minimize: fpr×3
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 15 objectives in canonical order — must match archive.OBJECTIVE_KEYS
_OBJ_KEYS = (
    [f"{m}_{h}d" for m in ["auc", "precision", "recall", "f1"] for h in [30, 60, 90]]
    + [f"fpr_{h}d" for h in [30, 60, 90]]
)
_DIRECTIONS = ["maximize"] * 12 + ["minimize"] * 3

# v0.3: Architecture parameters as control variables
# patch_len: Smaller → more granular, larger → more context per patch
# stride: Smaller → more overlap, larger → less redundancy
PARAM_BOUNDS = {
    "patch_len": (8, 32),   # v0.2: Fixed at 30 (v4-1-3_tst default)
    "stride":    (4, 24),   # v0.2: Fixed at 30
}


class PatchTSTSampler:
    """NSGA-II sampler for 4-parameter × 15-objective PatchTST optimisation."""

    def __init__(
        self,
        study_name: str = "ptst_dgm_multiobj",
        seed: int = 42,
        population_size: int = 20,
    ) -> None:
        self.study = optuna.create_study(
            study_name=study_name,
            directions=_DIRECTIONS,
            sampler=optuna.samplers.NSGAIISampler(
                population_size=population_size,
                seed=seed,
            ),
        )
        self._pending: dict[int, optuna.trial.Trial] = {}

    def suggest(self) -> Tuple[int, Dict[str, int]]:
        """Ask NSGA-II for next architecture parameter configuration."""
        trial = self.study.ask()
        params = {
            name: trial.suggest_int(name, lo, hi)
            for name, (lo, hi) in PARAM_BOUNDS.items()
        }
        self._pending[trial.number] = trial
        return trial.number, params

    def tell(self, trial_number: int, objectives: Dict[str, float]) -> None:
        """Report 15 objective values back to NSGA-II."""
        trial = self._pending.pop(trial_number)
        values = [objectives[k] for k in _OBJ_KEYS]
        self.study.tell(trial, values)

    def summary(self) -> str:
        n_done = len(self.study.trials)
        n_pareto = len(self.study.best_trials)
        if n_pareto == 0:
            return f"trials={n_done}  pareto=empty"
        best_f1 = max(
            sum(t.values[9:12]) / 3 for t in self.study.best_trials  # f1_30/60/90d
        )
        best_fpr = min(
            min(t.values[12:15]) for t in self.study.best_trials  # fpr×3
        )
        return (
            f"trials={n_done}  pareto={n_pareto}  "
            f"best_meanF1={best_f1:.4f}  best_minFPR={best_fpr:.4f}"
        )
