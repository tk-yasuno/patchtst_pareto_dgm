"""
ptst_sampler.py
NSGA-II multi-objective sampler for PatchTST DGM.

4 control variables (v0.2 - refined ranges based on 50-iter insights):
  focal_alpha  [0.75, 0.90]  ← High confidence focus (best performers)
  focal_gamma  [1.00, 2.00]  ← Moderate hard example focus (optimal range)
  w_normal     [1.50, 2.50]  ← Balanced normal weight (2:4 ratio)
  w_anomal     [3.50, 5.00]  ← Moderate anomaly emphasis (avoids extremes)

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

# v0.2: Refined parameter bounds based on 50-iteration experiment insights
# Key discoveries: High alpha (0.8+) + Low gamma (1.0-1.5) + Balanced weights (1.5:4.0)
PARAM_BOUNDS = {
    "focal_alpha": (0.75, 0.90),  # v0.1: (0.10, 0.90)
    "focal_gamma": (1.00, 2.00),  # v0.1: (0.50, 5.00)
    "w_normal":    (1.50, 2.50),  # v0.1: (0.10, 5.00)
    "w_anomal":    (3.50, 5.00),  # v0.1: (0.50, 10.0)
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

    def suggest(self) -> Tuple[int, Dict[str, float]]:
        """Ask NSGA-II for next parameter configuration."""
        trial = self.study.ask()
        params = {
            name: trial.suggest_float(name, lo, hi)
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
