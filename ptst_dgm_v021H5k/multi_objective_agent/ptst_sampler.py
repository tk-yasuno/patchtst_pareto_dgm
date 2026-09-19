"""
ptst_sampler.py
NSGA-II multi-objective sampler for PatchTST DGM.

v0.2.1H - Horizon-specific 9D optimization:
  9 control variables (3 params × 3 horizons):
    Per-horizon Focal Loss: alpha [0.5, 0.9], gamma [0.6, 2.0]
    Per-horizon Class weights: w_normal [0.3, 5.5]
  Constraint: w_normal_XXd + w_anomal_XXd = 5.88 (for each horizon)
  
  Architecture: FIXED v0.2.3 best (patch_len=26, stride=16)
  LoRA: FIXED v0.2.3 best (rank=16, alpha=47)
  Seed: RANDOM (diversity for DGM exploration)

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

# v0.2.1H: Horizon-specific 9D optimization
# Constraint: w_normal_XXd + w_anomal_XXd = 5.88 (for each horizon independently)
PARAM_BOUNDS = {
    # 30d horizon
    "focal_alpha_30d": (0.5, 0.9),
    "focal_gamma_30d": (0.6, 2.0),
    "w_normal_30d":    (0.3, 5.5),
    # 60d horizon
    "focal_alpha_60d": (0.5, 0.9),
    "focal_gamma_60d": (0.6, 2.0),
    "w_normal_60d":    (0.3, 5.5),
    # 90d horizon
    "focal_alpha_90d": (0.5, 0.9),
    "focal_gamma_90d": (0.6, 2.0),
    "w_normal_90d":    (0.3, 5.5),
}

# Fixed Architecture parameters (v0.2.3 best from 8D optimization)
FIXED_ARCH_PARAMS = {
    "patch_len": 26,
    "stride": 16,
}

# Fixed LoRA parameters (v0.2.3 best from 8D optimization)
FIXED_LORA_PARAMS = {
    "lora_rank": 16,
    "lora_alpha": 47,
}

# Random seed: NOT FIXED (diversity for DGM exploration)
# No SEED_RANGE - seed will be None (random per trial)


class PatchTSTSampler:
    """NSGA-II sampler for 9-parameter × 15-objective PatchTST optimisation (v0.2.1H - Horizon-specific optimization)."""

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
        """Ask NSGA-II for next parameter configuration (v0.2.1H: 9D horizon-specific Focal Loss)."""
        trial = self.study.ask()
        params = {}
        
        # Sample 9D parameters (3 params × 3 horizons)
        for horizon in [30, 60, 90]:
            h = f"{horizon}d"
            params[f"focal_alpha_{h}"] = trial.suggest_float(f"focal_alpha_{h}", *PARAM_BOUNDS[f"focal_alpha_{h}"])
            params[f"focal_gamma_{h}"] = trial.suggest_float(f"focal_gamma_{h}", *PARAM_BOUNDS[f"focal_gamma_{h}"])
            params[f"w_normal_{h}"] = trial.suggest_float(f"w_normal_{h}", *PARAM_BOUNDS[f"w_normal_{h}"])
            
            # Apply constraint per-horizon: w_normal_XXd + w_anomal_XXd = 5.88
            params[f"w_anomal_{h}"] = 5.88 - params[f"w_normal_{h}"]
        
        # Add fixed Architecture and LoRA parameters
        params.update(FIXED_ARCH_PARAMS)
        params.update(FIXED_LORA_PARAMS)
        
        # Random seed (no fixing for diversity)
        params["seed"] = None
        
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
