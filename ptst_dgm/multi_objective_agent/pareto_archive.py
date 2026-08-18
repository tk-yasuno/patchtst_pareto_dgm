"""
pareto_archive.py
Pareto frontier management for 15-objective PatchTST DGM.

Objectives 0-11  → maximise (auc×3, precision×3, recall×3, f1×3)
Objectives 12-14 → minimise (fpr×3)

Internally FPR values are negated so all comparisons use "higher is better".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ptst_dgm.agent.archive import OBJECTIVE_KEYS, HORIZONS

_N_OBJ = 15
_FPR_INDICES = [12, 13, 14]  # positions of fpr_30d, fpr_60d, fpr_90d


def _to_internal(objectives: Dict[str, float]) -> List[float]:
    """Convert objective dict to internal list; negate FPR so all are 'higher=better'."""
    vals = [objectives[k] for k in OBJECTIVE_KEYS]
    for i in _FPR_INDICES:
        vals[i] = -vals[i]
    return vals


def _dominates(a: List[float], b: List[float]) -> bool:
    """Return True if a Pareto-dominates b (all >= and at least one >)."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


class ParetoArchive:
    """Non-dominated front for mixed-direction 15-objective optimisation."""

    def __init__(self) -> None:
        # Each entry: (params_dict, objectives_dict, trial_number, internal_vals)
        self._solutions: List[Tuple[dict, dict, int, List[float]]] = []

    def add(
        self,
        params: Dict[str, float],
        objectives: Dict[str, float],
        trial_number: int,
    ) -> bool:
        """Add solution if non-dominated. Returns True if added."""
        internal = _to_internal(objectives)

        if any(_dominates(s[3], internal) for s in self._solutions):
            return False

        # Remove solutions that new entry dominates
        self._solutions = [
            s for s in self._solutions if not _dominates(internal, s[3])
        ]
        self._solutions.append((params, objectives, trial_number, internal))
        return True

    def get_size(self) -> int:
        return len(self._solutions)

    def get_best_by_macro_f1(self) -> Optional[Tuple[dict, dict, int]]:
        if not self._solutions:
            return None
        best = max(
            self._solutions,
            key=lambda s: sum(s[1][f"f1_{h}d"] for h in HORIZONS) / 3,
        )
        return best[0], best[1], best[2]

    def get_best_by_min_fpr(self) -> Optional[Tuple[dict, dict, int]]:
        if not self._solutions:
            return None
        best = min(
            self._solutions,
            key=lambda s: max(s[1][f"fpr_{h}d"] for h in HORIZONS),
        )
        return best[0], best[1], best[2]

    def get_best_for_parent(self) -> Optional[Tuple[dict, dict, int]]:
        """Get best solution from Pareto frontier for parent selection (by macro F1)."""
        return self.get_best_by_macro_f1()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for params, objs, trial_number, _ in self._solutions:
                entry = {
                    "trial_number": trial_number,
                    "params": params,
                    "objectives": objs,
                    "macro_f1": sum(objs[f"f1_{h}d"] for h in HORIZONS) / 3,
                    "mean_fpr":  sum(objs[f"fpr_{h}d"] for h in HORIZONS) / 3,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def summary(self) -> str:
        if not self._solutions:
            return "Pareto frontier: empty"
        best_f1 = max(
            sum(s[1][f"f1_{h}d"] for h in HORIZONS) / 3 for s in self._solutions
        )
        best_fpr = min(
            max(s[1][f"fpr_{h}d"] for h in HORIZONS) for s in self._solutions
        )
        return (
            f"Pareto frontier: {len(self._solutions)} solutions  "
            f"best_macroF1={best_f1:.4f}  best_maxFPR={best_fpr:.4f}"
        )
