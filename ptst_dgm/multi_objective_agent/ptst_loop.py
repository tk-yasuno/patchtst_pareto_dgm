"""
ptst_loop.py
Multi-Objective DGM loop for PatchTST anomaly detection.

Usage (CLI):
    python -m ptst_dgm.multi_objective_agent.ptst_loop \
        --archive ptst_dgm/results/ptst_archive.jsonl \
        --total-budget 50 \
        --python-exe .venv-ptstf/Scripts/python.exe
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ptst_dgm.agent.archive import PatchTSTAgentEntry, PatchTSTArchive, HORIZONS, OBJECTIVE_KEYS
from ptst_dgm.agent.evaluator import PatchTSTEvaluator
from ptst_dgm.multi_objective_agent.ptst_agent import PatchTSTMultiObjectiveAgent
from ptst_dgm.multi_objective_agent.pareto_archive import ParetoArchive


class PatchTSTDGMLoop:
    def __init__(
        self,
        model_name: str,
        archive_path: Path,
        evaluator: PatchTSTEvaluator,
        total_budget: int,
        population_size: int = 20,
    ) -> None:
        self.model_name = model_name
        self.archive = PatchTSTArchive(archive_path)
        self.evaluator = evaluator
        self.total_budget = total_budget
        self.agent = PatchTSTMultiObjectiveAgent(model_name, population_size=population_size)
        self.pareto = ParetoArchive()
        self._log_path = archive_path.parent / f"{archive_path.stem}_log.jsonl"
        self._pareto_path = archive_path.parent / f"{archive_path.stem}_pareto.jsonl"

    def run(self, dry_run: bool = False) -> None:
        print("=" * 80)
        print("[PatchTSTDGM] Multi-Objective Pareto DGM (PatchTST)")
        print(f"[PatchTSTDGM] Model      : {self.model_name}")
        print(f"[PatchTSTDGM] Budget     : {self.total_budget} iterations")
        print(f"[PatchTSTDGM] Objectives : 15 (AUC/P/R/F1/FPR × 30d/60d/90d)")
        print(f"[PatchTSTDGM] Optimizer  : NSGA-II (Pareto frontier)")
        print(f"[PatchTSTDGM] Archive    : {len(self.archive)} agents")
        if dry_run:
            print(f"[PatchTSTDGM] Mode       : DRY RUN")
        print("=" * 80)

        accepted = 0
        for t in range(1, self.total_budget + 1):
            print("=" * 60)
            print(f"[Iter {t:02d}/{self.total_budget}]")

            # Select parent from Pareto frontier (or archive if empty)
            pareto_parent = self.pareto.get_best_for_parent()
            if pareto_parent:
                params, objs, trial_num = pareto_parent
                # Create a temporary entry for display
                parent_macro_f1 = sum(objs[f"f1_{h}d"] for h in HORIZONS) / 3
                print(f"[Parent] trial=#{trial_num}  macro_F1={parent_macro_f1:.4f}  "
                      f"α={params['focal_alpha']:.3f}  γ={params['focal_gamma']:.3f}  "
                      f"w_n={params['w_normal']:.3f}  w_a={params['w_anomal']:.3f}")
                # Create PatchTSTAgentEntry for agent.generate()
                parent = PatchTSTAgentEntry.create_child(
                    parent_id=f"pareto_{trial_num}",
                    coding_model=self.model_name,
                    focal_alpha=params['focal_alpha'],
                    focal_gamma=params['focal_gamma'],
                    w_normal=params['w_normal'],
                    w_anomal=params['w_anomal'],
                    objectives=objs,
                    rationale=f"Parent from Pareto frontier (trial #{trial_num})",
                )
            elif len(self.archive) > 0:
                # Fallback to archive best if Pareto is empty but archive has entries
                parent = self.archive.get_best()
                print(f"[Parent] id={parent.id[:8]}  macro_F1={parent.macro_f1:.4f}  "
                      f"α={parent.focal_alpha:.3f}  γ={parent.focal_gamma:.3f}  "
                      f"w_n={parent.w_normal:.3f}  w_a={parent.w_anomal:.3f}")
            else:
                # First iteration with empty archive: use baseline parameters
                print(f"[Parent] baseline (first iteration)")
                baseline_objs = {k: 0.5 for k in OBJECTIVE_KEYS}
                parent = PatchTSTAgentEntry.create_baseline(baseline_objs)

            proposal, trial_number = self.agent.generate(parent, max_retries=2)

            fa  = proposal["focal_alpha"]
            fg  = proposal["focal_gamma"]
            wn  = proposal["w_normal"]
            wa  = proposal["w_anomal"]
            rat = proposal.get("rationale", "")

            print(f"[Proposal] focal_alpha={fa:.3f}  focal_gamma={fg:.3f}  "
                  f"w_normal={wn:.3f}  w_anomal={wa:.3f}")

            if dry_run:
                import random
                objectives = {
                    f"auc_{h}d":       0.85 + random.random() * 0.10 for h in HORIZONS
                }
                objectives.update({f"precision_{h}d": 0.60 + random.random() * 0.20 for h in HORIZONS})
                objectives.update({f"recall_{h}d":    0.60 + random.random() * 0.20 for h in HORIZONS})
                objectives.update({f"f1_{h}d":        0.65 + random.random() * 0.20 for h in HORIZONS})
                objectives.update({f"fpr_{h}d":       0.01 + random.random() * 0.05 for h in HORIZONS})
            else:
                objectives = self.evaluator.evaluate(fa, fg, wn, wa)

            self.agent.tell_result(trial_number, objectives)

            is_pareto = self.pareto.add(
                {"focal_alpha": fa, "focal_gamma": fg, "w_normal": wn, "w_anomal": wa},
                objectives,
                trial_number,
            )

            macro_f1 = sum(objectives[f"f1_{h}d"] for h in HORIZONS) / 3
            mean_fpr = sum(objectives[f"fpr_{h}d"] for h in HORIZONS) / 3

            if is_pareto:
                print(f"[Pareto] Added to frontier (now {self.pareto.get_size()} solutions)")
                accepted += 1
                entry = PatchTSTAgentEntry.create_child(
                    parent_id=parent.id,
                    coding_model=self.model_name,
                    focal_alpha=fa, focal_gamma=fg,
                    w_normal=wn, w_anomal=wa,
                    objectives=objectives,
                    rationale=rat,
                )
                self.archive.add(entry)
            else:
                print(f"[Reject] Dominated by existing solutions")

            self._log(t, trial_number, parent.id, fa, fg, wn, wa,
                      objectives, macro_f1, mean_fpr, is_pareto, rat)
            print()

        print("=" * 80)
        pct = accepted / self.total_budget * 100
        print(f"[PatchTSTDGM] Complete!  Pareto-optimal: {accepted}/{self.total_budget} ({pct:.1f}%)")
        print(f"[Pareto] {self.pareto.summary()}")

        self.pareto.save(self._pareto_path)
        print(f"[Pareto] Saved → {self._pareto_path}")

        best = self.pareto.get_best_by_macro_f1()
        if best:
            params, objs, trial = best
            mf1 = sum(objs[f"f1_{h}d"] for h in HORIZONS) / 3
            mfpr = sum(objs[f"fpr_{h}d"] for h in HORIZONS) / 3
            print(f"[Best F1] trial=#{trial}  macro_F1={mf1:.4f}  mean_FPR={mfpr:.4f}  {params}")

        best_fpr = self.pareto.get_best_by_min_fpr()
        if best_fpr:
            params, objs, trial = best_fpr
            mf1 = sum(objs[f"f1_{h}d"] for h in HORIZONS) / 3
            mfpr = sum(objs[f"fpr_{h}d"] for h in HORIZONS) / 3
            print(f"[Best FPR] trial=#{trial}  macro_F1={mf1:.4f}  mean_FPR={mfpr:.4f}  {params}")

    def _log(
        self, iteration, trial_number, parent_id,
        fa, fg, wn, wa,
        objectives, macro_f1, mean_fpr,
        is_pareto, rationale,
    ) -> None:
        entry = {
            "iteration": iteration,
            "trial_number": trial_number,
            "parent_id": parent_id,
            "params": {"focal_alpha": fa, "focal_gamma": fg, "w_normal": wn, "w_anomal": wa},
            "objectives": objectives,
            "macro_f1": macro_f1,
            "mean_fpr": mean_fpr,
            "is_pareto_optimal": is_pareto,
            "rationale": rationale,
            "timestamp": datetime.now().isoformat(),
        }
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           default="codestral:latest")
    parser.add_argument("--archive",         default="ptst_dgm/results/ptst_archive.jsonl")
    parser.add_argument("--total-budget",    type=int, default=50)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--dry-run",         action="store_true")
    parser.add_argument("--python-exe",
                        default=".venv-ptstf/Scripts/python.exe")
    parser.add_argument("--script",
                        default="ptst_dgm/training/train_patchtst_dgm.py")
    parser.add_argument("--data-path",
                        default="data/golden_testset")
    parser.add_argument("--output-dir",
                        default="ptst_dgm/results/temp_model")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    evaluator = PatchTSTEvaluator(
        python_exe=WORKSPACE_ROOT / args.python_exe,
        script_path=WORKSPACE_ROOT / args.script,
        data_path=WORKSPACE_ROOT / args.data_path,
        output_dir=WORKSPACE_ROOT / args.output_dir,
        epochs=args.epochs,
    )

    loop = PatchTSTDGMLoop(
        model_name=args.model,
        archive_path=WORKSPACE_ROOT / args.archive,
        evaluator=evaluator,
        total_budget=args.total_budget,
        population_size=args.population_size,
    )
    loop.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
