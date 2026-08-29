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
import shutil
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ptst_dgm_v021C5.agent.archive import PatchTSTAgentEntry, PatchTSTArchive, HORIZONS, OBJECTIVE_KEYS
from ptst_dgm_v021C5.agent.evaluator import PatchTSTEvaluator
from ptst_dgm_v021C5.multi_objective_agent.ptst_agent import PatchTSTMultiObjectiveAgent
from ptst_dgm_v021C5.multi_objective_agent.pareto_archive import ParetoArchive


class PatchTSTDGMLoop:
    def __init__(
        self,
        model_name: str,
        archive_path: Path,
        evaluator: PatchTSTEvaluator,
        total_budget: int,
        population_size: int = 20,
        checkpoint_interval: int = 100,  # v0.3.7: checkpoint every N iterations
    ) -> None:
        self.model_name = model_name
        self.archive = PatchTSTArchive(archive_path)
        self.evaluator = evaluator
        self.total_budget = total_budget
        self.checkpoint_interval = checkpoint_interval
        self.agent = PatchTSTMultiObjectiveAgent(model_name, population_size=population_size)
        self.pareto = ParetoArchive()
        self._log_path = archive_path.parent / f"{archive_path.stem}_log.jsonl"
        self._pareto_path = archive_path.parent / f"{archive_path.stem}_pareto.jsonl"
        
        # v0.3.7: checkpoint directory
        archive_version = archive_path.stem.replace("ptst_archive_", "")
        self._checkpoint_dir = WORKSPACE_ROOT / "models" / f"{archive_version}_checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run(self, dry_run: bool = False) -> None:
        print("=" * 80)
        print("[PatchTSTDGM] Multi-Objective Pareto DGM (PatchTST)")
        print(f"[PatchTSTDGM] Model      : {self.model_name}")
        print(f"[PatchTSTDGM] Budget     : {self.total_budget} iterations")
        print(f"[PatchTSTDGM] Objectives : 15 (AUC/P/R/F1/FPR × 30d/60d/90d)")
        print(f"[PatchTSTDGM] Optimizer  : NSGA-II (Pareto frontier)")
        print(f"[PatchTSTDGM] Archive    : {len(self.archive)} agents")
        print(f"[PatchTSTDGM] Checkpoint : Every {self.checkpoint_interval} iters → {self._checkpoint_dir}")
        if dry_run:
            print(f"[PatchTSTDGM] Mode       : DRY RUN")
        print("=" * 80)

        accepted = 0
        # v0.3.7: track interval best for checkpointing
        interval_best_f1 = -float('inf')
        interval_best_data = None  # (trial_num, params, objectives, iteration)
        
        for t in range(1, self.total_budget + 1):
            print("=" * 60)
            print(f"[Iter {t:02d}/{self.total_budget}]")

            # Select parent from Pareto frontier (or archive if empty)
            pareto_parent = self.pareto.get_best_for_parent()
            if pareto_parent:
                params, objs, trial_num = pareto_parent
                # Create a temporary entry for display
                parent_macro_f1 = sum(objs[f"f1_{h}d"] for h in HORIZONS) / 3
                # v0.3.7: Show current Pareto frontier status
                pareto_size = self.pareto.get_size()
                print(f"[Parent] BEST from Pareto frontier ({pareto_size} solutions)")
                print(f"  Trial #{trial_num}  macro_F1={parent_macro_f1:.4f}")
                print(f"  Focal: α={params['focal_alpha']:.3f} γ={params['focal_gamma']:.3f} "
                      f"w_n={params['w_normal']:.3f} w_a={params['w_anomal']:.3f}")
                print(f"  Arch: patch={params['patch_len']} stride={params['stride']}")
                print(f"  LoRA: rank={params['lora_rank']} alpha={params['lora_alpha']}")
                # Create PatchTSTAgentEntry for agent.generate()
                parent = PatchTSTAgentEntry.create_child(
                    parent_id=f"pareto_{trial_num}",
                    coding_model=self.model_name,
                    focal_alpha=params['focal_alpha'],
                    focal_gamma=params['focal_gamma'],
                    w_normal=params['w_normal'],
                    w_anomal=params['w_anomal'],
                    patch_len=params['patch_len'],
                    stride=params['stride'],
                    lora_rank=params['lora_rank'],
                    lora_alpha=params['lora_alpha'],
                    objectives=objs,
                    rationale=f"Parent from Pareto frontier (trial #{trial_num})",
                )
            elif len(self.archive) > 0:
                # Fallback to archive best if Pareto is empty but archive has entries
                parent = self.archive.get_best()
                print(f"[Parent] id={parent.id[:8]}  macro_F1={parent.macro_f1:.4f}")
                print(f"  Focal: α={parent.focal_alpha:.3f} γ={parent.focal_gamma:.3f} "
                      f"w_n={parent.w_normal:.3f} w_a={parent.w_anomal:.3f}")
                print(f"  Arch: patch={parent.patch_len} stride={parent.stride}")
                print(f"  LoRA: rank={parent.lora_rank} alpha={parent.lora_alpha}")
            else:
                # First iteration with empty archive: use baseline parameters
                print(f"[Parent] baseline (first iteration)")
                baseline_objs = {k: 0.5 for k in OBJECTIVE_KEYS}
                parent = PatchTSTAgentEntry.create_baseline(baseline_objs)

            proposal, trial_number = self.agent.generate(parent, max_retries=2)

            focal_alpha = proposal["focal_alpha"]
            focal_gamma = proposal["focal_gamma"]
            w_normal    = proposal["w_normal"]
            w_anomal    = proposal["w_anomal"]
            patch_len   = proposal["patch_len"]
            stride      = proposal["stride"]
            lora_rank   = proposal["lora_rank"]
            lora_alpha  = proposal["lora_alpha"]
            seed        = proposal.get("seed", None)  # v0.3.6: seed support
            rat         = proposal.get("rationale", "")

            seed_info = f" seed={seed}" if seed is not None else ""
            print(f"[Proposal] Focal: α={focal_alpha:.3f} γ={focal_gamma:.3f} "
                  f"w_n={w_normal:.3f} w_a={w_anomal:.3f}")
            print(f"[Proposal] Arch: patch={patch_len} stride={stride}{seed_info}")
            print(f"[Proposal] LoRA: rank={lora_rank} alpha={lora_alpha}")

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
                objectives = self.evaluator.evaluate(
                    focal_alpha, focal_gamma, w_normal, w_anomal,
                    patch_len, stride, lora_rank, lora_alpha, seed
                )

            self.agent.tell_result(trial_number, objectives)

            is_pareto = self.pareto.add(
                {
                    "focal_alpha": focal_alpha,
                    "focal_gamma": focal_gamma,
                    "w_normal": w_normal,
                    "w_anomal": w_anomal,
                    "patch_len": patch_len,
                    "stride": stride,
                    "lora_rank": lora_rank,
                    "lora_alpha": lora_alpha,
                },
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
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    w_normal=w_normal,
                    w_anomal=w_anomal,
                    patch_len=patch_len,
                    stride=stride,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    objectives=objectives,
                    rationale=rat,
                )
                self.archive.add(entry)
            else:
                print(f"[Reject] Dominated by existing solutions")

            self._log(
                t, trial_number, parent.id,
                focal_alpha, focal_gamma, w_normal, w_anomal,
                patch_len, stride, lora_rank, lora_alpha,
                objectives, macro_f1, mean_fpr, is_pareto, rat
            )
            
            # v0.3.7: track best in current interval
            if macro_f1 > interval_best_f1:
                interval_best_f1 = macro_f1
                interval_best_data = (
                    trial_number,
                    {
                        "focal_alpha": focal_alpha,
                        "focal_gamma": focal_gamma,
                        "w_normal": w_normal,
                        "w_anomal": w_anomal,
                        "patch_len": patch_len,
                        "stride": stride,
                        "lora_rank": lora_rank,
                        "lora_alpha": lora_alpha,
                        "seed": seed,
                    },
                    objectives,
                    t
                )
            
            # v0.3.7: checkpoint every N iterations or at end
            if t % self.checkpoint_interval == 0 or t == self.total_budget:
                if interval_best_data and not dry_run:
                    self._save_checkpoint(*interval_best_data)
                    # Reset interval tracker
                    interval_best_f1 = -float('inf')
                    interval_best_data = None
            
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

    def _save_checkpoint(
        self, trial_num: int, params: dict, objectives: dict, iteration: int
    ) -> None:
        """Save model checkpoint with metadata (v0.3.7)."""
        macro_f1 = sum(objectives[f"f1_{h}d"] for h in HORIZONS) / 3
        
        # Model checkpoint filename
        checkpoint_name = f"trial_{trial_num:04d}_iter_{iteration:04d}_f1_{macro_f1:.4f}.pt"
        checkpoint_path = self._checkpoint_dir / checkpoint_name
        
        # Copy model from temp_model to checkpoint directory
        temp_model_path = self.evaluator.output_dir / "best_ptst_dgm.pt"
        if temp_model_path.exists():
            shutil.copy(temp_model_path, checkpoint_path)
            print(f"[Checkpoint] Saved → {checkpoint_path.relative_to(WORKSPACE_ROOT)}")
        else:
            print(f"[Checkpoint] WARNING: Model not found at {temp_model_path}")
            return
        
        # Save metadata
        meta = {
            "trial_number": trial_num,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "objectives": objectives,
            "macro_f1": macro_f1,
            "mean_fpr": sum(objectives[f"fpr_{h}d"] for h in HORIZONS) / 3,
            "model_path": str(checkpoint_path.relative_to(WORKSPACE_ROOT)),
        }
        
        meta_path = checkpoint_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        print(f"[Checkpoint] Metadata → {meta_path.relative_to(WORKSPACE_ROOT)}")
        print(f"[Checkpoint] F1={macro_f1:.4f}  Trial=#{trial_num}  Iteration={iteration}")

    def _log(
        self, iteration, trial_number, parent_id,
        focal_alpha, focal_gamma, w_normal, w_anomal,
        patch_len, stride, lora_rank, lora_alpha,
        objectives, macro_f1, mean_fpr,
        is_pareto, rationale,
    ) -> None:
        entry = {
            "iteration": iteration,
            "trial_number": trial_number,
            "parent_id": parent_id,
            "params": {
                "focal_alpha": focal_alpha,
                "focal_gamma": focal_gamma,
                "w_normal": w_normal,
                "w_anomal": w_anomal,
                "patch_len": patch_len,
                "stride": stride,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
            },
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
    parser.add_argument("--archive",         default="ptst_dgm_v021/results/ptst_archive_v021.jsonl")
    parser.add_argument("--total-budget",    type=int, default=50)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Save best model every N iterations (v0.3.7)")
    parser.add_argument("--dry-run",         action="store_true")
    parser.add_argument("--python-exe",
                        default=".venv-ptstf/Scripts/python.exe")
    parser.add_argument("--script",
                        default="ptst_dgm_v021/training/train_patchtst_dgm.py")
    parser.add_argument("--data-path",
                        default="data/golden_testset")
    parser.add_argument("--output-dir",
                        default="ptst_dgm_v021/results/temp_model")
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
        checkpoint_interval=args.checkpoint_interval,
    )
    loop.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
