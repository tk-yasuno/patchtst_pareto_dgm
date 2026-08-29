"""
PatchTST Pareto Top-3 Ensemble Evaluation (v0.5)

Strategy: Soft voting ensemble of v0.3's top-3 Pareto solutions
- Load 3 best configurations from v0.3 Pareto frontier
- Train each model independently
- Average prediction probabilities (soft voting)
- Evaluate ensemble performance on test set
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class EnsembleEvaluator:
    """Evaluates ensemble of Pareto-optimal models"""
    
    def __init__(
        self,
        pareto_jsonl: Path,
        data_path: Path,
        python_exe: Path,
        train_script: Path,
        epochs: int = 100
    ):
        self.pareto_jsonl = pareto_jsonl
        self.data_path = data_path
        self.python_exe = python_exe
        self.train_script = train_script
        self.epochs = epochs
        
    def load_pareto_frontier(self, top_k: int = 3) -> List[Dict]:
        """Load top-K Pareto solutions sorted by macro F1"""
        solutions = []
        with open(self.pareto_jsonl, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                solutions.append(entry)
        
        # Sort by macro_f1 descending
        solutions.sort(key=lambda x: x['macro_f1'], reverse=True)
        return solutions[:top_k]
    
    def train_model(self, params: Dict, trial_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train a single model and return predictions
        
        Returns:
            probs: (N, 3) array of prediction probabilities
            labels: (N, 3) array of ground truth labels
        """
        print(f"\n=== Training Model #{trial_id} ===")
        print(f"  patch_len={params['patch_len']}, stride={params['stride']}")
        
        # Fixed Focal Loss from v0.2 best
        focal_alpha = 0.866
        focal_gamma = 1.156
        w_normal = 1.851
        w_anomal = 4.035
        
        # Fixed LoRA baseline
        lora_rank = 16
        lora_alpha = 32
        
        # Architecture from v0.3
        patch_len = params['patch_len']
        stride = params['stride']
        
        # Unique output directory for this model
        output_dir = Path("ptst_dgm/results/ensemble_models") / f"model_{trial_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            str(self.python_exe),
            str(self.train_script),
            "--data-path", str(self.data_path),
            "--focal-alpha", str(focal_alpha),
            "--focal-gamma", str(focal_gamma),
            "--w-normal", str(w_normal),
            "--w-anomal", str(w_anomal),
            "--patch-len", str(patch_len),
            "--stride", str(stride),
            "--lora-rank", str(lora_rank),
            "--lora-alpha", str(lora_alpha),
            "--epochs", str(self.epochs),
            "--output-dir", str(output_dir),
            "--output-predictions"  # Request prediction output
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Training failed:\n{result.stderr}")
        
        # Load predictions from output directory
        pred_file = output_dir / "predictions.npz"
        
        if not pred_file.exists():
            raise RuntimeError(f"Prediction file not found: {pred_file}")
        
        # Load predictions
        data = np.load(pred_file)
        probs = data['probs']  # (N, 3) probabilities
        labels = data['labels']  # (N, 3) binary labels
        
        print(f"  Loaded predictions: shape={probs.shape}")
        return probs, labels
    
    def evaluate_ensemble(
        self,
        all_probs: List[np.ndarray],
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate ensemble using soft voting
        
        Args:
            all_probs: List of (N, 3) probability arrays from each model
            labels: (N, 3) ground truth binary labels
        
        Returns:
            Dictionary with 15 metrics
        """
        from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
        
        # Average probabilities (soft voting)
        ensemble_probs = np.mean(all_probs, axis=0)  # (N, 3)
        
        # Convert to binary predictions
        ensemble_preds = (ensemble_probs > 0.5).astype(int)
        
        metrics = {}
        horizons = ['30d', '60d', '90d']
        
        for i, horizon in enumerate(horizons):
            y_true = labels[:, i]
            y_pred = ensemble_preds[:, i]
            y_prob = ensemble_probs[:, i]
            
            # AUC
            auc = roc_auc_score(y_true, y_prob)
            
            # Precision, Recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            
            # False Positive Rate
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            metrics[f'auc_{horizon}'] = float(auc)
            metrics[f'precision_{horizon}'] = float(precision)
            metrics[f'recall_{horizon}'] = float(recall)
            metrics[f'f1_{horizon}'] = float(f1)
            metrics[f'fpr_{horizon}'] = float(fpr)
        
        # Macro averages
        metrics['macro_f1'] = np.mean([metrics[f'f1_{h}'] for h in horizons])
        metrics['mean_fpr'] = np.mean([metrics[f'fpr_{h}'] for h in horizons])
        
        return metrics
    
    def run(self, top_k: int = 3) -> Dict:
        """
        Run full ensemble evaluation
        
        Returns:
            Dictionary with ensemble results and comparison
        """
        print("=" * 60)
        print("PatchTST Pareto Top-3 Ensemble Evaluation (v0.5)")
        print("=" * 60)
        
        # Load Pareto frontier
        print(f"\n[1/4] Loading Pareto frontier from {self.pareto_jsonl}")
        pareto_solutions = self.load_pareto_frontier(top_k)
        print(f"  Top-{top_k} solutions loaded:")
        for i, sol in enumerate(pareto_solutions):
            print(f"    #{i+1} | F1={sol['macro_f1']:.4f} | patch={sol['params']['patch_len']}, stride={sol['params']['stride']}")
        
        # Train each model
        print(f"\n[2/4] Training {top_k} models...")
        all_probs = []
        labels = None
        
        for i, solution in enumerate(pareto_solutions):
            probs, labs = self.train_model(solution['params'], trial_id=i+1)
            all_probs.append(probs)
            if labels is None:
                labels = labs
        
        # Evaluate ensemble
        print(f"\n[3/4] Evaluating ensemble (soft voting)...")
        ensemble_metrics = self.evaluate_ensemble(all_probs, labels)
        
        print(f"\n  Ensemble Results:")
        print(f"    Macro F1:  {ensemble_metrics['macro_f1']:.4f}")
        print(f"    Mean FPR:  {ensemble_metrics['mean_fpr']:.4f}")
        print(f"    30d: F1={ensemble_metrics['f1_30d']:.4f}, FPR={ensemble_metrics['fpr_30d']:.4f}")
        print(f"    60d: F1={ensemble_metrics['f1_60d']:.4f}, FPR={ensemble_metrics['fpr_60d']:.4f}")
        print(f"    90d: F1={ensemble_metrics['f1_90d']:.4f}, FPR={ensemble_metrics['fpr_90d']:.4f}")
        
        # Compare with individual models
        print(f"\n[4/4] Comparison with v0.3 best...")
        v03_best_f1 = pareto_solutions[0]['macro_f1']
        improvement = ((ensemble_metrics['macro_f1'] - v03_best_f1) / v03_best_f1) * 100
        
        print(f"    v0.3 Best:     F1={v03_best_f1:.4f}")
        print(f"    v0.5 Ensemble: F1={ensemble_metrics['macro_f1']:.4f}")
        print(f"    Improvement:   {improvement:+.2f}%")
        
        # Prepare result summary
        result = {
            'pareto_solutions': pareto_solutions,
            'ensemble_metrics': ensemble_metrics,
            'v03_best_f1': v03_best_f1,
            'improvement_pct': improvement
        }
        
        print("\n" + "=" * 60)
        print("Ensemble evaluation complete!")
        print("=" * 60)
        
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Pareto top-K ensemble")
    parser.add_argument(
        "--pareto-jsonl",
        type=Path,
        default=Path("ptst_dgm/results/ptst_archive_v0.3_pareto.jsonl"),
        help="Path to Pareto frontier JSONL"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/golden_testset"),
        help="Path to golden testset"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top Pareto solutions to ensemble"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs per model"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ptst_dgm/results/ensemble_v0.5_result.json"),
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    # Paths
    python_exe = Path(".venv-ptstf/Scripts/python.exe")
    train_script = Path("ptst_dgm/training/train_patchtst_dgm.py")
    
    # Run evaluation
    evaluator = EnsembleEvaluator(
        pareto_jsonl=args.pareto_jsonl,
        data_path=args.data_path,
        python_exe=python_exe,
        train_script=train_script,
        epochs=args.epochs
    )
    
    result = evaluator.run(top_k=args.top_k)
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
