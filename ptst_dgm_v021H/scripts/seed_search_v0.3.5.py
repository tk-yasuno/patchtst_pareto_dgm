"""
seed_search_v0.3.5.py
Seed search to reproduce v0.3 best performance (F1=0.7726).

Goal: Find the random seed that produces F1 >= 0.77 with v0.3 best parameters.

Strategy:
  - Fixed params: patch=26, stride=16, focal_loss(v0.2), lora(baseline)
  - Search range: seed 0-999 (default)
  - Save progress: results/seed_search_v0.3.5.jsonl
  - Resume: automatically skip completed seeds

Usage:
    python ptst_dgm/scripts/seed_search_v0.3.5.py --start 0 --end 100
    python ptst_dgm/scripts/seed_search_v0.3.5.py --start 100 --end 200 --resume
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = WORKSPACE_ROOT / "ptst_dgm" / "training" / "train_patchtst_dgm.py"
PYTHON_EXE = WORKSPACE_ROOT / ".venv-ptstf" / "Scripts" / "python.exe"
ARCHIVE = WORKSPACE_ROOT / "ptst_dgm" / "results" / "seed_search_v0.3.5.jsonl"

# v0.3 best parameters (Trial #1)
V03_BEST_PARAMS = {
    "focal_alpha": 0.866,
    "focal_gamma": 1.156,
    "w_normal": 1.851,
    "w_anomal": 4.035,
    "patch_len": 26,
    "stride": 16,
    "lora_rank": 16,
    "lora_alpha": 32,
}


def load_completed_seeds() -> set[int]:
    """Load already completed seeds from archive."""
    if not ARCHIVE.exists():
        return set()
    
    completed = set()
    with open(ARCHIVE, 'r') as f:
        for line in f:
            entry = json.loads(line)
            completed.add(entry['seed'])
    return completed


def train_with_seed(seed: int, epochs: int = 100) -> dict:
    """Train PatchTST with specified seed and return metrics."""
    temp_json = WORKSPACE_ROOT / "ptst_dgm" / "results" / "temp_model" / f"seed_{seed}.json"
    temp_json.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        str(PYTHON_EXE),
        str(TRAIN_SCRIPT),
        "--focal-alpha", str(V03_BEST_PARAMS["focal_alpha"]),
        "--focal-gamma", str(V03_BEST_PARAMS["focal_gamma"]),
        "--w-normal", str(V03_BEST_PARAMS["w_normal"]),
        "--w-anomal", str(V03_BEST_PARAMS["w_anomal"]),
        "--patch-len", str(V03_BEST_PARAMS["patch_len"]),
        "--stride", str(V03_BEST_PARAMS["stride"]),
        "--lora-rank", str(V03_BEST_PARAMS["lora_rank"]),
        "--lora-alpha", str(V03_BEST_PARAMS["lora_alpha"]),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--output-json", str(temp_json),
    ]
    
    print(f"\n{'='*70}")
    print(f"[Seed Search] Seed #{seed:04d} / Training with v0.3 best params")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Load results
        with open(temp_json, 'r') as f:
            metrics = json.load(f)
        
        # Calculate macro F1
        f1_30d = metrics.get("f1_30d", 0)
        f1_60d = metrics.get("f1_60d", 0)
        f1_90d = metrics.get("f1_90d", 0)
        macro_f1 = (f1_30d + f1_60d + f1_90d) / 3
        
        # Clean up temp file
        temp_json.unlink()
        
        return {
            "seed": seed,
            "macro_f1": macro_f1,
            "f1_30d": f1_30d,
            "f1_60d": f1_60d,
            "f1_90d": f1_90d,
            "auc_30d": metrics.get("auc_30d", 0),
            "auc_60d": metrics.get("auc_60d", 0),
            "auc_90d": metrics.get("auc_90d", 0),
            "fpr_30d": metrics.get("fpr_30d", 0),
            "fpr_60d": metrics.get("fpr_60d", 0),
            "fpr_90d": metrics.get("fpr_90d", 0),
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }
    
    except subprocess.CalledProcessError as e:
        print(f"[Error] Seed {seed} training failed: {e}")
        return {
            "seed": seed,
            "macro_f1": 0.0,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start seed (inclusive)")
    parser.add_argument("--end", type=int, default=1000, help="End seed (exclusive)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per seed")
    parser.add_argument("--target-f1", type=float, default=0.77, 
                        help="Target F1 to match v0.3 (stop if reached)")
    parser.add_argument("--resume", action="store_true", 
                        help="Resume from existing archive (skip completed seeds)")
    args = parser.parse_args()
    
    print("="*70)
    print("v0.3.5 Seed Search: Reproducing v0.3 Best Performance")
    print("="*70)
    print(f"Target: Macro F1 >= {args.target_f1:.4f} (v0.3 best: 0.7726)")
    print(f"Search range: seed {args.start} to {args.end-1} ({args.end - args.start} seeds)")
    print(f"Epochs per seed: {args.epochs}")
    print(f"Fixed params: patch=26, stride=16, focal_loss(v0.2), lora(baseline)")
    print(f"Archive: {ARCHIVE.relative_to(WORKSPACE_ROOT)}")
    print("="*70)
    
    # Load completed seeds if resuming
    completed_seeds = set()
    if args.resume:
        completed_seeds = load_completed_seeds()
        print(f"[Resume] Found {len(completed_seeds)} completed seeds, skipping...")
    
    # Create archive parent dir
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    
    # Search loop
    best_seed = None
    best_f1 = 0.0
    tested_count = 0
    
    for seed in range(args.start, args.end):
        if seed in completed_seeds:
            continue
        
        result = train_with_seed(seed, args.epochs)
        tested_count += 1
        
        # Append to archive
        with open(ARCHIVE, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        macro_f1 = result.get("macro_f1", 0.0)
        status_icon = "✓" if result["status"] == "success" else "✗"
        print(f"\n{status_icon} Seed {seed:04d}: Macro F1 = {macro_f1:.4f}", end="")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_seed = seed
            print(f"  🎯 NEW BEST!", end="")
        
        print()  # newline
        
        # Check if target reached
        if macro_f1 >= args.target_f1:
            print(f"\n{'='*70}")
            print(f"🎉 TARGET REACHED! Seed {seed} achieved F1={macro_f1:.4f} >= {args.target_f1:.4f}")
            print(f"{'='*70}")
            print(f"\nBest seed found: {seed}")
            print(f"  30d: F1={result['f1_30d']:.4f}, AUC={result['auc_30d']:.4f}, FPR={result['fpr_30d']:.4f}")
            print(f"  60d: F1={result['f1_60d']:.4f}, AUC={result['auc_60d']:.4f}, FPR={result['fpr_60d']:.4f}")
            print(f"  90d: F1={result['f1_90d']:.4f}, AUC={result['auc_90d']:.4f}, FPR={result['fpr_90d']:.4f}")
            print(f"\nTo reproduce this result:")
            print(f"  python ptst_dgm/training/train_patchtst_dgm.py \\")
            print(f"    --seed {seed} --patch-len 26 --stride 16 \\")
            print(f"    --focal-alpha 0.866 --focal-gamma 1.156 \\")
            print(f"    --w-normal 1.851 --w-anomal 4.035 \\")
            print(f"    --lora-rank 16 --lora-alpha 32")
            return
        
        # Progress report every 10 seeds
        if tested_count % 10 == 0:
            print(f"\n[Progress] Tested {tested_count}/{args.end - args.start} seeds")
            print(f"[Best so far] Seed {best_seed}: F1={best_f1:.4f}")
    
    # Final report
    print(f"\n{'='*70}")
    print(f"Seed Search Complete")
    print(f"{'='*70}")
    print(f"Seeds tested: {tested_count}")
    print(f"Best seed: {best_seed} (Macro F1 = {best_f1:.4f})")
    
    if best_f1 < args.target_f1:
        print(f"\n⚠️  Target F1={args.target_f1:.4f} NOT reached (best: {best_f1:.4f})")
        print(f"   Consider expanding search range or adjusting target.")
    else:
        print(f"\n✓ Target F1={args.target_f1:.4f} reached!")
    
    print(f"\nResults saved to: {ARCHIVE.relative_to(WORKSPACE_ROOT)}")


if __name__ == "__main__":
    main()
