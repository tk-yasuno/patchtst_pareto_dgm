"""
Compare v0.3 vs v0.3.7 Performance
Analyze performance degradation causes
"""
import json
from pathlib import Path

def load_pareto(filepath):
    """Load Pareto frontier solutions."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def load_log_stats(filepath):
    """Load log file and compute statistics."""
    with open(filepath, 'r', encoding='utf-8') as f:
        trials = [json.loads(line) for line in f if line.strip()]
    
    pareto_count = sum(1 for t in trials if t.get('is_pareto_optimal', False))
    macro_f1s = [t['macro_f1'] for t in trials]
    
    return {
        'total_trials': len(trials),
        'pareto_count': pareto_count,
        'pareto_rate': pareto_count / len(trials) if trials else 0,
        'best_f1': max(macro_f1s) if macro_f1s else 0,
        'mean_f1': sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0,
        'median_f1': sorted(macro_f1s)[len(macro_f1s)//2] if macro_f1s else 0
    }

# Load v0.3 data
print("=" * 80)
print("v0.3 vs v0.3.7 Performance Comparison")
print("=" * 80)
print()

v03_pareto = load_pareto('ptst_dgm/results/ptst_archive_v0.3_pareto.jsonl')
v03_log = load_log_stats('ptst_dgm/results/ptst_archive_v0.3_log.jsonl')
best_v03 = max(v03_pareto, key=lambda s: s['macro_f1'])

print("v0.3 (200 iterations):")
print(f"  Pareto solutions: {len(v03_pareto)} / {v03_log['total_trials']} ({v03_log['pareto_rate']*100:.1f}%)")
print(f"  Best macro-F1: {best_v03['macro_f1']:.4f} (Trial #{best_v03['trial_number']})")
print(f"  Mean FPR: {best_v03['mean_fpr']:.4f}")
print(f"  Parameters:")
print(f"    patch_len={best_v03['params']['patch_len']}, stride={best_v03['params']['stride']}")
if 'focal_alpha' in best_v03['params']:
    print(f"    focal_alpha={best_v03['params']['focal_alpha']:.3f}, focal_gamma={best_v03['params']['focal_gamma']:.3f}")
print()

# Load v0.3.7 data
v037_pareto = load_pareto('ptst_dgm/results/ptst_archive_v0.3.7_300iters_pareto.jsonl')
v037_log = load_log_stats('ptst_dgm/results/ptst_archive_v0.3.7_300iters_log.jsonl')
best_v037 = max(v037_pareto, key=lambda s: s['macro_f1'])

print("v0.3.7 (300 iterations):")
print(f"  Pareto solutions: {len(v037_pareto)} / {v037_log['total_trials']} ({v037_log['pareto_rate']*100:.1f}%)")
print(f"  Best macro-F1: {best_v037['macro_f1']:.4f} (Trial #{best_v037['trial_number']})")
print(f"  Mean FPR: {best_v037['mean_fpr']:.4f}")
print(f"  Parameters:")
print(f"    patch_len={best_v037['params']['patch_len']}, stride={best_v037['params']['stride']}")
if 'focal_alpha' in best_v037['params']:
    print(f"    focal_alpha={best_v037['params']['focal_alpha']:.3f}, focal_gamma={best_v037['params']['focal_gamma']:.3f}")
print()

# Performance comparison
print("=" * 80)
print("Performance Degradation Analysis")
print("=" * 80)
f1_drop = best_v03['macro_f1'] - best_v037['macro_f1']
f1_drop_pct = (1 - best_v037['macro_f1'] / best_v03['macro_f1']) * 100

print(f"Macro-F1 drop: {f1_drop:.4f} ({f1_drop_pct:.1f}% degradation)")
print(f"Pareto optimality rate: v0.3={v03_log['pareto_rate']*100:.1f}%, v0.3.7={v037_log['pareto_rate']*100:.1f}%")
print(f"Mean F1: v0.3={v03_log['mean_f1']:.4f}, v0.3.7={v037_log['mean_f1']:.4f}")
print()

# Per-horizon comparison
print("Per-horizon F1 comparison (best solutions):")
for horizon in ['30d', '60d', '90d']:
    v03_f1 = best_v03['objectives'][f'f1_{horizon}']
    v037_f1 = best_v037['objectives'][f'f1_{horizon}']
    print(f"  {horizon}: v0.3={v03_f1:.4f}, v0.3.7={v037_f1:.4f} (Δ={v03_f1-v037_f1:.4f})")
print()

# Parameter comparison
print("Parameter space comparison:")
print(f"  patch_len: v0.3={best_v03['params']['patch_len']}, v0.3.7={best_v037['params']['patch_len']}")
print(f"  stride: v0.3={best_v03['params']['stride']}, v0.3.7={best_v037['params']['stride']}")
print(f"  Overlap: v0.3={(1-best_v03['params']['stride']/best_v03['params']['patch_len'])*100:.1f}%, " +
      f"v0.3.7={(1-best_v037['params']['stride']/best_v037['params']['patch_len'])*100:.1f}%")
print()

print("=" * 80)
print("Possible Causes:")
print("=" * 80)
print("1. Dataset difference: Check if train/test split or data preprocessing changed")
print("2. CUDA deterministic mode: May reduce model capacity vs. non-deterministic v0.3")
print("3. Random seed effect: seed=42 may lead to suboptimal initialization")
print("4. Architecture constraint: L_p > s constraint may be too restrictive")
print("5. Baseline model: Check if v0.3.7 baseline performance matches v0.3")
print("6. Early stopping patience: May need adjustment for deterministic mode")
print()
