"""Analyze v0.2.1C5 results (quick summary script)"""
import json
import sys
import os

archive_path = "ptst_dgm_v021C5/results/ptst_archive_v021C5.jsonl"
pareto_path = "ptst_dgm_v021C5/results/ptst_archive_v021C5_pareto.jsonl"

# Load all entries
with open(archive_path) as f:
    entries = [json.loads(line) for line in f]

# Load pareto entries if available
pareto_entries = []
if os.path.exists(pareto_path):
    with open(pareto_path) as f:
        pareto_entries = [json.loads(line) for line in f]
else:
    print("[Note] Pareto archive not found - computing from full archive...")
    # Compute pareto frontier manually if needed
    pareto_entries = entries  # Simplified - use all for now

# Find best solution
best = max(entries, key=lambda x: x['macro_f1'])

print("=" * 70)
print("v0.2.1C5 Results Summary (579 iterations)")
print("=" * 70)
print()
print(f"Total archive entries: {len(entries)}")
if os.path.exists(pareto_path):
    print(f"Pareto solutions: {len(pareto_entries)} ({100*len(pareto_entries)/len(entries):.1f}%)")
else:
    print(f"[Note] Pareto archive not generated yet (experiment interrupted)")
print()
print("BEST SOLUTION (highest macro F1):")
print(f"  ID: {best['id'][:8]}...")
print(f"  macro F1: {best['macro_f1']:.4f}")
print(f"  Focal Loss: α={best['focal_alpha']:.3f} γ={best['focal_gamma']:.3f}")
print(f"  Class Weights: w_normal={best['w_normal']:.3f} w_anomal={best['w_anomal']:.3f}")
print(f"  Weight Sum: {best['w_normal']+best['w_anomal']:.2f} (constraint: 5.88)")
print(f"  Weight Ratio: {best['w_anomal']/best['w_normal']:.2f} (anomal/normal)")
print()
print("  Per-Horizon F1:")
print(f"    30d: {best['objectives']['f1_30d']:.4f}")
print(f"    60d: {best['objectives']['f1_60d']:.4f}")
print(f"    90d: {best['objectives']['f1_90d']:.4f}")
print()
print("  Per-Horizon AUC:")
print(f"    30d: {best['objectives']['auc_30d']:.4f}")
print(f"    60d: {best['objectives']['auc_60d']:.4f}")
print(f"    90d: {best['objectives']['auc_90d']:.4f}")
print()

# Pareto frontier analysis
if len(pareto_entries) > 0 and os.path.exists(pareto_path):
    pareto_f1_range = [min(e['macro_f1'] for e in pareto_entries), 
                       max(e['macro_f1'] for e in pareto_entries)]
    high_f1_pareto = [e for e in pareto_entries if e['macro_f1'] > 0.63]

    print("PARETO FRONTIER:")
    print(f"  F1 range: [{pareto_f1_range[0]:.4f}, {pareto_f1_range[1]:.4f}]")
    print(f"  Solutions with F1 > 0.63: {len(high_f1_pareto)}")
    print()

# Archive statistics
f1_distribution = {
    'F1 >= 0.63': len([e for e in entries if e['macro_f1'] >= 0.63]),
    '0.60 <= F1 < 0.63': len([e for e in entries if 0.60 <= e['macro_f1'] < 0.63]),
    '0.55 <= F1 < 0.60': len([e for e in entries if 0.55 <= e['macro_f1'] < 0.60]),
    'F1 < 0.55': len([e for e in entries if e['macro_f1'] < 0.55])
}
print("F1 DISTRIBUTION (archive entries):")
for range_name, count in f1_distribution.items():
    print(f"  {range_name}: {count} ({100*count/len(entries):.1f}%)")
print()

# Comparison with v0.2.1C
print("COMPARISON WITH v0.2.1C:")
print(f"  v0.2.1C best: macro F1 = 0.6372")
print(f"  v0.2.1C5 best: macro F1 = {best['macro_f1']:.4f}")
print(f"  Difference: {best['macro_f1'] - 0.6372:.4f} ({100*(best['macro_f1']/0.6372-1):.1f}%)")
print(f"  Target (0.655): {'✓ ACHIEVED' if best['macro_f1'] >= 0.655 else '✗ NOT REACHED'}")
print()
print("=" * 70)
