"""
Compare v0.3.6 vs v0.3.12.2 Results
===================================

Generate 5 comparison visualizations:
1. F1 Score Distribution Comparison
2. Patch Length vs Stride 2D Scatter
3. Top 10 Solutions Comparison
4. Overlap Distribution Comparison
5. Multi-Objective Trade-off (AUC vs FPR)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

# Set Japanese font for matplotlib
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'MS Gothic']
plt.rcParams['axes.unicode_minus'] = False

# Paths
v0_3_6_path = Path(r"I:\ACT2025.5.26-2030\MVP\patchtst_pareto_dgm\0_LogBAK\v0-3-6_repro\ptst_dgm\results\ptst_archive_v0.3.6.jsonl")
v0_3_12_2_path = Path(r"I:\ACT2025.5.26-2030\MVP\patchtst_pareto_dgm\ptst_dgm\results\ptst_archive_v0.3.12.2_1000iters.jsonl")
output_dir = Path(r"I:\ACT2025.5.26-2030\MVP\patchtst_pareto_dgm\ptst_dgm\results\comparison_v0.3.6_vs_v0.3.12.2")
output_dir.mkdir(exist_ok=True)


def load_archive(path: Path) -> List[Dict]:
    """Load JSONL archive file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def filter_valid(data: List[Dict]) -> List[Dict]:
    """Filter entries with valid macro_f1."""
    return [d for d in data if d.get('macro_f1') is not None]


def calculate_overlap(patch_len: int, stride: int) -> float:
    """Calculate overlap ratio (%)."""
    return (1 - stride / patch_len) * 100


def calculate_mean_fpr(objectives: Dict) -> float:
    """Calculate mean FPR across 3 horizons."""
    fprs = [objectives.get(f'fpr_{h}', 0) for h in ['30d', '60d', '90d']]
    return np.mean(fprs)


def calculate_mean_auc(objectives: Dict) -> float:
    """Calculate mean AUC across 3 horizons."""
    aucs = [objectives.get(f'auc_{h}', 0) for h in ['30d', '60d', '90d']]
    return np.mean(aucs)


# Load data
print("Loading archives...")
v0_3_6_raw = load_archive(v0_3_6_path)
v0_3_12_2_raw = load_archive(v0_3_12_2_path)

v0_3_6 = filter_valid(v0_3_6_raw)
v0_3_12_2 = filter_valid(v0_3_12_2_raw)

print(f"v0.3.6: {len(v0_3_6)} valid trials (from {len(v0_3_6_raw)} total)")
print(f"v0.3.12.2: {len(v0_3_12_2)} valid trials (from {len(v0_3_12_2_raw)} total)")

# Extract metrics
v0_3_6_f1 = [d['macro_f1'] for d in v0_3_6]
v0_3_12_2_f1 = [d['macro_f1'] for d in v0_3_12_2]

v0_3_6_sorted = sorted(v0_3_6, key=lambda x: x['macro_f1'], reverse=True)
v0_3_12_2_sorted = sorted(v0_3_12_2, key=lambda x: x['macro_f1'], reverse=True)

print(f"\nv0.3.6 Best F1: {v0_3_6_sorted[0]['macro_f1']:.4f}")
print(f"v0.3.12.2 Best F1: {v0_3_12_2_sorted[0]['macro_f1']:.4f}")

# ============================================================================
# Figure 1: F1 Score Distribution Comparison
# ============================================================================
print("\n[1/5] Creating F1 Distribution Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('F1スコア分布比較: v0.3.6 vs v0.3.12.2', fontsize=16, fontweight='bold')

# v0.3.6
axes[0].hist(v0_3_6_f1, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0].axvline(v0_3_6_sorted[0]['macro_f1'], color='red', linestyle='--', linewidth=2, label=f"Best: {v0_3_6_sorted[0]['macro_f1']:.4f}")
axes[0].axvline(np.median(v0_3_6_f1), color='orange', linestyle='--', linewidth=2, label=f"Median: {np.median(v0_3_6_f1):.4f}")
axes[0].set_title(f'v0.3.6 (n={len(v0_3_6)}, seed=42)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Macro F1 Score', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# v0.3.12.2
axes[1].hist(v0_3_12_2_f1, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
axes[1].axvline(v0_3_12_2_sorted[0]['macro_f1'], color='red', linestyle='--', linewidth=2, label=f"Best: {v0_3_12_2_sorted[0]['macro_f1']:.4f}")
axes[1].axvline(np.median(v0_3_12_2_f1), color='orange', linestyle='--', linewidth=2, label=f"Median: {np.median(v0_3_12_2_f1):.4f}")
axes[1].set_title(f'v0.3.12.2 (n={len(v0_3_12_2)}, seed=None)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Macro F1 Score', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '1_f1_distribution_comparison.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '1_f1_distribution_comparison.png'}")
plt.close()

# ============================================================================
# Figure 2: Patch Length vs Stride 2D Scatter
# ============================================================================
print("\n[2/5] Creating 2D Architecture Space Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Architecture空間: Patch Length vs Stride', fontsize=16, fontweight='bold')

# v0.3.6
patches_v6 = [d['patch_len'] for d in v0_3_6]
strides_v6 = [d['stride'] for d in v0_3_6]
f1_v6 = [d['macro_f1'] for d in v0_3_6]

scatter1 = axes[0].scatter(patches_v6, strides_v6, c=f1_v6, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
best_v6 = v0_3_6_sorted[0]
axes[0].scatter([best_v6['patch_len']], [best_v6['stride']], color='red', s=300, marker='*', edgecolors='black', linewidths=2, label=f"Best (F1={best_v6['macro_f1']:.4f})")
axes[0].plot([0, 40], [0, 40], 'r--', alpha=0.5, linewidth=1, label='stride=patch (no overlap)')
axes[0].set_title(f'v0.3.6 (Best: patch={best_v6["patch_len"]}, stride={best_v6["stride"]})', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Patch Length', fontsize=11)
axes[0].set_ylabel('Stride', fontsize=11)
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xlim(10, 45)
axes[0].set_ylim(0, 30)
cbar1 = plt.colorbar(scatter1, ax=axes[0])
cbar1.set_label('Macro F1', fontsize=10)

# v0.3.12.2
patches_v12 = [d['patch_len'] for d in v0_3_12_2]
strides_v12 = [d['stride'] for d in v0_3_12_2]
f1_v12 = [d['macro_f1'] for d in v0_3_12_2]

scatter2 = axes[1].scatter(patches_v12, strides_v12, c=f1_v12, cmap='plasma', s=100, alpha=0.6, edgecolors='black')
best_v12 = v0_3_12_2_sorted[0]
axes[1].scatter([best_v12['patch_len']], [best_v12['stride']], color='red', s=300, marker='*', edgecolors='black', linewidths=2, label=f"Best (F1={best_v12['macro_f1']:.4f})")
axes[1].plot([0, 40], [0, 40], 'r--', alpha=0.5, linewidth=1, label='stride=patch (no overlap)')
axes[1].set_title(f'v0.3.12.2 (Best: patch={best_v12["patch_len"]}, stride={best_v12["stride"]})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Patch Length', fontsize=11)
axes[1].set_ylabel('Stride', fontsize=11)
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xlim(10, 45)
axes[1].set_ylim(0, 30)
cbar2 = plt.colorbar(scatter2, ax=axes[1])
cbar2.set_label('Macro F1', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / '2_architecture_space_2d.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '2_architecture_space_2d.png'}")
plt.close()

# ============================================================================
# Figure 3: Top 10 Solutions Comparison
# ============================================================================
print("\n[3/5] Creating Top 10 Solutions Comparison...")

top10_v6 = v0_3_6_sorted[:10]
top10_v12 = v0_3_12_2_sorted[:10]

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(10)
width = 0.35

f1_top10_v6 = [d['macro_f1'] for d in top10_v6]
f1_top10_v12 = [d['macro_f1'] for d in top10_v12]

bars1 = ax.bar(x - width/2, f1_top10_v6, width, label='v0.3.6 (seed=42)', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, f1_top10_v12, width, label='v0.3.12.2 (seed=None)', color='lightcoral', edgecolor='black')

# Add value labels
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.01, f'{height1:.4f}', ha='center', va='bottom', fontsize=8)
    ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.01, f'{height2:.4f}', ha='center', va='bottom', fontsize=8)

ax.set_title('Top 10 Solutions比較: v0.3.6 vs v0.3.12.2', fontsize=16, fontweight='bold')
ax.set_xlabel('Rank', fontsize=12)
ax.set_ylabel('Macro F1 Score', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f'#{i+1}' for i in range(10)])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(max(f1_top10_v6), max(f1_top10_v12)) * 1.15)

# Add annotations for best parameters
best_v6_text = f"v0.3.6 Best: patch={best_v6['patch_len']}, stride={best_v6['stride']}, overlap={calculate_overlap(best_v6['patch_len'], best_v6['stride']):.1f}%"
best_v12_text = f"v0.3.12.2 Best: patch={best_v12['patch_len']}, stride={best_v12['stride']}, overlap={calculate_overlap(best_v12['patch_len'], best_v12['stride']):.1f}%"
ax.text(0.5, 0.95, best_v6_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='skyblue', alpha=0.5))
ax.text(0.5, 0.90, best_v12_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / '3_top10_comparison.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '3_top10_comparison.png'}")
plt.close()

# ============================================================================
# Figure 4: Overlap Distribution Comparison
# ============================================================================
print("\n[4/5] Creating Overlap Distribution Comparison...")

overlaps_v6 = [calculate_overlap(d['patch_len'], d['stride']) for d in v0_3_6]
overlaps_v12 = [calculate_overlap(d['patch_len'], d['stride']) for d in v0_3_12_2]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Overlap分布比較: v0.3.6 vs v0.3.12.2', fontsize=16, fontweight='bold')

# v0.3.6
axes[0].hist(overlaps_v6, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0].axvline(61.8, color='gold', linestyle='--', linewidth=2, label='Golden Ratio (61.8%)')
overlap_v6_best = calculate_overlap(best_v6['patch_len'], best_v6['stride'])
axes[0].axvline(overlap_v6_best, color='red', linestyle='--', linewidth=2, label=f'Best: {overlap_v6_best:.1f}%')
axes[0].set_title(f'v0.3.6 (median: {np.median(overlaps_v6):.1f}%)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Overlap (%)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_xlim(-50, 100)

# v0.3.12.2
axes[1].hist(overlaps_v12, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
axes[1].axvline(61.8, color='gold', linestyle='--', linewidth=2, label='Golden Ratio (61.8%)')
overlap_v12_best = calculate_overlap(best_v12['patch_len'], best_v12['stride'])
axes[1].axvline(overlap_v12_best, color='red', linestyle='--', linewidth=2, label=f'Best: {overlap_v12_best:.1f}%')
axes[1].set_title(f'v0.3.12.2 (median: {np.median(overlaps_v12):.1f}%)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Overlap (%)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_xlim(-50, 100)

plt.tight_layout()
plt.savefig(output_dir / '4_overlap_distribution.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '4_overlap_distribution.png'}")
plt.close()

# ============================================================================
# Figure 5: Multi-Objective Trade-off (AUC vs FPR)
# ============================================================================
print("\n[5/5] Creating Multi-Objective Trade-off Visualization...")

# Calculate mean AUC and FPR
auc_v6 = [calculate_mean_auc(d['objectives']) for d in v0_3_6]
fpr_v6 = [calculate_mean_fpr(d['objectives']) for d in v0_3_6]
f1_v6_colored = [d['macro_f1'] for d in v0_3_6]

auc_v12 = [calculate_mean_auc(d['objectives']) for d in v0_3_12_2]
fpr_v12 = [calculate_mean_fpr(d['objectives']) for d in v0_3_12_2]
f1_v12_colored = [d['macro_f1'] for d in v0_3_12_2]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Multi-Objective Trade-off: Mean AUC vs Mean FPR', fontsize=16, fontweight='bold')

# v0.3.6
scatter1 = axes[0].scatter(fpr_v6, auc_v6, c=f1_v6_colored, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
best_auc_v6 = calculate_mean_auc(best_v6['objectives'])
best_fpr_v6 = calculate_mean_fpr(best_v6['objectives'])
axes[0].scatter([best_fpr_v6], [best_auc_v6], color='red', s=300, marker='*', edgecolors='black', linewidths=2, label=f"Best (F1={best_v6['macro_f1']:.4f})")
axes[0].set_title(f'v0.3.6 (n={len(v0_3_6)})', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Mean FPR (↓ better)', fontsize=11)
axes[0].set_ylabel('Mean AUC (↑ better)', fontsize=11)
axes[0].legend()
axes[0].grid(alpha=0.3)
cbar1 = plt.colorbar(scatter1, ax=axes[0])
cbar1.set_label('Macro F1', fontsize=10)

# v0.3.12.2
scatter2 = axes[1].scatter(fpr_v12, auc_v12, c=f1_v12_colored, cmap='plasma', s=100, alpha=0.6, edgecolors='black')
best_auc_v12 = calculate_mean_auc(best_v12['objectives'])
best_fpr_v12 = calculate_mean_fpr(best_v12['objectives'])
axes[1].scatter([best_fpr_v12], [best_auc_v12], color='red', s=300, marker='*', edgecolors='black', linewidths=2, label=f"Best (F1={best_v12['macro_f1']:.4f})")
axes[1].set_title(f'v0.3.12.2 (n={len(v0_3_12_2)})', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Mean FPR (↓ better)', fontsize=11)
axes[1].set_ylabel('Mean AUC (↑ better)', fontsize=11)
axes[1].legend()
axes[1].grid(alpha=0.3)
cbar2 = plt.colorbar(scatter2, ax=axes[1])
cbar2.set_label('Macro F1', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / '5_auc_fpr_tradeoff.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '5_auc_fpr_tradeoff.png'}")
plt.close()

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("📊 Summary Statistics")
print("="*70)

print("\n【v0.3.6】")
print(f"  Total Trials: {len(v0_3_6)}")
print(f"  Best F1: {v0_3_6_sorted[0]['macro_f1']:.4f} (patch={best_v6['patch_len']}, stride={best_v6['stride']}, overlap={overlap_v6_best:.1f}%)")
print(f"  Median F1: {np.median(v0_3_6_f1):.4f}")
print(f"  Mean F1: {np.mean(v0_3_6_f1):.4f}")
print(f"  Std F1: {np.std(v0_3_6_f1):.4f}")

print("\n【v0.3.12.2】")
print(f"  Total Trials: {len(v0_3_12_2)}")
print(f"  Best F1: {v0_3_12_2_sorted[0]['macro_f1']:.4f} (patch={best_v12['patch_len']}, stride={best_v12['stride']}, overlap={overlap_v12_best:.1f}%)")
print(f"  Median F1: {np.median(v0_3_12_2_f1):.4f}")
print(f"  Mean F1: {np.mean(v0_3_12_2_f1):.4f}")
print(f"  Std F1: {np.std(v0_3_12_2_f1):.4f}")

print("\n【Comparison】")
improvement = ((v0_3_6_sorted[0]['macro_f1'] - v0_3_12_2_sorted[0]['macro_f1']) / v0_3_12_2_sorted[0]['macro_f1']) * 100
print(f"  v0.3.6 vs v0.3.12.2 Best F1: {improvement:+.2f}% (v0.3.6が{abs(improvement):.2f}%優位)")
print(f"  Overlap (Best): v0.3.6={overlap_v6_best:.1f}%, v0.3.12.2={overlap_v12_best:.1f}%")
print(f"  Golden Ratio Deviation: v0.3.6={abs(overlap_v6_best - 61.8):.1f}%, v0.3.12.2={abs(overlap_v12_best - 61.8):.1f}%")

print("\n✅ All visualizations saved to:")
print(f"   {output_dir}")
print("="*70)
