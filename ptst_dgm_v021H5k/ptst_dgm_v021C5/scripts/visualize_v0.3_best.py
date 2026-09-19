import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read v0.3 Pareto solutions
solutions = []
with open('ptst_dgm/results/ptst_archive_v0.3_pareto.jsonl', 'r') as f:
    for line in f:
        solutions.append(json.loads(line))

# Extract data
df = pd.DataFrame([
    {
        'trial': s['trial_number'],
        'patch_len': s['params']['patch_len'],
        'stride': s['params']['stride'],
        'overlap_ratio': s['params']['stride'] / s['params']['patch_len'],
        'macro_f1': s['macro_f1'],
        'mean_fpr': s['mean_fpr'],
        'f1_30d': s['objectives']['f1_30d'],
        'f1_60d': s['objectives']['f1_60d'],
        'f1_90d': s['objectives']['f1_90d'],
        'auc_30d': s['objectives']['auc_30d'],
        'auc_60d': s['objectives']['auc_60d'],
        'auc_90d': s['objectives']['auc_90d'],
        'fpr_30d': s['objectives']['fpr_30d'],
        'fpr_60d': s['objectives']['fpr_60d'],
        'fpr_90d': s['objectives']['fpr_90d'],
    }
    for s in solutions
])

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('v0.3 Best Solution Analysis (Trial #1: F1=0.7726)', fontsize=16, fontweight='bold')

# 1. Pareto frontier: F1 vs FPR
ax = axes[0, 0]
colors = ['red' if trial == 1 else 'blue' for trial in df['trial']]
sizes = [300 if trial == 1 else 100 for trial in df['trial']]
ax.scatter(df['mean_fpr'], df['macro_f1'], c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=2)
for idx, row in df.iterrows():
    label = f"#{row['trial']}\npatch={row['patch_len']}\nstride={row['stride']}"
    ax.annotate(label, (row['mean_fpr'], row['macro_f1']), 
                fontsize=9, ha='center', va='bottom', fontweight='bold' if row['trial'] == 1 else 'normal')
ax.set_xlabel('Mean FPR (minimize)', fontsize=12)
ax.set_ylabel('Macro F1 (maximize)', fontsize=12)
ax.set_title('Pareto Frontier: F1 vs FPR Trade-off', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(0.7726, color='red', linestyle='--', alpha=0.3, label='Best F1=0.7726')
ax.legend()

# 2. Architecture parameters
ax = axes[0, 1]
x = np.arange(len(df))
width = 0.35
bars1 = ax.bar(x - width/2, df['patch_len'], width, label='Patch Length', alpha=0.7, color='skyblue')
bars2 = ax.bar(x + width/2, df['stride'], width, label='Stride', alpha=0.7, color='orange')
ax.set_xlabel('Trial', fontsize=12)
ax.set_ylabel('Value (days)', fontsize=12)
ax.set_title('Architecture Parameters', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"#{t}" for t in df['trial']])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
# Highlight best trial
bars1[1].set_color('red')
bars1[1].set_alpha(1.0)
bars2[1].set_color('darkred')
bars2[1].set_alpha(1.0)

# 3. Overlap ratio
ax = axes[0, 2]
bars = ax.bar(df['trial'], df['overlap_ratio'], color=['red' if t == 1 else 'gray' for t in df['trial']], alpha=0.7)
ax.set_xlabel('Trial', fontsize=12)
ax.set_ylabel('Overlap Ratio (stride / patch_len)', fontsize=12)
ax.set_title('Patch Overlap Ratio', fontsize=13, fontweight='bold')
ax.set_xticks(df['trial'])
ax.set_xticklabels([f"#{t}" for t in df['trial']])
ax.axhline(0.618, color='gold', linestyle='--', linewidth=2, label='Golden Ratio (0.618)')
ax.grid(True, alpha=0.3, axis='y')
ax.legend()
for i, (t, v) in enumerate(zip(df['trial'], df['overlap_ratio'])):
    ax.text(t, v + 0.02, f"{v:.3f}", ha='center', fontsize=10, fontweight='bold' if t == 1 else 'normal')

# 4. F1 by horizon
ax = axes[1, 0]
horizons = ['30d', '60d', '90d']
trial_1_f1 = df[df['trial'] == 1][['f1_30d', 'f1_60d', 'f1_90d']].values[0]
trial_0_f1 = df[df['trial'] == 0][['f1_30d', 'f1_60d', 'f1_90d']].values[0]
trial_2_f1 = df[df['trial'] == 2][['f1_30d', 'f1_60d', 'f1_90d']].values[0]

x = np.arange(len(horizons))
width = 0.25
ax.bar(x - width, trial_0_f1, width, label='Trial #0 (patch=17)', alpha=0.6, color='lightblue')
ax.bar(x, trial_1_f1, width, label='Trial #1 (patch=26) ⭐', alpha=0.9, color='red')
ax.bar(x + width, trial_2_f1, width, label='Trial #2 (patch=11)', alpha=0.6, color='lightgray')
ax.set_xlabel('Prediction Horizon', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('F1 Score by Horizon (Trial #1 excels at 90d)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(horizons)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.6, 0.85])

# 5. AUC by horizon
ax = axes[1, 1]
trial_1_auc = df[df['trial'] == 1][['auc_30d', 'auc_60d', 'auc_90d']].values[0]
trial_0_auc = df[df['trial'] == 0][['auc_30d', 'auc_60d', 'auc_90d']].values[0]
trial_2_auc = df[df['trial'] == 2][['auc_30d', 'auc_60d', 'auc_90d']].values[0]

ax.bar(x - width, trial_0_auc, width, label='Trial #0', alpha=0.6, color='lightblue')
ax.bar(x, trial_1_auc, width, label='Trial #1 ⭐', alpha=0.9, color='red')
ax.bar(x + width, trial_2_auc, width, label='Trial #2', alpha=0.6, color='lightgray')
ax.set_xlabel('Prediction Horizon', fontsize=12)
ax.set_ylabel('AUC Score', fontsize=12)
ax.set_title('AUC by Horizon (Trial #1 highest at 90d)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(horizons)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.80, 0.96])

# 6. FPR by horizon
ax = axes[1, 2]
trial_1_fpr = df[df['trial'] == 1][['fpr_30d', 'fpr_60d', 'fpr_90d']].values[0]
trial_0_fpr = df[df['trial'] == 0][['fpr_30d', 'fpr_60d', 'fpr_90d']].values[0]
trial_2_fpr = df[df['trial'] == 2][['fpr_30d', 'fpr_60d', 'fpr_90d']].values[0]

ax.bar(x - width, trial_0_fpr, width, label='Trial #0', alpha=0.6, color='lightblue')
ax.bar(x, trial_1_fpr, width, label='Trial #1 ⭐', alpha=0.9, color='red')
ax.bar(x + width, trial_2_fpr, width, label='Trial #2 (lowest)', alpha=0.6, color='lightgray')
ax.set_xlabel('Prediction Horizon', fontsize=12)
ax.set_ylabel('False Positive Rate', fontsize=12)
ax.set_title('FPR by Horizon (Trial #1 excellent at 30d)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(horizons)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ptst_dgm/results/v0.3_best_solution_analysis.png', dpi=300, bbox_inches='tight')
print(f"[Saved] ptst_dgm/results/v0.3_best_solution_analysis.png")

# Summary statistics
print("\n" + "="*60)
print("v0.3 Best Solution Summary (Trial #1)")
print("="*60)
best = df[df['trial'] == 1].iloc[0]
print(f"Architecture: patch_len={best['patch_len']}, stride={best['stride']}")
print(f"Overlap Ratio: {best['overlap_ratio']:.3f} (61.5% overlap)")
print(f"Macro F1: {best['macro_f1']:.4f}")
print(f"Mean FPR: {best['mean_fpr']:.4f}")
print(f"\nHorizon Performance:")
print(f"  30d: F1={best['f1_30d']:.4f}, AUC={best['auc_30d']:.4f}, FPR={best['fpr_30d']:.4f}")
print(f"  60d: F1={best['f1_60d']:.4f}, AUC={best['auc_60d']:.4f}, FPR={best['fpr_60d']:.4f}")
print(f"  90d: F1={best['f1_90d']:.4f}, AUC={best['auc_90d']:.4f}, FPR={best['fpr_90d']:.4f} ⭐ BEST")
print("="*60)
