"""
visualize_seed_search.py
Visualize v0.3.5 seed search results.

Usage:
    python ptst_dgm/scripts/visualize_seed_search.py
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = WORKSPACE_ROOT / "ptst_dgm" / "results" / "seed_search_v0.3.5.jsonl"
OUTPUT = WORKSPACE_ROOT / "ptst_dgm" / "results" / "seed_search_v0.3.5_analysis.png"

V03_BEST_F1 = 0.7726
V03_BEST_30D = 0.701
V03_BEST_60D = 0.717
V03_BEST_90D = 0.810


def load_results():
    """Load seed search results from JSONL."""
    results = []
    with open(ARCHIVE, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry['status'] == 'success':
                results.append(entry)
    return results


def main():
    results = load_results()
    
    if not results:
        print("[Error] No results found in archive")
        return
    
    # Extract data
    seeds = [r['seed'] for r in results]
    macro_f1 = [r['macro_f1'] for r in results]
    f1_30d = [r['f1_30d'] for r in results]
    f1_60d = [r['f1_60d'] for r in results]
    f1_90d = [r['f1_90d'] for r in results]
    auc_30d = [r['auc_30d'] for r in results]
    fpr_30d = [r['fpr_30d'] for r in results]
    
    # Statistics
    mean_f1 = np.mean(macro_f1)
    std_f1 = np.std(macro_f1)
    min_f1 = np.min(macro_f1)
    max_f1 = np.max(macro_f1)
    best_seed = seeds[np.argmax(macro_f1)]
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Macro F1 by Seed
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.bar(seeds, macro_f1, color='steelblue', alpha=0.7, edgecolor='black')
    # Highlight best seed
    best_idx = np.argmax(macro_f1)
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgoldenrod')
    bars[best_idx].set_linewidth(2)
    
    ax1.axhline(V03_BEST_F1, color='red', linestyle='--', linewidth=2, label=f'v0.3 best: {V03_BEST_F1:.4f}')
    ax1.axhline(mean_f1, color='green', linestyle=':', linewidth=1.5, label=f'Mean: {mean_f1:.4f}')
    ax1.set_xlabel('Seed', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Macro F1', fontsize=12, fontweight='bold')
    ax1.set_title('Macro F1 by Seed', fontsize=14, fontweight='bold')
    ax1.set_xticks(seeds)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add annotation for best seed
    ax1.annotate(f'Best: {max_f1:.4f}\n(seed {best_seed})',
                xy=(best_seed, max_f1),
                xytext=(best_seed, max_f1 + 0.05),
                fontsize=10, fontweight='bold', color='darkgoldenrod',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='darkgoldenrod', lw=1.5))
    
    # 2. F1 Distribution (Histogram)
    ax2 = plt.subplot(2, 3, 2)
    n, bins, patches = ax2.hist(macro_f1, bins=8, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(V03_BEST_F1, color='red', linestyle='--', linewidth=2, label=f'v0.3: {V03_BEST_F1:.4f}')
    ax2.axvline(mean_f1, color='green', linestyle=':', linewidth=1.5, label=f'Mean: {mean_f1:.4f}')
    ax2.set_xlabel('Macro F1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    stats_text = f'n={len(results)}\nμ={mean_f1:.4f}\nσ={std_f1:.4f}\nRange: [{min_f1:.4f}, {max_f1:.4f}]'
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. F1 by Horizon
    ax3 = plt.subplot(2, 3, 3)
    x = np.arange(len(seeds))
    width = 0.25
    
    ax3.bar(x - width, f1_30d, width, label='30d', color='#FF6B6B', alpha=0.7, edgecolor='black')
    ax3.bar(x, f1_60d, width, label='60d', color='#4ECDC4', alpha=0.7, edgecolor='black')
    ax3.bar(x + width, f1_90d, width, label='90d', color='#95E1D3', alpha=0.7, edgecolor='black')
    
    ax3.axhline(V03_BEST_30D, color='#FF6B6B', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(V03_BEST_60D, color='#4ECDC4', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(V03_BEST_90D, color='#95E1D3', linestyle='--', linewidth=1, alpha=0.5)
    
    ax3.set_xlabel('Seed', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax3.set_title('F1 by Horizon', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(seeds)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. AUC vs F1 Scatter
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(auc_30d, macro_f1, c=seeds, cmap='viridis', s=100, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('30d AUC', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Macro F1', fontsize=12, fontweight='bold')
    ax4.set_title('AUC vs F1 (30d)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Seed', fontsize=10, fontweight='bold')
    
    # Add v0.3 reference point
    ax4.scatter([0.915], [V03_BEST_F1], marker='*', s=300, color='red', 
               edgecolor='darkred', linewidth=2, label='v0.3 best', zorder=5)
    ax4.legend(fontsize=10)
    
    # 5. FPR vs F1 Scatter
    ax5 = plt.subplot(2, 3, 5)
    scatter2 = ax5.scatter(fpr_30d, macro_f1, c=seeds, cmap='viridis', s=100, alpha=0.7, edgecolor='black')
    ax5.set_xlabel('30d FPR', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Macro F1', fontsize=12, fontweight='bold')
    ax5.set_title('FPR vs F1 (30d)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar2 = plt.colorbar(scatter2, ax=ax5)
    cbar2.set_label('Seed', fontsize=10, fontweight='bold')
    
    # Add v0.3 reference point (estimate FPR from data)
    v03_fpr = 0.033  # From v0.3 analysis
    ax5.scatter([v03_fpr], [V03_BEST_F1], marker='*', s=300, color='red',
               edgecolor='darkred', linewidth=2, label='v0.3 best', zorder=5)
    ax5.legend(fontsize=10)
    
    # 6. Summary Statistics Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'Value'],
        ['━━━━━━━━━━━━━━━━', '━━━━━━━━━━'],
        ['Seeds Tested', f'{len(results)}'],
        ['Best Seed', f'{best_seed}'],
        ['Best F1', f'{max_f1:.4f}'],
        ['Mean F1', f'{mean_f1:.4f}'],
        ['Std F1', f'{std_f1:.4f}'],
        ['Min F1', f'{min_f1:.4f}'],
        ['━━━━━━━━━━━━━━━━', '━━━━━━━━━━'],
        ['v0.3 Best', f'{V03_BEST_F1:.4f}'],
        ['Gap', f'{max_f1 - V03_BEST_F1:.4f} ({(max_f1/V03_BEST_F1 - 1)*100:.1f}%)'],
        ['━━━━━━━━━━━━━━━━', '━━━━━━━━━━'],
        ['Status', '❌ Target NOT reached'],
    ]
    
    table = ax6.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor('#4ECDC4')
        cell.set_text_props(weight='bold', color='white')
    
    # Style status row
    cell = table[(12, 0)]
    cell.set_text_props(weight='bold')
    cell = table[(12, 1)]
    cell.set_text_props(weight='bold', color='red')
    
    ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle(f'v0.3.5 Seed Search Results (n={len(results)} seeds)',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {OUTPUT.relative_to(WORKSPACE_ROOT)}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Seed Search Summary")
    print(f"{'='*60}")
    print(f"Seeds tested: {len(results)}")
    print(f"Best seed: {best_seed} (F1={max_f1:.4f})")
    print(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Range: [{min_f1:.4f}, {max_f1:.4f}]")
    print(f"\nv0.3 best: {V03_BEST_F1:.4f}")
    print(f"Gap: {max_f1 - V03_BEST_F1:.4f} ({(max_f1/V03_BEST_F1 - 1)*100:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
