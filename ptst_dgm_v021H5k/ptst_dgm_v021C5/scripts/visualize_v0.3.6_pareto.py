"""
visualize_v0.3.6_pareto.py
Visualize v0.3.6 Architecture Optimization Pareto Frontier

Creates 5 visualizations:
1. Macro F1 vs Mean FPR (main trade-off)
2. Per-horizon F1 comparison (3 horizons)
3. Architecture space (patch_len vs stride)
4. Overlap ratio vs F1
5. All 15 objectives heatmap
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

HORIZONS = ["30d", "60d", "90d"]
METRICS = ["auc", "precision", "recall", "f1", "fpr"]

def load_pareto_frontier(filepath):
    """Load Pareto frontier solutions from JSONL."""
    solutions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                solutions.append(json.loads(line.strip()))
    return solutions

def plot_macro_f1_vs_fpr(solutions, output_dir):
    """Plot 1: Macro F1 vs Mean FPR - Main trade-off"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    macro_f1s = [s['macro_f1'] for s in solutions]
    mean_fprs = [s['mean_fpr'] for s in solutions]
    trials = [s['trial_number'] for s in solutions]
    
    # Plot all solutions
    scatter = ax.scatter(mean_fprs, macro_f1s, s=150, alpha=0.7,
                        c=range(len(solutions)), cmap='viridis',
                        edgecolor='black', linewidth=1.5, zorder=3)
    
    # Highlight best F1
    best_idx = np.argmax(macro_f1s)
    ax.scatter(mean_fprs[best_idx], macro_f1s[best_idx], s=500, 
              marker='*', c='red', edgecolor='black', linewidth=2, 
              zorder=10, label=f'Best F1 (Trial #{trials[best_idx]})')
    
    # Annotate top 5
    top5_idx = np.argsort(macro_f1s)[-5:]
    for i in top5_idx:
        ax.annotate(f'#{trials[i]}', (mean_fprs[i], macro_f1s[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=10,
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                   facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Mean FPR (False Positive Rate)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Macro F1 Score', fontsize=13, fontweight='bold')
    ax.set_title('v0.3.6 Pareto Frontier: F1 vs FPR Trade-off\n(Seed 42, Architecture Optimization)',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', fontsize=11)
    
    cbar = plt.colorbar(scatter, ax=ax, label='Trial Order')
    cbar.set_label('Trial Order', fontsize=11)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'v0.3.6_pareto_f1_vs_fpr.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_per_horizon_f1(solutions, output_dir):
    """Plot 2: Per-horizon F1 comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, horizon in enumerate(HORIZONS):
        ax = axes[idx]
        f1_key = f'f1_{horizon}'
        fpr_key = f'fpr_{horizon}'
        
        f1s = [s['objectives'][f1_key] for s in solutions]
        fprs = [s['objectives'][fpr_key] for s in solutions]
        trials = [s['trial_number'] for s in solutions]
        
        scatter = ax.scatter(fprs, f1s, s=120, alpha=0.7,
                           c=range(len(solutions)), cmap='plasma',
                           edgecolor='black', linewidth=1.2)
        
        # Highlight best F1 for this horizon
        best_idx = np.argmax(f1s)
        ax.scatter(fprs[best_idx], f1s[best_idx], s=400, marker='*',
                  c='red', edgecolor='black', linewidth=2, zorder=10,
                  label=f'Best F1={f1s[best_idx]:.4f}')
        
        ax.set_xlabel(f'FPR {horizon}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'F1 {horizon}', fontsize=12, fontweight='bold')
        ax.set_title(f'{horizon} Horizon', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower left', fontsize=10)
    
    fig.suptitle('v0.3.6 Per-Horizon Performance (Seed 42)',
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = Path(output_dir) / 'v0.3.6_pareto_per_horizon.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_architecture_space(solutions, output_dir):
    """Plot 3: Architecture parameter space (patch_len vs stride)"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Check if solutions have 'params' or direct keys
    if 'params' in solutions[0]:
        patch_lens = [s['params']['patch_len'] for s in solutions]
        strides = [s['params']['stride'] for s in solutions]
    else:
        patch_lens = [s['patch_len'] for s in solutions]
        strides = [s['stride'] for s in solutions]
    
    macro_f1s = [s['macro_f1'] for s in solutions]
    trials = [s['trial_number'] for s in solutions]
    
    # Plot all solutions
    scatter = ax.scatter(patch_lens, strides, s=200, c=macro_f1s,
                        cmap='RdYlGn', alpha=0.8, edgecolor='black',
                        linewidth=1.5, vmin=min(macro_f1s), vmax=max(macro_f1s))
    
    # Highlight best F1
    best_idx = np.argmax(macro_f1s)
    ax.scatter(patch_lens[best_idx], strides[best_idx], s=600,
              marker='*', c='blue', edgecolor='black', linewidth=3,
              zorder=10, label=f'Best: patch={patch_lens[best_idx]}, stride={strides[best_idx]}')
    
    # Annotate top 5
    top5_idx = np.argsort(macro_f1s)[-5:]
    for i in top5_idx:
        overlap = (1 - strides[i] / patch_lens[i]) * 100
        ax.annotate(f'#{trials[i]}\nF1={macro_f1s[i]:.3f}\noverlap={overlap:.1f}%',
                   (patch_lens[i], strides[i]),
                   xytext=(8, 8), textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))
    
    # Add v0.3 original best point
    ax.scatter(26, 16, s=400, marker='D', c='red', edgecolor='black',
              linewidth=2, zorder=9, label='v0.3 Best (seed unknown): patch=26, stride=16')
    
    ax.set_xlabel('Patch Length', fontsize=13, fontweight='bold')
    ax.set_ylabel('Stride', fontsize=13, fontweight='bold')
    ax.set_title('v0.3.6 Architecture Parameter Space (Seed 42)\npatch_len ∈ [20,32], stride ∈ [10,20]',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Macro F1 Score', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'v0.3.6_pareto_architecture_space.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_overlap_vs_f1(solutions, output_dir):
    """Plot 4: Overlap ratio vs F1"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Check if solutions have 'params' or direct keys
    if 'params' in solutions[0]:
        patch_lens = [s['params']['patch_len'] for s in solutions]
        strides = [s['params']['stride'] for s in solutions]
    else:
        patch_lens = [s['patch_len'] for s in solutions]
        strides = [s['stride'] for s in solutions]
    
    overlaps = [(1 - s / p) * 100 for p, s in zip(patch_lens, strides)]
    macro_f1s = [s['macro_f1'] for s in solutions]
    trials = [s['trial_number'] for s in solutions]
    
    # Plot all solutions
    scatter = ax.scatter(overlaps, macro_f1s, s=150, alpha=0.7,
                        c=range(len(solutions)), cmap='coolwarm',
                        edgecolor='black', linewidth=1.5)
    
    # Highlight best F1
    best_idx = np.argmax(macro_f1s)
    ax.scatter(overlaps[best_idx], macro_f1s[best_idx], s=500,
              marker='*', c='gold', edgecolor='black', linewidth=2,
              zorder=10, label=f'Best: overlap={overlaps[best_idx]:.1f}%')
    
    # Add golden ratio reference line
    golden_overlap = 61.8
    ax.axvline(golden_overlap, color='green', linestyle='--', linewidth=2,
              alpha=0.7, label=f'Golden Ratio: {golden_overlap}%')
    
    # Add v0.3 reference
    v03_overlap = (1 - 16/26) * 100
    ax.axvline(v03_overlap, color='red', linestyle='--', linewidth=2,
              alpha=0.7, label=f'v0.3 Best: {v03_overlap:.1f}%')
    
    # Annotate top 5
    top5_idx = np.argsort(macro_f1s)[-5:]
    for i in top5_idx:
        ax.annotate(f'#{trials[i]}',
                   (overlaps[i], macro_f1s[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=10,
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                   facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Overlap Ratio (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Macro F1 Score', fontsize=13, fontweight='bold')
    ax.set_title('v0.3.6 Overlap Ratio vs Performance (Seed 42)\nOverlap = 1 - stride/patch_len',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    cbar = plt.colorbar(scatter, ax=ax, label='Trial Order')
    cbar.set_label('Trial Order', fontsize=11)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'v0.3.6_pareto_overlap_vs_f1.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_objectives_heatmap(solutions, output_dir):
    """Plot 5: All 15 objectives heatmap"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Build objective matrix (trials x 15 metrics)
    obj_names = []
    for h in HORIZONS:
        for m in METRICS:
            obj_names.append(f"{m}_{h}")
    
    matrix = []
    for s in solutions:
        row = [s['objectives'][name] for name in obj_names]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Sort by macro_f1 descending
    macro_f1s = [s['macro_f1'] for s in solutions]
    sorted_idx = np.argsort(macro_f1s)[::-1]
    matrix = matrix[sorted_idx]
    
    # Plot heatmap
    im = ax.imshow(matrix.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    
    ax.set_yticks(range(len(obj_names)))
    ax.set_yticklabels(obj_names, fontsize=9)
    ax.set_xlabel('Solutions (sorted by Macro F1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Objectives', fontsize=12, fontweight='bold')
    ax.set_title('v0.3.6 All 15 Objectives Heatmap (Seed 42)',
                fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Objective Value', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'v0.3.6_pareto_objectives_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize v0.3.6 Pareto Frontier')
    parser.add_argument('--archive', type=str, required=True,
                       help='Path to Pareto archive JSONL file')
    parser.add_argument('--output-dir', type=str,
                       default='ptst_dgm/results/visualizations_v0.3.6',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    print(f"Loading Pareto frontier from: {args.archive}")
    solutions = load_pareto_frontier(args.archive)
    print(f"✓ Loaded {len(solutions)} Pareto-optimal solutions")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating visualizations → {output_dir}")
    print("=" * 60)
    
    plot_macro_f1_vs_fpr(solutions, output_dir)
    plot_per_horizon_f1(solutions, output_dir)
    plot_architecture_space(solutions, output_dir)
    plot_overlap_vs_f1(solutions, output_dir)
    plot_objectives_heatmap(solutions, output_dir)
    
    print("=" * 60)
    print(f"✓ All visualizations saved to: {output_dir}")
    
    # Print summary
    macro_f1s = [s['macro_f1'] for s in solutions]
    best_idx = np.argmax(macro_f1s)
    best = solutions[best_idx]
    
    print("\n" + "=" * 60)
    print("BEST SOLUTION SUMMARY:")
    print("=" * 60)
    print(f"Trial: #{best['trial_number']}")
    print(f"Macro F1: {best['macro_f1']:.4f}")
    if 'params' in best:
        print(f"Architecture: patch={best['params']['patch_len']}, stride={best['params']['stride']}")
        overlap = (1 - best['params']['stride'] / best['params']['patch_len']) * 100
    else:
        print(f"Architecture: patch={best['patch_len']}, stride={best['stride']}")
        overlap = (1 - best['stride'] / best['patch_len']) * 100
    print(f"Overlap: {overlap:.1f}%")
    print(f"Mean FPR: {best['mean_fpr']:.4f}")
    print("=" * 60)

if __name__ == '__main__':
    main()
