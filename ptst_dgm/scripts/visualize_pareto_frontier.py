"""
visualize_pareto_frontier.py
Visualize PatchTST Multi-Objective Pareto Frontier

Creates 4 visualizations:
1. Macro F1 vs Mean FPR (main trade-off)
2. Per-horizon F1 vs FPR (3 subplots)
3. Parameter space distribution (4 control variables)
4. All 15 objectives heatmap
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Configuration
PARETO_FILE = "ptst_dgm/results/ptst_archive_pareto.jsonl"
OUTPUT_DIR = "ptst_dgm/results/visualizations_pareto"
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
    """
    Plot 1: Macro F1 vs Mean FPR
    Main trade-off visualization for anomaly detection.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    macro_f1s = [s['macro_f1'] for s in solutions]
    mean_fprs = [s['mean_fpr'] for s in solutions]
    trials = [s['trial_number'] for s in solutions]
    
    # Plot all solutions
    scatter = ax.scatter(mean_fprs, macro_f1s, s=150, alpha=0.7,
                        c=range(len(solutions)), cmap='viridis',
                        edgecolor='black', linewidth=1.5, zorder=3)
    
    # Annotate with trial numbers
    for i, trial in enumerate(trials):
        ax.annotate(f'#{trial}', (mean_fprs[i], macro_f1s[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    # Highlight best solutions
    best_f1 = max(solutions, key=lambda s: s['macro_f1'])
    best_fpr = min(solutions, key=lambda s: s['mean_fpr'])
    
    ax.scatter(best_fpr['mean_fpr'], best_fpr['macro_f1'],
              s=400, marker='s', color='green', edgecolor='black', linewidth=2,
              label=f'Best FPR (#{best_fpr["trial_number"]})', zorder=5)
    ax.scatter(best_f1['mean_fpr'], best_f1['macro_f1'],
              s=400, marker='*', color='red', edgecolor='black', linewidth=2,
              label=f'Best F1 (#{best_f1["trial_number"]})', zorder=5)
    
    # Draw Pareto frontier curve
    sorted_sols = sorted(solutions, key=lambda s: s['mean_fpr'])
    sorted_fprs = [s['mean_fpr'] for s in sorted_sols]
    sorted_f1s = [s['macro_f1'] for s in sorted_sols]
    ax.plot(sorted_fprs, sorted_f1s, 'k--', alpha=0.3, linewidth=2,
           label='Pareto Frontier', zorder=2)
    
    ax.set_xlabel('Mean FPR (False Positive Rate)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Frontier: Macro F1 vs Mean FPR\n(PatchTST Multi-Objective DGM)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Solution Order', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_macro_f1_vs_fpr.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'pareto_macro_f1_vs_fpr.png'}")
    plt.close()

def plot_per_horizon_f1_fpr(solutions, output_dir):
    """
    Plot 2: Per-horizon F1 vs FPR
    Three subplots showing trade-offs at each time horizon.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, horizon in enumerate(HORIZONS):
        ax = axes[idx]
        
        # Extract data for this horizon
        f1s = [s['objectives'][f'f1_{horizon}'] for s in solutions]
        fprs = [s['objectives'][f'fpr_{horizon}'] for s in solutions]
        trials = [s['trial_number'] for s in solutions]
        
        # Plot solutions
        scatter = ax.scatter(fprs, f1s, s=120, alpha=0.7,
                           c=range(len(solutions)), cmap='plasma',
                           edgecolor='black', linewidth=1.5, zorder=3)
        
        # Annotate
        for i, trial in enumerate(trials):
            ax.annotate(f'#{trial}', (fprs[i], f1s[i]),
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=8)
        
        # Best solutions for this horizon
        best_f1_idx = np.argmax(f1s)
        best_fpr_idx = np.argmin(fprs)
        
        ax.scatter(fprs[best_fpr_idx], f1s[best_fpr_idx],
                  s=300, marker='s', color='green', edgecolor='black', linewidth=2,
                  label=f'Best FPR', zorder=5)
        ax.scatter(fprs[best_f1_idx], f1s[best_f1_idx],
                  s=300, marker='*', color='red', edgecolor='black', linewidth=2,
                  label=f'Best F1', zorder=5)
        
        # Pareto frontier line
        sorted_indices = np.argsort(fprs)
        sorted_fprs_h = [fprs[i] for i in sorted_indices]
        sorted_f1s_h = [f1s[i] for i in sorted_indices]
        ax.plot(sorted_fprs_h, sorted_f1s_h, 'k--', alpha=0.3, linewidth=2, zorder=2)
        
        ax.set_xlabel(f'FPR ({horizon})', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'F1 Score ({horizon})', fontsize=11, fontweight='bold')
        ax.set_title(f'{horizon} Horizon', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    plt.suptitle('Pareto Frontier: Per-Horizon F1 vs FPR\n(PatchTST Multi-Objective DGM)',
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_per_horizon_f1_fpr.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'pareto_per_horizon_f1_fpr.png'}")
    plt.close()

def plot_parameter_space(solutions, output_dir):
    """
    Plot 3: Parameter space distribution
    4 control variables: focal_alpha, focal_gamma, w_normal, w_anomal
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Extract data
    macro_f1s = np.array([s['macro_f1'] for s in solutions])
    focal_alphas = np.array([s['params']['focal_alpha'] for s in solutions])
    focal_gammas = np.array([s['params']['focal_gamma'] for s in solutions])
    w_normals = np.array([s['params']['w_normal'] for s in solutions])
    w_anomals = np.array([s['params']['w_anomal'] for s in solutions])
    trials = [s['trial_number'] for s in solutions]
    
    # Panel 1: focal_alpha vs focal_gamma
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(focal_alphas, focal_gammas, s=150, c=macro_f1s,
                          cmap='RdYlGn', alpha=0.8, edgecolor='black',
                          linewidth=1.5, vmin=min(macro_f1s), vmax=max(macro_f1s))
    for i, trial in enumerate(trials):
        ax1.annotate(f'#{trial}', (focal_alphas[i], focal_gammas[i]),
                    xytext=(3, 3), textcoords='offset points', fontsize=8)
    ax1.set_xlabel('Focal Alpha', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Focal Gamma', fontsize=11, fontweight='bold')
    ax1.set_title('Focal Loss Parameters', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Macro F1', fontsize=9)
    
    # Panel 2: w_normal vs w_anomal
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(w_normals, w_anomals, s=150, c=macro_f1s,
                          cmap='RdYlGn', alpha=0.8, edgecolor='black',
                          linewidth=1.5, vmin=min(macro_f1s), vmax=max(macro_f1s))
    for i, trial in enumerate(trials):
        ax2.annotate(f'#{trial}', (w_normals[i], w_anomals[i]),
                    xytext=(3, 3), textcoords='offset points', fontsize=8)
    ax2.set_xlabel('Weight Normal', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Weight Anomal', fontsize=11, fontweight='bold')
    ax2.set_title('Class Weights', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Macro F1', fontsize=9)
    
    # Panel 3: focal_alpha vs w_anomal (cross-parameter)
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(focal_alphas, w_anomals, s=150, c=macro_f1s,
                          cmap='RdYlGn', alpha=0.8, edgecolor='black',
                          linewidth=1.5, vmin=min(macro_f1s), vmax=max(macro_f1s))
    for i, trial in enumerate(trials):
        ax3.annotate(f'#{trial}', (focal_alphas[i], w_anomals[i]),
                    xytext=(3, 3), textcoords='offset points', fontsize=8)
    ax3.set_xlabel('Focal Alpha', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Weight Anomal', fontsize=11, fontweight='bold')
    ax3.set_title('Cross-Parameter: Alpha vs Anomal Weight', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Macro F1', fontsize=9)
    
    # Panel 4: focal_gamma vs w_normal (cross-parameter)
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(focal_gammas, w_normals, s=150, c=macro_f1s,
                          cmap='RdYlGn', alpha=0.8, edgecolor='black',
                          linewidth=1.5, vmin=min(macro_f1s), vmax=max(macro_f1s))
    for i, trial in enumerate(trials):
        ax4.annotate(f'#{trial}', (focal_gammas[i], w_normals[i]),
                    xytext=(3, 3), textcoords='offset points', fontsize=8)
    ax4.set_xlabel('Focal Gamma', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Weight Normal', fontsize=11, fontweight='bold')
    ax4.set_title('Cross-Parameter: Gamma vs Normal Weight', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('Macro F1', fontsize=9)
    
    plt.suptitle('Pareto Frontier: Parameter Space Distribution\n(PatchTST Multi-Objective DGM)',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_parameter_space.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'pareto_parameter_space.png'}")
    plt.close()

def plot_objectives_heatmap(solutions, output_dir):
    """
    Plot 4: All 15 objectives heatmap
    Rows = solutions (sorted by macro F1), Columns = 15 objectives
    """
    fig, ax = plt.subplots(figsize=(14, max(8, len(solutions) * 0.5)))
    
    # Sort solutions by macro F1 descending
    sorted_solutions = sorted(solutions, key=lambda s: s['macro_f1'], reverse=True)
    
    # Build matrix: rows = solutions, cols = 15 objectives
    objective_keys = []
    for horizon in HORIZONS:
        for metric in METRICS:
            objective_keys.append(f'{metric}_{horizon}')
    
    matrix = []
    row_labels = []
    for sol in sorted_solutions:
        row = [sol['objectives'][key] for key in objective_keys]
        matrix.append(row)
        row_labels.append(f"Trial #{sol['trial_number']}")
    
    matrix = np.array(matrix)
    
    # Normalize each column to [0, 1] for better visualization
    # For FPR, invert so lower is better (green)
    matrix_normalized = np.zeros_like(matrix)
    for col_idx, key in enumerate(objective_keys):
        col_data = matrix[:, col_idx]
        if 'fpr' in key:
            # Invert FPR: lower is better
            col_min, col_max = col_data.min(), col_data.max()
            if col_max > col_min:
                matrix_normalized[:, col_idx] = 1 - (col_data - col_min) / (col_max - col_min)
            else:
                matrix_normalized[:, col_idx] = 0.5
        else:
            # Higher is better for AUC/P/R/F1
            col_min, col_max = col_data.min(), col_data.max()
            if col_max > col_min:
                matrix_normalized[:, col_idx] = (col_data - col_min) / (col_max - col_min)
            else:
                matrix_normalized[:, col_idx] = 0.5
    
    # Plot heatmap
    im = ax.imshow(matrix_normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(objective_keys)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(objective_keys, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    
    # Annotate with actual values
    for i in range(len(row_labels)):
        for j in range(len(objective_keys)):
            value = matrix[i, j]
            text_color = 'white' if matrix_normalized[i, j] < 0.5 else 'black'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                   color=text_color, fontsize=7)
    
    ax.set_xlabel('Objectives', fontsize=12, fontweight='bold')
    ax.set_ylabel('Solutions (sorted by Macro F1)', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Frontier: All 15 Objectives Heatmap\n(PatchTST Multi-Objective DGM, normalized per column)',
                fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Score (0=worst, 1=best, FPR inverted)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_objectives_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'pareto_objectives_heatmap.png'}")
    plt.close()

def print_summary(solutions):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("Pareto Frontier Summary (PatchTST Multi-Objective DGM)")
    print("="*80)
    print(f"Total solutions: {len(solutions)}")
    
    best_f1 = max(solutions, key=lambda s: s['macro_f1'])
    best_fpr = min(solutions, key=lambda s: s['mean_fpr'])
    
    print(f"\nBest Macro F1: {best_f1['macro_f1']:.4f} (Trial #{best_f1['trial_number']})")
    print(f"  - Mean FPR: {best_f1['mean_fpr']:.4f}")
    print(f"  - Parameters: α={best_f1['params']['focal_alpha']:.3f}, γ={best_f1['params']['focal_gamma']:.3f}, "
          f"w_n={best_f1['params']['w_normal']:.3f}, w_a={best_f1['params']['w_anomal']:.3f}")
    
    print(f"\nBest Mean FPR: {best_fpr['mean_fpr']:.4f} (Trial #{best_fpr['trial_number']})")
    print(f"  - Macro F1: {best_fpr['macro_f1']:.4f}")
    print(f"  - Parameters: α={best_fpr['params']['focal_alpha']:.3f}, γ={best_fpr['params']['focal_gamma']:.3f}, "
          f"w_n={best_fpr['params']['w_normal']:.3f}, w_a={best_fpr['params']['w_anomal']:.3f}")
    
    macro_f1s = [s['macro_f1'] for s in solutions]
    mean_fprs = [s['mean_fpr'] for s in solutions]
    
    print(f"\nMacro F1 Range: {min(macro_f1s):.4f} - {max(macro_f1s):.4f}")
    print(f"Mean FPR Range: {min(mean_fprs):.4f} - {max(mean_fprs):.4f}")
    
    print("\nPer-Horizon Best F1:")
    for horizon in HORIZONS:
        best_h = max(solutions, key=lambda s: s['objectives'][f'f1_{horizon}'])
        print(f"  {horizon}: {best_h['objectives'][f'f1_{horizon}']:.4f} "
              f"(AUC={best_h['objectives'][f'auc_{horizon}']:.4f}, "
              f"FPR={best_h['objectives'][f'fpr_{horizon}']:.4f}, "
              f"Trial #{best_h['trial_number']})")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Visualize PatchTST Pareto Frontier')
    parser.add_argument('--pareto-file', type=str, default=PARETO_FILE,
                       help='Path to Pareto frontier JSONL file')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                       help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Load data
    print("Loading Pareto frontier data...")
    solutions = load_pareto_frontier(args.pareto_file)
    print(f"Loaded {len(solutions)} Pareto-optimal solutions")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Print summary
    print_summary(solutions)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_macro_f1_vs_fpr(solutions, output_dir)
    plot_per_horizon_f1_fpr(solutions, output_dir)
    plot_parameter_space(solutions, output_dir)
    plot_objectives_heatmap(solutions, output_dir)
    
    print("\n" + "="*80)
    print(f"✓ All visualizations saved to: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
