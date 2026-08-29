"""
compare_v0.2_current.py
v0.2とcurrentのtrain_patchtst_dgm.pyを詳細比較

重要な違いを抽出：
- Seed関連
- DataLoader設定
- モデル初期化
- Optimizer/Scheduler
- 訓練ループ
"""
import difflib
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
V02_FILE = WORKSPACE_ROOT / "0_LogBAK" / "v0-2_500itr" / "ptst_dgm" / "training" / "train_patchtst_dgm.py"
CURRENT_FILE = WORKSPACE_ROOT / "ptst_dgm" / "training" / "train_patchtst_dgm.py"

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

def main():
    v02_lines = read_file(V02_FILE)
    current_lines = read_file(CURRENT_FILE)
    
    print("="*80)
    print("v0.2 vs Current: train_patchtst_dgm.py Comparison")
    print("="*80)
    print(f"v0.2 file:     {V02_FILE}")
    print(f"Current file:  {CURRENT_FILE}")
    print(f"v0.2 lines:    {len(v02_lines)}")
    print(f"Current lines: {len(current_lines)}")
    print("="*80)
    
    # Unified diff
    diff = difflib.unified_diff(
        v02_lines, 
        current_lines,
        fromfile='v0.2 (0_LogBAK/v0-2_500itr)',
        tofile='Current',
        lineterm='',
        n=3  # context lines
    )
    
    diff_lines = list(diff)
    
    if not diff_lines:
        print("\n✓ Files are IDENTICAL!")
        return
    
    print(f"\n{len(diff_lines)} lines of diff found.\n")
    
    # Filter important sections
    important_keywords = [
        'seed', 'random', 'torch.manual_seed', 'np.random.seed',
        'DataLoader', 'shuffle', 'num_workers', 'worker_init_fn',
        'build_patch_tst', 'PatchTSTWrapper', 'LoRAParams',
        'AdamW', 'optimizer', 'scheduler', 'ReduceLROnPlateau',
        '_train_epoch', 'for epoch', 'torch.save',
        'deterministic', 'benchmark', 'cudnn',
    ]
    
    print("="*80)
    print("IMPORTANT DIFFERENCES (filtered by keywords)")
    print("="*80)
    
    important_diffs = []
    for line in diff_lines:
        if any(kw in line.lower() for kw in important_keywords):
            important_diffs.append(line)
    
    if important_diffs:
        for line in important_diffs:
            print(line)
    else:
        print("(No differences in important sections - likely only comments/docs changed)")
    
    print("\n" + "="*80)
    print("FULL DIFF (first 200 lines)")
    print("="*80)
    for i, line in enumerate(diff_lines[:200]):
        print(line)
    
    # Save full diff to file
    output_file = WORKSPACE_ROOT / "ptst_dgm" / "results" / "diff_v0.2_vs_current.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(diff_lines))
    
    print(f"\n✓ Full diff saved to: {output_file.relative_to(WORKSPACE_ROOT)}")

if __name__ == "__main__":
    main()
