# PatchTST DGM v0.2.1C - Focal Loss Optimization with Constraint

## 実験概要

v0.2.3の8D同時最適化の成果を活用し、Loss function parametersに特化した最適化を実施する追加実験。

### 背景

v0.2.3では以下の成果を得た：
- **macro F1 = 0.655** を達成（限界に到達）
- Architecture最適値を特定: `patch_length=26, stride=16`
- LoRA最適値を特定: `r=16, alpha=47`
- しかし、Loss function parameters（Focal alpha, gamma, Class weights）が十分に最適化されていない

### 実験目的

Loss function parametersの最適化により **macro F1 > 0.655** を達成する。

## 実験設定

### 制御変数（3D + 制約条件）

1. **Focal Loss alpha**: [0.5, 0.9]
2. **Focal Loss gamma**: [0.6, 2.0]
3. **Class weight w_normal**: [0.1, 1.9]

### 制約条件

- **w_normal + w_anomalous = 2.0**
  - w_anomalousは自動計算される: `w_anomal = 2.0 - w_normal`

### 固定パラメータ（v0.2.3 best）

| Parameter | Value | Source |
|-----------|-------|--------|
| **patch_length** | 26 | v0.2.3 8D optimization |
| **stride** | 16 | v0.2.3 8D optimization |
| **lora_rank** | 16 | v0.2.3 8D optimization |
| **lora_alpha** | 47 | v0.2.3 8D optimization |
| **seed** | random | Diversity for DGM exploration |

### 最適化設定

- **Total Budget**: 1,000 iterations
- **Population Size**: 20
- **Checkpoint Interval**: 100 iterations
- **Training Epochs**: 100
- **Optimizer**: NSGA-II (Multi-objective Pareto frontier)
- **Objectives**: 15 (AUC/Precision/Recall/F1/FPR × 30d/60d/90d)

## 実装の特徴

### 制約条件の実装

`ptst_dgm_v021C/multi_objective_agent/ptst_sampler.py`:

```python
# Sample 3D Focal Loss parameters
params["focal_alpha"] = trial.suggest_float("focal_alpha", 0.5, 0.9)
params["focal_gamma"] = trial.suggest_float("focal_gamma", 0.6, 2.0)
params["w_normal"] = trial.suggest_float("w_normal", 0.1, 1.9)

# Apply constraint: w_normal + w_anomal = 2.0
params["w_anomal"] = 2.0 - params["w_normal"]
```

これにより、常に `w_normal + w_anomal = 2.0` が満たされる。

### コードベース

- **ベースコード**: `ptst_dgm_v021` を完全移植
- **フォルダー**: `ptst_dgm_v021C`
- **変更ファイル**: 
  - `multi_objective_agent/ptst_sampler.py` (制約条件実装)
  - `scripts/run_v021C.ps1` (実行スクリプト)

## 実行方法

### 基本実行

```powershell
.\ptst_dgm_v021C\scripts\run_v021C.ps1
```

### カスタム設定

```powershell
.\ptst_dgm_v021C\scripts\run_v021C.ps1 `
    -TotalBudget 1000 `
    -CheckpointInterval 100 `
    -PopulationSize 20 `
    -Epochs 100
```

### Dry Run（テスト実行）

```powershell
.\ptst_dgm_v021C\scripts\run_v021C.ps1 -DryRun
```

## 出力

### Archive

- **Path**: `ptst_dgm_v021C/results/ptst_archive_v021C.jsonl`
- **Format**: JSONL (1 trial per line)
- **Contents**: Parameters + 15 objectives per trial

### Pareto Frontier

- **Path**: `ptst_dgm_v021C/results/ptst_archive_v021C_pareto.jsonl`
- **Contents**: Non-dominated solutions (Pareto optimal)

### Checkpoints

- **Directory**: `models/v021C_checkpoints/`
- **Interval**: Every 100 iterations
- **Naming**: `checkpoint_iter{N}_trial{T}.pth`

### Log

- **Path**: `ptst_dgm_v021C/results/ptst_archive_v021C_log.jsonl`
- **Contents**: Detailed trial logs

## 期待される成果

1. **macro F1 > 0.655** の達成
2. Loss function parametersの最適値の特定
3. v0.2.3との比較によるArchitecture/LoRA固定の効果検証

## v0.2.3との主な違い

| Aspect | v0.2.3 | v0.2.1C |
|--------|--------|---------|
| **最適化次元** | 8D | 3D (+ 制約条件) |
| **Architecture** | 制御変数 | 固定 (v0.2.3 best) |
| **LoRA** | 制御変数 | 固定 (v0.2.3 best) |
| **Class weights** | 独立 | 制約あり (合計=2) |
| **Iterations** | - | 1,000 |
| **目標** | 探索的 | macro F1 > 0.655 |

## ノート

- v0.2.3で特定した最適値を完全に固定することで、Loss function parametersの最適化に計算資源を集中
- 制約条件によりclass weightsの探索空間を制限し、より効率的な最適化を実現
- NSGA-IIによる多目的最適化により、複数のトレードオフ解を発見可能
