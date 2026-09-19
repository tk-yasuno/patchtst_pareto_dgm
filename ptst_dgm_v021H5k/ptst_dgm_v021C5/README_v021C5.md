# PatchTST DGM v0.2.1C5 - Focal Loss Optimization with Relaxed Constraint

## 実験概要

v0.2.1Cの制約条件を緩和し、v0.2の最適解の知見を活用したLoss function parametersの最適化実験。

### 背景

**v0.2.1Cの結果と課題:**
- macro F1 = 0.637 を達成（ベースライン比+29.3%）
- 目標0.655に対して97.2%到達
- **制約条件が強すぎた**: w_normal + w_anomal = 2.0

**v0.2の知見:**
- 最適解: w_normal = 1.85, w_anomal = 4.03
- **重みの合計 = 5.88**
- 重み比率: w_anomal / w_normal = 2.18

### 実験目的

制約条件を緩和（w_normal + w_anomal = **5.88**）し、**macro F1 > 0.655** を達成する。

## 実験設定

### 制御変数（3D + 緩和制約条件）

1. **Focal Loss alpha**: [0.5, 0.9]
2. **Focal Loss gamma**: [0.6, 2.0]
3. **Class weight w_normal**: [0.3, 5.5]

### 制約条件（緩和）

- **w_normal + w_anomalous = 5.88** (v0.2 best sum)
  - w_anomalousは自動計算される: `w_anomal = 5.88 - w_normal`
  - v0.2.1Cの2.0から大幅に緩和
  - v0.2の最適比率2.18を探索可能

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

### 緩和された制約条件の実装

`ptst_dgm_v021C5/multi_objective_agent/ptst_sampler.py`:

```python
# Sample 3D Focal Loss parameters
params["focal_alpha"] = trial.suggest_float("focal_alpha", 0.5, 0.9)
params["focal_gamma"] = trial.suggest_float("focal_gamma", 0.6, 2.0)
params["w_normal"] = trial.suggest_float("w_normal", 0.3, 5.5)

# Apply relaxed constraint: w_normal + w_anomal = 5.88 (v0.2 best sum)
params["w_anomal"] = 5.88 - params["w_normal"]
```

これにより：
- ✅ v0.2の最適解（w_n=1.85, w_a=4.03）を探索可能
- ✅ 重み比率2.18前後を柔軟に探索
- ✅ より大きな重み値で強力なclass weightingを実現

### コードベース

- **ベースコード**: `ptst_dgm_v021C` を完全移植
- **フォルダー**: `ptst_dgm_v021C5`
- **変更ファイル**: 
  - `multi_objective_agent/ptst_sampler.py` (制約条件を5.88に変更)
  - `multi_objective_agent/ptst_agent.py` (制約条件を5.88に変更)
  - `scripts/run_v021C5.ps1` (実行スクリプト)

## 実行方法

### 基本実行

```powershell
.\ptst_dgm_v021C5\scripts\run_v021C5.ps1
```

### カスタム設定

```powershell
.\ptst_dgm_v021C5\scripts\run_v021C5.ps1 `
    -TotalBudget 1000 `
    -CheckpointInterval 100 `
    -PopulationSize 20 `
    -Epochs 100
```

### Dry Run（テスト実行）

```powershell
.\ptst_dgm_v021C5\scripts\run_v021C5.ps1 -DryRun
```

## 出力

### Archive

- **Path**: `ptst_dgm_v021C5/results/ptst_archive_v021C5.jsonl`
- **Format**: JSONL (1 trial per line)
- **Contents**: Parameters + 15 objectives per trial

### Pareto Frontier

- **Path**: `ptst_dgm_v021C5/results/ptst_archive_v021C5_pareto.jsonl`
- **Contents**: Non-dominated solutions (Pareto optimal)

### Checkpoints

- **Directory**: `models/v021C5_checkpoints/`
- **Interval**: Every 100 iterations
- **Naming**: `checkpoint_iter{N}_trial{T}.pth`

### Log

- **Path**: `ptst_dgm_v021C5/results/ptst_archive_v021C5_log.jsonl`
- **Contents**: Detailed trial logs

## 期待される成果

1. **macro F1 > 0.655** の達成（v0.2.1Cを超える）
2. v0.2の最適重み比率2.18前後の再発見
3. より強力なclass weightingによる性能向上

## 各バージョンとの違い

| Aspect | v0.2.1C | v0.2.1C5 |
|--------|---------|----------|
| **重みの制約** | w_n + w_a = 2.0 | w_n + w_a = 5.88 |
| **w_normal範囲** | [0.1, 1.9] | [0.3, 5.5] |
| **v0.2最適解** | 探索不可 | 探索可能 ✅ |
| **達成F1** | 0.637 | TBD (目標>0.655) |
| **制約の根拠** | v0.2.3知見 | v0.2知見 |

## v0.2との比較

| Aspect | v0.2 | v0.2.1C5 |
|--------|------|----------|
| **最適化次元** | 4D独立 | 3D + 制約 |
| **重み制約** | なし | w_n + w_a = 5.88 |
| **Architecture** | 制御変数 | 固定 (v0.2.3 best) |
| **LoRA** | 制御変数 | 固定 (v0.2.3 best) |
| **Iterations** | - | 1,000 |

## ノート

- v0.2の知見（重み合計5.88）を制約条件として活用
- Architecture/LoRAは v0.2.3 の最適値で固定し、Loss functionに集中
- NSGA-IIによる多目的最適化で複数のトレードオフ解を発見
- v0.2.1Cより自由度が高く、より良い解の発見が期待される
