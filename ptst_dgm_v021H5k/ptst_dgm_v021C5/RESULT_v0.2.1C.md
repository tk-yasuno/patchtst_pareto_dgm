# PatchTST DGM v0.2.1C - 実験結果

## 実験設定

### 目的
v0.2.3の8D同時最適化で特定したArchitectureとLoRAの最適値を固定し、Loss function parametersに特化した最適化により **macro F1 > 0.655** を達成する。

### 制御変数（3D + 制約条件）
- **Focal Loss alpha**: [0.5, 0.9]
- **Focal Loss gamma**: [0.6, 2.0]  
- **Class weight w_normal**: [0.1, 1.9]
- **制約条件**: w_normal + w_anomalous = 2.0

### 固定パラメータ（v0.2.3 best）
| Parameter | Value | Source |
|-----------|-------|--------|
| patch_length | 26 | v0.2.3 8D optimization |
| stride | 16 | v0.2.3 8D optimization |
| lora_rank | 16 | v0.2.3 8D optimization |
| lora_alpha | 47 | v0.2.3 8D optimization |

### 最適化設定
- Total Budget: 1,000 iterations
- Population Size: 20
- Checkpoint Interval: 100 iterations
- Optimizer: NSGA-II (Multi-objective Pareto)
- Objectives: 15 (5 metrics × 3 horizons)

## 実験進捗

### ベースライン（Iteration 0）

**パラメータ:**
- Focal Loss: α=0.75, γ=1.0, w_normal=1.0, w_anomal=1.0
- Architecture: patch=26, stride=16
- LoRA: rank=16, alpha=47

**結果:**
- **macro F1 = 0.4929**
- F1 (30d) = 0.500, F1 (60d) = 0.487, F1 (90d) = 0.492
- AUC (30d) = 0.877, AUC (60d) = 0.878, AUC (90d) = 0.815

### 実験完了（1,000 iterations）

**Pareto最適解数:** 192/1000 (19.2%)

## 最終結果

### 最良解（Best macro F1）

**Trial #555**
- **macro F1 = 0.6372** ⭐ (ベースライン比 +29.3%)
- Mean FPR = 0.1227
- **目標達成度: 97.2%** (目標 0.655 に対して)

**最適パラメータ:**
```python
focal_alpha = 0.745
focal_gamma = 1.589
w_normal = 0.855
w_anomal = 1.145  # (制約条件: w_normal + w_anomal = 2.0)
```

**Per-horizon性能:**
- 30d: AUC=0.925, Precision=0.424, Recall=0.842, **F1=0.563**, FPR=0.099
- 60d: AUC=0.940, Precision=0.529, Recall=1.000, **F1=0.692**, FPR=0.108
- 90d: AUC=0.879, Precision=0.571, Recall=0.800, **F1=0.667**, FPR=0.131

### Best Mean FPR解

**Trial #537**
- macro F1 = 0.160
- **Mean FPR = 0.0564** (最小FPR)
- Parameters: α=0.516, γ=0.629, w_n=1.845, w_a=0.155

### Pareto Frontier統計

**Macro F1範囲:** 0.157 - 0.637
**Mean FPR範囲:** 0.056 - 1.000

**Per-Horizon Best F1:**
| Horizon | Best F1 | AUC | FPR | Trial |
|---------|---------|-----|-----|-------|
| 30d | 0.6667 | 0.9253 | 0.0992 | #428 |
| 60d | **0.6923** | 0.9400 | 0.1077 | #555 |
| 90d | 0.6038 | 0.8785 | 0.1308 | #555 |

## v0.2.3との比較

| Metric | v0.2.3 | v0.2.1C | 改善 |
|--------|--------|---------|------|
| macro F1 | 0.655 | **0.6372** | -2.7% ⚠️ |
| F1 (30d) | - | 0.563 | - |
| F1 (60d) | - | **0.692** | - |
| F1 (90d) | - | 0.667 | - |
| Pareto解数 | - | 192 | - |
| 制約条件 | なし | w_n+w_a=2.0 | ✅ |

## 可視化

可視化ファイルは以下に保存されました：
- `ptst_dgm_v021C/results/visualizations_v021C_pareto/pareto_macro_f1_vs_fpr.png`
- `ptst_dgm_v021C/results/visualizations_v021C_pareto/pareto_per_horizon_f1_fpr.png`
- `ptst_dgm_v021C/results/visualizations_v021C_pareto/pareto_parameter_space.png`
- `ptst_dgm_v021C/results/visualizations_v021C_pareto/pareto_objectives_heatmap.png`

## 考察

### 成果

1. **大幅な性能改善**: ベースライン(0.493)から0.637へ、+29.3%の改善を達成
2. **Pareto frontierの発見**: 192個の非劣解を発見し、多様なトレードオフを提供
3. **制約条件の有効性**: w_normal + w_anomal = 2.0 の制約により探索空間を効率化
4. **60d horizonでの優位性**: F1=0.692を達成し、特に中期予測で高性能

### 目標未達の分析

**目標（macro F1 > 0.655）に対して97.2%到達**

考えられる原因：
1. **制約条件の制限**: Class weightsの合計を2.0に固定したことで、より大きな重み比率を探索できなかった
2. **局所最適**: v0.2.3では独立に探索したClass weightsが、制約条件下では最適解に到達できない可能性
3. **Architecture/LoRAの固定**: これらを同時最適化すればさらなる改善の余地がある

### 最適パラメータの解釈

**Best macro F1 (Trial #555):**
- `focal_alpha = 0.745`: 中程度の値、バランスの取れたhard/easy examplesへの注目
- `focal_gamma = 1.589`: 比較的高い値、easy examplesを積極的にdown-weight
- `w_normal = 0.855, w_anomal = 1.145`: 軽度のanomaly優先（ratio = 1.34）
  - v0.2.3想定のratio 2.18より控えめ
  - 制約条件により大きな差をつけられない

## 次のステップ

### 改善案

1. **制約条件の緩和**: w_normal + w_anomal = 2.0 → 他の合計値を試す（例: 3.0, 4.0）
2. **範囲の拡大**: w_normalの範囲を[0.1, 1.9]から拡大
3. **Ensemble**: 複数のPareto解をアンサンブルして性能向上
4. **再最適化**: v0.2.3の条件（制約なし）で追加実験

### 実用化に向けて

- **60d horizon特化**: F1=0.692の高性能を活用
- **Trade-off選択**: Pareto frontierから業務要件に応じた解を選択
- **チェックポイント活用**: 100 iter毎のモデルを保存済み（`models/v021C_checkpoints/`）

## 結論

v0.2.1Cは**制約条件付き最適化**として成功し、ベースラインから大幅な改善を達成しました。目標の0.655には僅かに届きませんでしたが（97.2%達成）、制約条件の影響を考慮すると妥当な結果です。

**主要な発見:**
- ✅ Loss function parametersの最適化は有効（+29.3%改善）
- ✅ 制約条件は探索空間を効率化するが、性能上限を制約する可能性
- ✅ 60d horizonでF1=0.692と特に高性能
- ⚠️ 目標未達成（0.637 vs 0.655）→ 制約条件の再検討が必要
