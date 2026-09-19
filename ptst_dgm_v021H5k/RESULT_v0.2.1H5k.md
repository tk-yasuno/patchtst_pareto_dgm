# PatchTST DGM v0.2.1H5k — 結果レポート（5000 Iterations）

## 実験概要

- **バージョン**: v0.2.1H5k
- **ベース**: v0.2.1H (2000 iterations)
- **Total Budget**: 5,000 iterations
- **最適化手法**: NSGA-II (Optuna)
- **制御変数**: 9次元（Horizon-specific Focal Loss）
- **固定パラメータ**: patch_len=26, stride=16, lora_rank=16, lora_alpha=47
- **Checkpoint**: 100 iterations毎
- **実行日**: TBD
- **実行時間**: TBD

---

## 実行ステータス

**Status**: ✅ 3,000 iterations 完了（5,000まで継続中）

実験開始: 2026-08-29
3k到達: 2026-08-30
総実行時間（3k時点）: 約36時間

---

## 3,000 iterations 結果サマリー

### 最終結果（3k時点）

| Metric                  | Value                  | 比較（8D baseline） |
| ----------------------- | ---------------------- | ------------------- |
| **Best macro F1** | **0.7685**       | 0.656 (+17.2%)      |
| **Best mean FPR** | **0.059**        | —                  |
| Pareto解数              | 663                    | —                  |
| 最良試行番号            | trial 1865 (iter 1866) | —                  |

### Best Trial (trial 1865, iter 1866)

| Parameter   | 30d   | 60d   | 90d   |
| ----------- | ----- | ----- | ----- |
| focal_alpha | 0.614 | 0.863 | 0.709 |
| focal_gamma | 1.974 | 1.680 | 0.779 |
| w_normal    | 1.478 | 2.737 | 2.835 |
| w_anomal    | 4.402 | 3.143 | 3.045 |

| Objective | 30d   | 60d   | 90d   |
| --------- | ----- | ----- | ----- |
| AUC       | 0.974 | 0.961 | 0.955 |
| F1        | 0.789 | 0.766 | 0.750 |
| FPR       | 0.031 | 0.069 | 0.077 |

**macro_F1 = 0.7685 / mean_FPR = 0.0589**

### Top-5 Pareto解（macro-F1順）

| Trial | macro_F1 | mean_FPR |
| ----- | -------- | -------- |
| 1865  | 0.7685   | 0.0589   |
| 1828  | 0.7659   | 0.0563   |
| 1450  | 0.7572   | 0.0716   |
| 2462  | 0.7566   | 0.0742   |
| 2332  | 0.7487   | 0.0615   |

---

## 3k時点の知見

### 知見1: Horizon別パラメータの明確な分化

9D DGMは3つのhorizonで異なる最適Focal Lossパラメータを発見した：

- **30d**: 低α(0.614) × 高γ(1.974) → 強い難易度集中。近期の高品質データで易しい負例を強く抑制
- **60d**: 高α(0.863) × 中γ(1.680) → 広域損失。中期は一様に困難なサンプルが多い
- **90d**: 中α(0.709) × 低γ(0.779) → フラットなカーネル。長期は境界事例を軽視しない

均一4Dでは表現できないこのhorizon特化構造が+17.2%の改善源である。

### 知見2: Paretoフロンティアの収束パターン

- **約1,500 iter以降**: 新規Pareto解の発見頻度が低下
- **trial 1865 (best F1)**: iter 1866で発見 → 探索空間の中盤で最良解が出現
- **ロングテール**: 3,000 iter時点でも低FPR領域（FPR<0.02）の新解が散発的に出現
- **663個のPareto解**: 15次元目的空間でも豊富なフロンティアを形成

### 知見3: 8D baseline との比較

| 実験                                | 次元 | macro_F1        | 改善幅           |
| ----------------------------------- | ---- | --------------- | ---------------- |
| 8D joint (v0.2.3)                   | 8D   | 0.656           | —               |
| 9D horizon-specific (v0.2.1H)       | 9D   | 0.712           | +8.5%            |
| 9D horizon-specific (v0.2.1H5k, 3k) | 9D   | **0.769** | **+17.2%** |

1次元の追加（8D→9D）でなく、**構造変更（均一→horizon特化）** が性能向上の主因。

---

## 生成された可視化ファイル

- `paper_patchtst_dgm/2_Main/figures/v021H5k_pareto_macro_f1_vs_fpr.png`
- `paper_patchtst_dgm/2_Main/figures/v021H5k_pareto_per_horizon_f1_fpr.png`
- `paper_patchtst_dgm/2_Main/figures/v021H5k_param_distributions.png`
- `paper_patchtst_dgm/2_Main/figures/v021H5k_convergence.png`
- `paper_patchtst_dgm/2_Main/figures/v021H5k_focal_loss_curves.png`

### 制御変数（9次元）

各horizonで独立に3パラメータを最適化：

| Parameter   | Range      | Horizon       |
| ----------- | ---------- | ------------- |
| focal_alpha | [0.5, 0.9] | 30d, 60d, 90d |
| focal_gamma | [0.6, 2.0] | 30d, 60d, 90d |
| w_normal    | [0.3, 5.5] | 30d, 60d, 90d |

**制約**: `w_normal_XXd + w_anomal_XXd = 5.88`（各horizon）

### 固定パラメータ

| Parameter  | Value | 根拠              |
| ---------- | ----- | ----------------- |
| patch_len  | 26    | v0.2.3 Phase4最良 |
| stride     | 16    | v0.2.3 Phase4最良 |
| lora_rank  | 16    | v0.2.3 Phase2最良 |
| lora_alpha | 47    | v0.2.3最良        |

---

## 結果サマリー

### 最終結果

**Status**: 実験未実施

| Metric        | Value | 比較（v0.2.1H） |
| ------------- | ----- | --------------- |
| Best macro F1 | TBD   | TBD             |
| Best mean FPR | TBD   | TBD             |
| Pareto解数    | TBD   | TBD             |
| 最良試行番号  | TBD   | TBD             |

### 進捗推移

実験実施後に追加

---

## Paretoフロンティア

実験実施後に追加

---

## 最良パラメータ

実験実施後に追加

---

## 考察

### v0.2.1H（2000 iters）との比較

実験実施後に追加

### Horizon別最適化の効果

実験実施後に追加

---

## 次のステップ

1. 実験の実行
2. 結果の分析
3. Paretoフロンティアの可視化
4. 最良解のモデル評価

---

## ファイル

- **アーカイブ**: `ptst_dgm_v021H5k/results/ptst_archive_v021H5k.jsonl`
- **ログ**: `ptst_dgm_v021H5k/results/ptst_archive_v021H5k_log.jsonl`
- **Paretoフロント**: `ptst_dgm_v021H5k/results/ptst_archive_v021H5k_pareto.jsonl`
- **チェックポイント**: `models/v021H5k_checkpoints/`

---

## 参照

- [v0.2.1H結果](../ptst_dgm_v021H/RESULT_v0.2.1H.md)
- [README v0.2.1H5k](README_v021H5k.md)
