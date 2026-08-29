# PatchTST DGM v0.2.1H — Horizon-Specific 9D Optimisation

## 概要

v0.2.1H は、各予測horizon（30日/60日/90日）で**独立にFocal Lossパラメータを最適化**する手法です。
従来の4D均一最適化（全horizonで同一パラメータ）から、horizon特化型9D最適化へと発展させ、
短期・中期・長期それぞれの予測難度に応じた損失関数を自動構築します。

---

## 背景と動機

v0.2.3の8D全結合最適化（macro-F1=0.656）は、全horizonに単一のFocal Lossパラメータを適用します。
しかし、異常検知の困難度は予測期間に依存します：

- **30日予測**：直近センサー値との相関が強く、高Precision達成が容易
- **60日予測**：中期的劣化パターンへの追従が必要
- **90日予測**：長期トレンドを捉えるために高Recall重視が必要

この観察から、各horizonに最適なFocal Lossパラメータを**独立に**最適化するアーキテクチャを設計しました。

---

## 最適化設計

### 制御変数（9次元）

各horizonで独立に3パラメータを制御し、合計9次元の探索空間を構成します：

| Horizon | focal_alpha (α) | focal_gamma (γ) | w_normal | 制約 |
|---------|----------------|-----------------|----------|------|
| 30d | [0.5, 0.9] | [0.6, 2.0] | [0.3, 5.5] | w_n + w_a = 5.88 |
| 60d | [0.5, 0.9] | [0.6, 2.0] | [0.3, 5.5] | w_n + w_a = 5.88 |
| 90d | [0.5, 0.9] | [0.6, 2.0] | [0.3, 5.5] | w_n + w_a = 5.88 |

**制約条件**（各horizonで独立に適用）：
```
w_normal_XXd + w_anomal_XXd = 5.88
```
この定数5.88はv0.2.3の最良解（w_n=2.501, w_a=2.225 → 合計4.726）から、
より広い重み和を許容する上限として設定。

### 固定パラメータ（v0.2.3最良値を継承）

| パラメータ | 値 | 根拠 |
|-----------|-----|------|
| patch_len | 26 | v0.2.3 Phase4最良 |
| stride | 16 | v0.2.3 Phase4最良 |
| lora_rank | 16 | v0.2.3 Phase2最良 |
| lora_alpha | 47 | v0.2.3最良（α/r≈2.94） |

### 実験設定

| 項目 | 値 |
|------|-----|
| 最適化手法 | NSGA-II（Optuna） |
| 反復数 | 2,000 iterations |
| 集団サイズ | 20 |
| LLMエージェント | Codestral 22B（Ollama） |
| 目的関数 | 15次元（AUC/P/R/F1/FPR × 3 horizons） |

---

## アーキテクチャ

### WeightedFocalLoss（Horizon別）

```python
class WeightedFocalLoss(nn.Module):
    """3つの独立FocalLossインスタンスで各horizonを制御."""
    def __init__(self, focal_alpha_30d, focal_gamma_30d, w_normal_30d, w_anomal_30d,
                 focal_alpha_60d, focal_gamma_60d, w_normal_60d, w_anomal_60d,
                 focal_alpha_90d, focal_gamma_90d, w_normal_90d, w_anomal_90d):
        self.focal_30d = FocalLoss(alpha=focal_alpha_30d, gamma=focal_gamma_30d)
        self.focal_60d = FocalLoss(alpha=focal_alpha_60d, gamma=focal_gamma_60d)
        self.focal_90d = FocalLoss(alpha=focal_alpha_90d, gamma=focal_gamma_90d)
        # per-horizon class weights
        ...

    def forward(self, logits, targets):
        # logits/targets: [B, 3] (30d/60d/90d)
        loss_30d = weighted_focal(logits[:,0], targets[:,0], self.focal_30d, w_30d)
        loss_60d = weighted_focal(logits[:,1], targets[:,1], self.focal_60d, w_60d)
        loss_90d = weighted_focal(logits[:,2], targets[:,2], self.focal_90d, w_90d)
        return (loss_30d + loss_60d + loss_90d) / 3.0
```

### パラメータサンプリング（NSGA-II）

```python
# ptst_sampler.py: 9Dパラメータを独立サンプリング
PARAM_BOUNDS = {
    "focal_alpha_30d": (0.5, 0.9), "focal_gamma_30d": (0.6, 2.0), "w_normal_30d": (0.3, 5.5),
    "focal_alpha_60d": (0.5, 0.9), "focal_gamma_60d": (0.6, 2.0), "w_normal_60d": (0.3, 5.5),
    "focal_alpha_90d": (0.5, 0.9), "focal_gamma_90d": (0.6, 2.0), "w_normal_90d": (0.3, 5.5),
}
FIXED_WEIGHT_SUM = 5.88  # w_anomal = 5.88 - w_normal (per horizon)
```

---

## 実装ファイル構成

| ファイル | 役割 |
|---------|------|
| `multi_objective_agent/ptst_sampler.py` | 9D NSGA-IIサンプラー、horizon別制約適用 |
| `multi_objective_agent/ptst_agent.py` | LLM検証ロジック（9Dパラメータ対応） |
| `multi_objective_agent/ptst_loop.py` | メインDGMループ（9D抽出・評価・アーカイブ） |
| `agent/archive.py` | PatchTSTAgentEntry（12フィールド対応） |
| `agent/evaluator.py` | サブプロセス評価器（12パラメータCLI） |
| `training/train_patchtst_dgm.py` | PyTorch訓練スクリプト（horizon別損失） |
| `scripts/run_v021H.ps1` | 2000 iterations実行スクリプト |

---

## 実行方法

```powershell
# ベースライン初期化
.venv-ptstf\Scripts\python.exe ptst_dgm_v021H\scripts\init_baseline.ps1

# 2000 iterations最適化実行
.\ptst_dgm_v021H\scripts\run_v021H.ps1

# Pareto frontier可視化
.venv-codagt\Scripts\python.exe ptst_dgm_v021H\scripts\visualize_pareto_frontier.py
```

---

## 実験結果サマリー

詳細は `RESULT_v0.2.1H.md` を参照。

| メトリック | v0.2.3 (8D均一) | v0.2.1H (9Dホライズン別) | 改善 |
|-----------|----------------|------------------------|------|
| Macro F1 | 0.656 | **0.6509** | -0.7% |
| 30d F1 (best) | 0.731 | **0.7317** | ≈同等 |
| 60d F1 (best) | 0.720 | **0.7200** | ≈同等 |
| 90d F1 (best) | 0.640 | **0.6538** | +2.1% |
| Pareto解数 | 352 | **283** | — |

**主要な知見**：
- 各horizonに特化したγ値の最適範囲を特定（γ=1.4〜1.9が全horizonで有効）
- 30日予測でF1=0.7317達成（v0.2系列最高値）
- Horizon別パラメータ戦略：短期は保守的（高precision）、長期はバランス型

# evaluator.evaluate()も9Dパラメータに対応させる
objectives = self.evaluator.evaluate(
    **params_30d, **params_60d, **params_90d,
    patch_len, stride, lora_rank, lora_alpha, seed
)
```

### evaluator.py

`evaluate()`メソッドのシグネチャを9Dパラメータに変更：

```python
def evaluate(
    self,
    focal_alpha_30d, focal_gamma_30d, w_normal_30d, w_anomal_30d,
    focal_alpha_60d, focal_gamma_60d, w_normal_60d, w_anomal_60d,
    focal_alpha_90d, focal_gamma_90d, w_normal_90d, w_anomal_90d,
    patch_len, stride, lora_rank, lora_alpha, seed=None
) -> Dict[str, float]:
    # train_patchtst_dgm.pyに9Dパラメータを渡す
    ...
```

## 実験目標

- **Target**: macro F1 > 0.655
- **Budget**: 2000 iterations
- **Checkpoint**: 100 iterations毎

## 背景

| Version | Approach | Best F1 | Notes |
|---------|----------|---------|-------|
| v0.2.1C | 3D (sum=2.0) | 0.6372 | 制約が厳しすぎ |
| v0.2.1C5 | 3D (sum=5.88) | 0.6342 | 緩和でも改善せず |
| v0.2.0R | 4D independent | 0.6538 | 99.8%達成 |
| **v0.2.1H** | **9D horizon-specific** | **TBD** | **各horizon最適化** |

## 実行方法

```powershell
# 1. 残りの修正完了後
.\ptst_dgm_v021H\scripts\run_v021H.ps1

# 2. 実行監視
# Terminal出力でhorizon別パラメータを確認

# 3. 結果分析
python ptst_dgm_v021H\scripts\analyze_results.py
```

## 期待される効果

1. **Horizon特化最適化**: 各horizonに最適な損失パラメータを発見
2. **柔軟性向上**: horizonごとに異なる戦略（例: 30dは保守的、90dは積極的）
3. **性能向上**: 個別最適化により全体のmacro F1向上

## 注意事項

- **複雑性増加**: 9Dパラメータ空間は探索が困難
- **計算コスト**: 2000 iterations × 100 epochs = 長時間実行
- **収束性**: 高次元空間でのNSGA-II収束を注意深く監視

## Status

- Created: 2026-08-27
- Status: **Setup Complete - Needs ptst_loop.py + evaluator.py fixes**
- Next: ptst_loop.py と evaluator.py の9D対応完了後、実行開始
