# PatchTST DGM v0.2.1H5k — Horizon-Specific 9D Optimisation (5000 iterations)

## 概要

v0.2.1H5k は、v0.2.1Hを基に**5000イテレーション**に拡張した深堀探索版です。
各予測horizon（30日/60日/90日）で**独立にFocal Lossパラメータを最適化**する手法を、
より長期間の探索により、さらなる性能向上を目指します。

---

## 背景と動機

v0.2.1H（2000イテレーション）で良好な成果を達成しましたが、
探索空間が9次元と大きいため、より深い探索によりさらなる最適解の発見が期待できます。
v0.2.1H5kでは、反復数を2000→5000に拡張し、より広範な探索を実施します。

### 過去の実績

- **v0.2.1H (2000 iters)**: 成果を上げた
- **v0.2.1H5k (5000 iters)**: さらなる探索深堀

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

### 固定パラメータ（v0.2.3最良値を継承）

| パラメータ | 値 | 根拠 |
|-----------|-----|------|
| patch_len | 26 | v0.2.3 Phase4最良 |
| stride | 16 | v0.2.3 Phase4最良 |
| lora_rank | 16 | v0.2.3 Phase2最良 |
| lora_alpha | 47 | v0.2.3最良（α/r≈2.94） |

### 実験設定

| 項目 | 値 | 変更点 |
|------|-----|--------|
| 最適化手法 | NSGA-II（Optuna） | - |
| 反復数 | **5,000 iterations** | **2000→5000に拡張** |
| 集団サイズ | 20 | - |
| LLMエージェント | Codestral 22B（Ollama） | - |
| 目的関数 | 15次元（AUC/P/R/F1/FPR × 3 horizons） | - |
| Checkpoint | 100 iters毎 | - |

---

## アーキテクチャ

v0.2.1Hと同一のアーキテクチャを使用します。

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

---

## 実行方法

### 1. 実験の開始

```powershell
# デフォルト設定（5000イテレーション）
.\ptst_dgm_v021H5k\scripts\run_v021H5k.ps1

# カスタム設定
.\ptst_dgm_v021H5k\scripts\run_v021H5k.ps1 -TotalBudget 5000 -CheckpointInterval 100
```

### 2. パラメータ

- `-TotalBudget`: 総反復数（デフォルト: 5000）
- `-PopulationSize`: NSGA-II集団サイズ（デフォルト: 20）
- `-Epochs`: 各トライアルの訓練エポック数（デフォルト: 100）
- `-CheckpointInterval`: チェックポイント保存間隔（デフォルト: 100）
- `-DataPath`: データセットパス（デフォルト: `data\golden_testset`）
- `-Archive`: アーカイブファイルパス（デフォルト: `ptst_dgm_v021H5k\results\ptst_archive_v021H5k.jsonl`）

---

## 期待される成果

- **v0.2.1Hからの改善**: 2000イテレーションで発見できなかった最適解の発見
- **Paretoフロンティアの拡張**: より多様な高性能解の発見
- **Horizon特化最適化の深化**: 各horizonに最適なFocal Lossパラメータの精緻化
- **目標**: macro F1 > 0.655 の達成と維持

---

## チェックポイントとモニタリング

実験中、以下のファイルで進捗を確認できます：

- **アーカイブ**: `ptst_dgm_v021H5k\results\ptst_archive_v021H5k.jsonl`
- **モデル**: `models\v021H5k_checkpoints\best_trial_XXXX.pth`（100イテレーション毎）
- **ログ**: `ptst_dgm_v021H5k\results\ptst_archive_v021H5k_log.jsonl`

---

## 注意事項

- 実験時間: 5000イテレーション × 約30-40分/iter = 約2500-3300時間（約104-138日）
- GPU使用: CUDA対応GPU推奨（`CUDA_VISIBLE_DEVICES=0`）
- ディスク容量: チェックポイントファイルが大量に生成されるため、十分な空き容量を確保

---

## バージョン履歴

- **v0.2.1H**: 2000イテレーション版（ベースライン）
- **v0.2.1H5k**: 5000イテレーション版（深堀探索）

---

## 参照

- ベース実験: [v0.2.1H](../ptst_dgm_v021H/README_v021H.md)
- 結果: [RESULT_v0.2.1H5k.md](RESULT_v0.2.1H5k.md)
