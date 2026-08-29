# PatchTST DGM v0.2.1C5 - 実験結果

## 実験概要

- **Version**: v0.2.1C5
- **実験日**: TBD
- **Total Budget**: 1,000 iterations
- **制約条件**: w_normal + w_anomal = 5.88 (v0.2 best sum)
- **目標**: macro F1 > 0.655

## 背景

v0.2.1Cで macro F1 = 0.637 を達成したが、制約条件 w_n + w_a = 2.0 が強すぎて目標0.655に到達できなかった。v0.2の最適解（w_n=1.85, w_a=4.03, sum=5.88）を基に制約を緩和。

## 固定パラメータ

| Parameter | Value | Source |
|-----------|-------|--------|
| patch_length | 26 | v0.2.3 best |
| stride | 16 | v0.2.3 best |
| lora_rank | 16 | v0.2.3 best |
| lora_alpha | 47 | v0.2.3 best |

## 制御変数

| Parameter | Range | Constraint |
|-----------|-------|------------|
| focal_alpha | [0.5, 0.9] | - |
| focal_gamma | [0.6, 2.0] | - |
| w_normal | [0.3, 5.5] | w_n + w_a = 5.88 |

## 実験結果

### ベースライン性能

```
TBD
```

### 最良解

```
TBD
```

### Pareto Frontier

```
TBD
```

### Horizon別性能

```
TBD
```

## 分析

### v0.2.1Cとの比較

```
TBD
```

### v0.2との比較

```
TBD
```

## 結論

```
TBD
```

## 次のステップ

```
TBD
```
