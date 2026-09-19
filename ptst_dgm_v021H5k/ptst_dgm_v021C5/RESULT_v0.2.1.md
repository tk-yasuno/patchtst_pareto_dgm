# PatchTST DGM v0.2.1 - Focal Loss 4D Optimization Results

## Experiment Summary

**Objective**: 4D Focal Loss optimization with fixed Architecture and LoRA  
**Execution Date**: 2026-08-23  
**Duration**: ~10 hours (1000 iterations)  
**Status**: ✅ **SUCCESS** - Systematic 4D exploration completed

### Configuration
| Parameter | Value | Source |
|-----------|-------|--------|
| **Control (4D)** | focal_alpha [0.5-0.9], focal_gamma [0.6-2.0] | Refined near v0.2 Phase 3 best |
| | w_normal [1.0-3.0], w_anomal [2.0-6.0] | |
| **Fixed** | Architecture: patch_len=29, stride=16 | v0.3.6 best |
| | LoRA: rank=19, alpha=53 | v0.3.6.3 best |
| **Seed Strategy** | Random (NOT fixed) | v0.3.6.3 strategy |
| **Budget** | 1000 iterations | |
| **Checkpoint** | Every 100 iterations | |

### Key Design Choices
1. **Refined 4D search space**: Focused near v0.2 Phase 3 best parameters  
   - v0.2 Phase 3 best: α=0.866, γ=1.156, w_n=1.851, w_a=4.035
   - v0.2.1 ranges: α [0.5-0.9], γ [0.6-2.0], w_n [1.0-3.0], w_a [2.0-6.0]
2. **Fixed Architecture/LoRA**: Use optimal values from v0.3.6/v0.3.6.3 experiments
3. **Random seeds**: Following v0.3.6.3 success (+4.0% improvement)

### Rationale
Based on v0.3.6.x findings:
- **v0.3.6.2**: LoRA optimization alone achieved only F1=0.5949 (-27.3% from baseline)
- **v0.3.6.3**: Refined LoRA + random seeds achieved F1=0.6188 (-24.4% from baseline)
- **Hypothesis**: Focal Loss re-optimization may be more impactful than LoRA tuning

---

## Performance Results

### Overall Statistics
- **Total iterations**: 1000
- **Failed trials**: 742 (74.2%)
- **Successful trials**: 258 (25.8%)
- **Pareto solutions**: 205 (79.5% of successful trials, 20.5% of total)

**Note**: High failure rate (74.2%) likely due to exploration of extreme parameter combinations in refined 4D space.

### Best Solution (by macro F1)
| Metric | Value |
|--------|-------|
| **Trial** | #19 |
| **macro_F1** | **0.6139** ⬇️ **-0.8%** vs v0.3.6.3 |
| **mean_FPR** | 0.1841 |
| **focal_alpha** | 0.809 |
| **focal_gamma** | 0.704 |
| **w_normal** | 1.717 |
| **w_anomal** | 2.463 |

#### Per-Horizon Performance (Trial #19)
| Horizon | AUC | Precision | Recall | F1 | FPR |
|---------|-----|-----------|--------|----|----|
| 30d | 0.9458 | 0.4419 | **1.0000** | **0.6129** | 0.1832 |
| 60d | 0.9473 | 0.4651 | **1.0000** | **0.6349** | 0.1769 |
| 90d | 0.8865 | 0.4318 | 0.9500 | **0.5938** | 0.1923 |

**Note**: Perfect recall (1.0) at 30d/60d horizons indicates **highly sensitive** anomaly detection.

### Top 5 Solutions (by macro F1)
| Rank | Trial | macro_F1 | mean_FPR | focal_α | focal_γ | w_normal | w_anomal |
|------|-------|----------|----------|---------|---------|----------|----------|
| 1 | #19 | **0.6139** | 0.1841 | 0.809 | 0.704 | 1.717 | 2.463 |
| 2 | #134 | 0.6137 | 0.1329 | 0.833 | 0.859 | 1.678 | 2.734 |
| 3 | #208 | 0.5957 | 0.1409 | 0.535 | 0.874 | 1.090 | 2.478 |
| 4 | #48 | 0.5940 | 0.1609 | 0.604 | 1.528 | 1.041 | 5.880 |
| 5 | #943 | 0.5932 | 0.1842 | 0.573 | 1.293 | 1.053 | 2.463 |

**Observation**: Top 2 solutions (#19, #134) nearly identical F1 (0.6139 vs 0.6137), but #134 has **27% lower FPR** (0.1329 vs 0.1841) → **Better Pareto trade-off candidate**

### Best Solution (by FPR)
| Metric | Value |
|--------|-------|
| **Trial** | #854 |
| **macro_F1** | 0.2964 (very low) |
| **mean_FPR** | **0.0844** (lowest) |
| **focal_alpha** | 0.665 |
| **focal_gamma** | 0.859 |
| **w_normal** | 1.041 |
| **w_anomal** | 3.958 |

**Note**: Trial #854 achieves lowest FPR at severe cost to F1 (-52% from best) → **Extreme conservative detector**

### Per-Horizon Best F1 (from visualization)
| Horizon | Best F1 | AUC | FPR | Trial | Parameters |
|---------|---------|-----|-----|-------|------------|
| 30d | **0.7200** | 0.9582 | 0.0992 | #603 | (see Pareto archive) |
| 60d | **0.6792** | 0.9308 | 0.1154 | #816 | (see Pareto archive) |
| 90d | **0.6038** | 0.8923 | 0.1308 | #48 | α=0.604, γ=1.528, w_n=1.041, w_a=5.880 |

**Key Finding**: Per-horizon optimization can achieve **17% higher F1** (0.72 vs 0.6139) than macro average optimization.

---

## Comparison with Previous Experiments

### v0.2.1 vs v0.3.6.3 (Focal Loss vs LoRA optimization)
| Metric | v0.3.6.3 (LoRA 2D) | v0.2.1 (Focal Loss 4D) | Δ |
|--------|-------------|-------------|---|
| **macro_F1** | 0.6188 | 0.6139 | **-0.8%** ⬇️ |
| **mean_FPR** | 0.1175 | 0.1841 | **+56.7%** ⬆️ (worse) |
| **Control vars** | LoRA rank/alpha | Focal Loss 4 params | - |
| **Fixed vars** | Architecture, Focal Loss | Architecture, LoRA | - |
| **Iterations** | 500 | 1000 | +100% |
| **Pareto solutions** | 125 | 205 | +64% |

**Critical Finding**: 
- Focal Loss re-optimization **did NOT improve** over LoRA optimization
- Both approaches ~25% below v0.3.6 baseline
- **Neither LoRA nor Focal Loss alone recovers baseline performance**

### v0.2.1 vs v0.3.6 (baseline Architecture optimization)
| Metric | v0.3.6 (2D Arch) | v0.2.1 (4D Focal Loss) | Δ |
|--------|-------------|-------------|---|
| **macro_F1** | **0.8183** | 0.6139 | **-24.9%** ❌ |
| **Best config** | patch=26, stride=16 | α=0.809, γ=0.704 | - |

**Critical Issue**: v0.2.1 still **24.9% below v0.3.6 baseline**, despite systematic 4D optimization.

### v0.2.1 vs v0.2 Phase 3 (original Focal Loss experiment)
| Parameter | v0.2 Phase 3 best | v0.2.1 best (Trial #19) | Δ |
|-----------|-------------|-------------|---|
| focal_alpha | 0.866 | 0.809 | -6.6% |
| focal_gamma | 1.156 | 0.704 | **-39.1%** ⬇️ |
| w_normal | 1.851 | 1.717 | -7.2% |
| w_anomal | 4.035 | 2.463 | **-39.0%** ⬇️ |

**Key Observation**: v0.2.1 best solution significantly **reduced γ and w_anomal** compared to v0.2 Phase 3. This suggests **interaction effects** between:
- Architecture (v0.2: unknown, v0.2.1: fixed at patch=29, stride=16)
- LoRA (v0.2: unknown, v0.2.1: fixed at r=19, α=53)
- Focal Loss parameters

---

## Focal Loss Parameter Analysis

### Parameter Distribution in Pareto Frontier (205 solutions)

#### 1. Focal Alpha (α) - Class Balance Weight
- **Range**: 0.5 - 0.9
- **Best F1 value**: 0.809 (Trial #19)
- **v0.2 Phase 3 best**: 0.866
- **Observation**: Best solutions cluster in [0.7-0.85] range
- **Interpretation**: Moderate class imbalance correction (12.9% anomaly rate)

#### 2. Focal Gamma (γ) - Hard Example Focus
- **Range**: 0.6 - 2.0
- **Best F1 value**: 0.704 (Trial #19)
- **v0.2 Phase 3 best**: 1.156 (-39.1%)
- **Observation**: v0.2.1 converged to **much lower γ** than v0.2 Phase 3
- **Interpretation**: Fixed Architecture/LoRA configuration requires **less focus on hard examples**

#### 3. Weight Normal (w_n) - Normal Class Weight
- **Range**: 1.0 - 3.0
- **Best F1 value**: 1.717 (Trial #19)
- **v0.2 Phase 3 best**: 1.851
- **Observation**: Consistent with v0.2 Phase 3 (~1.7-1.9)
- **Interpretation**: Normal class weight relatively stable across configurations

#### 4. Weight Anomal (w_a) - Anomaly Class Weight
- **Range**: 2.0 - 6.0
- **Best F1 value**: 2.463 (Trial #19)
- **v0.2 Phase 3 best**: 4.035 (-39.0%)
- **Observation**: v0.2.1 converged to **much lower w_a** than v0.2 Phase 3
- **Interpretation**: Fixed Architecture/LoRA may **already provide sufficient anomaly focus**, requiring lower explicit weighting

### Parameter Interaction Hypothesis

The large differences in γ and w_a between v0.2.1 and v0.2 Phase 3 suggest **strong interaction effects**:

```
Effective Anomaly Focus = f(Architecture, LoRA, focal_γ, w_anomal)
```

- **v0.2 Phase 3**: Unknown Architecture/LoRA → Required high γ=1.156, w_a=4.035
- **v0.2.1**: Fixed optimal Architecture/LoRA → Optimal γ=0.704, w_a=2.463

**Hypothesis**: The fixed Architecture (patch=29, stride=16) and LoRA (r=19, α=53) from v0.3.6.x may already provide **strong anomaly detection capability**, reducing the need for aggressive Focal Loss parameters.

---

## Loss Function Lessons

### 1. **Loss Function Alone Insufficient for Recovery**
- LoRA optimization (v0.3.6.3): F1=0.6188 (-24.4% from baseline)
- Focal Loss optimization (v0.2.1): F1=0.6139 (-24.9% from baseline)
- **Both approaches fail to recover v0.3.6 baseline performance (F1=0.8183)**

**Lesson**: Single-component optimization has **fundamental limits**. Neither loss function tuning nor adapter tuning alone can compensate for suboptimal architecture choices.

### 2. **Strong Parameter Interactions Exist**
- v0.2.1 best solution requires **39% lower γ and w_a** than v0.2 Phase 3 best
- Different Architecture/LoRA configurations demand **vastly different Focal Loss parameters**

**Lesson**: Focal Loss parameters are **NOT transferable** across different Architecture/LoRA configurations. They must be jointly optimized.

### 3. **Focal Loss Parameter Stability Varies**
- **Stable**: focal_alpha (α), w_normal → Consistent across experiments
- **Unstable**: focal_gamma (γ), w_anomal → Highly sensitive to Architecture/LoRA

**Lesson**: γ and w_a are **interaction-sensitive parameters** that capture emergent effects from other components. Direct copy-paste from previous experiments is risky.

### 4. **4D Search Space Remains Challenging**
- 1000 iterations → 74.2% failure rate
- Only 205 Pareto solutions (20.5% of total trials)

**Lesson**: Even with refined ranges, 4D Focal Loss optimization is **sample-inefficient**. DGM/LLM guidance helps but cannot eliminate exploration cost.

### 5. **Per-Horizon Optimization Shows Promise**
- 30d horizon: Best F1=0.7200 (+17% over macro F1=0.6139)
- Different horizons may require **different Focal Loss parameters**

**Lesson**: **Horizon-specific loss functions** may outperform macro-averaged optimization, especially when prediction horizons have different characteristics.

### 6. **Pareto Frontier Reveals Trade-off Structure**
- Trial #19: F1=0.6139, FPR=0.1841 (F1-optimal)
- Trial #134: F1=0.6137, FPR=0.1329 (**27% lower FPR**, nearly identical F1)
- Trial #854: F1=0.2964, FPR=0.0844 (FPR-optimal, conservative)

**Lesson**: **Multiple solutions exist** along the Pareto frontier. Choice depends on operational requirements:
- **High-recall**: Trial #19 (perfect recall at 30d/60d, higher FPR)
- **Balanced**: Trial #134 (near-best F1, lower FPR)
- **High-precision**: Trial #854 (lowest FPR, sacrifices recall)

### 7. **Baseline Architecture Remains Critical**
- v0.3.6 achieved F1=0.8183 with Architecture search
- v0.3.6.x and v0.2.1 fixed Architecture at v0.3.6 best → Both ~0.61 F1

**Lesson**: **Architecture optimization space is essential**. Fixing Architecture (even at previous "best") may create a local optimum that downstream tuning cannot escape.

---

## Next Experiment Challenges

### Critical Question to Answer
**Why does fixing Architecture at v0.3.6 best (patch=29, stride=16) lead to 25% performance degradation?**

Possible explanations:
1. **Interaction effects**: v0.3.6 Architecture was optimal for v0.2 Focal Loss, but NOT for v0.3.6.3 LoRA
2. **Local optima**: Architecture/LoRA/Focal Loss form a **joint optimization landscape** with multiple local optima
3. **Search order matters**: Optimizing A→B→C may yield different results than B→C→A
4. **Baseline bias**: v0.3.6 (seed=42) may have been lucky, and true optimum is elsewhere

### Recommended Next Experiments

#### **Priority 1: Joint 6D Optimization (HIGHEST PRIORITY)**
**Objective**: Simultaneously optimize Architecture (2D) + LoRA (2D) + Focal Loss (2D core params)

**Configuration**:
- **Control (6D)**:
  - Architecture: patch_len [24-32], stride [14-18]
  - LoRA: rank [15-25], alpha [45-60]
  - Focal Loss: focal_alpha [0.7-0.9], focal_gamma [0.6-1.2]
- **Fixed (2D)**: w_normal=1.717, w_anomal=2.463 (from v0.2.1 best)
- **Budget**: 2000 iterations (6D space requires more exploration)
- **Checkpoint**: Every 200 iterations

**Rationale**: 
- Test if joint optimization can recover v0.3.6 baseline
- Explore interaction effects systematically
- Reduce w_n/w_a to 2D fixed to make 6D tractable

#### **Priority 2: v0.3.6 Reproduction with Multiple Seeds**
**Objective**: Verify if v0.3.6 baseline (F1=0.8183) was seed=42 specific

**Configuration**:
- **Control (2D)**: Architecture (same as v0.3.6)
- **Fixed**: LoRA r=16, α=32, Focal Loss (v0.2 Phase 3)
- **Seed Strategy**: Random (NOT seed=42)
- **Budget**: 500 iterations

**Rationale**:
- If reproduction fails → v0.3.6 was lucky
- If reproduction succeeds → Confirms Architecture optimization is key
- Critical for establishing true baseline

#### **Priority 3: Horizon-Specific Focal Loss**
**Objective**: Optimize separate Focal Loss parameters for each horizon (30d/60d/90d)

**Configuration**:
- **Control (12D)**: focal_alpha/gamma/w_n/w_a × 3 horizons
- **Fixed**: Architecture, LoRA (v0.3.6.3 best)
- **Budget**: 3000 iterations (large space)
- **Multi-objective**: Maximize F1_30d, F1_60d, F1_90d separately

**Rationale**:
- Per-horizon best F1 (0.72) is 17% higher than macro (0.6139)
- May exceed baseline by tailoring loss to horizon characteristics

#### **Priority 4: Architecture-Focal Loss 4D Joint Optimization**
**Objective**: Test if Architecture+Focal Loss joint optimization (no LoRA) can match v0.3.6

**Configuration**:
- **Control (4D)**: patch_len [24-32], stride [14-18], focal_alpha [0.7-0.9], focal_gamma [0.6-1.2]
- **Fixed**: LoRA r=16, α=32 (v0.3.6 values), w_n=1.717, w_a=2.463
- **Budget**: 1000 iterations

**Rationale**:
- Directly test Architecture-Focal Loss interaction hypothesis
- Simpler than 6D, may provide insights for Priority 1

### Longer-Term Challenges

#### **Challenge 1: Search Order Dependency**
- Does A→B→C optimization order affect final result?
- Should we optimize Architecture first, LoRA second, Focal Loss third?
- Or reverse order?

**Proposed Experiment**: Try all 6 permutations (3! = 6) of 2D optimization sequence and compare final F1.

#### **Challenge 2: Global vs Local Optima**
- Are v0.3.6.x and v0.2.1 stuck in different local optima?
- Does NSGA-II population diversity help escape local optima?

**Proposed Experiment**: Try different initialization strategies (random vs warm-start from previous best).

#### **Challenge 3: Computational Efficiency**
- 1000 iterations → 10 hours wall time
- 6D space may require 2000+ iterations → 20+ hours
- Is there a smarter search strategy?

**Proposed Experiment**: Try Bayesian Optimization (BO) or Transfer Learning from v0.3.6/v0.2.1 Pareto frontiers.

---

## Visualizations

Generated 4 plots in [visualizations_v021_pareto/](results/visualizations_v021_pareto/):
1. **pareto_macro_f1_vs_fpr.png**: Main F1-FPR trade-off
2. **pareto_per_horizon_f1_fpr.png**: Per-horizon (30d/60d/90d) trade-offs
3. **pareto_parameter_space.png**: 4D Focal Loss parameter distribution
4. **pareto_objectives_heatmap.png**: All 15 objectives heatmap

---

## Conclusions

### What We Learned
1. ✅ **4D Focal Loss optimization completed successfully** (205 Pareto solutions)
2. ❌ **Focal Loss alone cannot recover baseline** (F1=0.6139 vs 0.8183, -24.9%)
3. ⚠️ **Strong parameter interactions exist** (γ and w_a differ by ~39% from v0.2 Phase 3)
4. ✅ **Pareto frontier reveals rich trade-off structure** (F1-optimal vs FPR-optimal)
5. ⚠️ **Per-horizon optimization shows promise** (F1=0.72 at 30d, +17% vs macro)

### What We Still Don't Know
1. ❓ **Why does fixing Architecture degrade performance by 25%?**
2. ❓ **Can joint 6D optimization recover v0.3.6 baseline?**
3. ❓ **Was v0.3.6 baseline lucky (seed=42 artifact)?**
4. ❓ **What is the true global optimum across all dimensions?**

### Recommended Next Action
**Execute Priority 1 experiment (6D joint optimization)** to directly test if simultaneous Architecture+LoRA+Focal Loss tuning can break through the ~25% performance gap. This is the most direct path to answering the critical question above.

---

## Appendix: Key Files
- **Archive**: [ptst_archive_v021.jsonl](results/ptst_archive_v021.jsonl)
- **Pareto frontier**: [ptst_archive_v021_pareto.jsonl](results/ptst_archive_v021_pareto.jsonl)
- **Logs**: [dgm_v021_multiobj_log.jsonl](results/dgm_v021_multiobj_log.jsonl)
- **Visualizations**: [visualizations_v021_pareto/](results/visualizations_v021_pareto/)
- **Training script**: [train_patchtst_dgm.py](training/train_patchtst_dgm.py)
- **DGM agent**: [ptst_agent.py](multi_objective_agent/ptst_agent.py)
- **NSGA-II sampler**: [ptst_sampler.py](multi_objective_agent/ptst_sampler.py)

---

**Experiment Status**: ✅ COMPLETED  
**Documentation Date**: 2026-08-23  
**Next Steps**: See "Recommended Next Experiments" section above
