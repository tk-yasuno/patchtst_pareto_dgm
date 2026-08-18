# DGM v0.7: Multi-Objective Pareto Optimization Design

**Date**: 2026-08-12  
**Status**: 🚧 In Design  
**Based on**: v0.5 Optuna Adaptive (F1=0.9433)

---

## 1. Motivation

### 1.1 Problem: Single-Objective Bias

Current DGM strategies optimize **macro F1 only**, which can lead to:

- **Unbalanced class performance**: High F1 but poor performance on minority classes
- **Pareto sub-optimality**: Trade-offs between classes are not explored
- **Hidden weaknesses**: Some classes may have low accuracy despite high overall F1

**Example from v0.5 Best (F1=0.9433)**:
```python
per_class_recall = [1.0, 0.733, 1.0, 1.0, 1.0, 0.917, 1.0, 1.0, 1.0, 1.0]
#                        ^^^^                      ^^^^
#                      Class 1                   Class 5
# Two classes have recall < 1.0, indicating potential for improvement
```

### 1.2 Solution: Multi-Objective Pareto Optimization

Instead of optimizing F1 alone, **simultaneously optimize 10 class-level accuracies**:

$$
\text{Maximize: } (Acc_{\text{class0}}, Acc_{\text{class1}}, \ldots, Acc_{\text{class9}})
$$

This produces a **Pareto frontier** of non-dominated solutions, where no single class can be improved without sacrificing another.

**Benefits**:
1. **Balanced performance**: All classes receive equal optimization priority
2. **Trade-off exploration**: Discover weight configurations that balance competing objectives
3. **Multiple solutions**: Pareto frontier provides a choice of models based on use-case priorities

---

## 2. Design: v0.7 Multi-Objective Optuna DGM

### 2.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  v0.7 Multi-Objective DGM Loop                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │ MultiObjSampler  │──────▶│ MultiObjAgent    │            │
│  │ (NSGA-II)        │      │ (LLM Validator)  │            │
│  └────────┬─────────┘      └────────┬─────────┘            │
│           │ suggest()               │ validate()            │
│           │ 10D weights             │ micro-adjust          │
│           ▼                         ▼                        │
│  ┌────────────────────────────────────────┐                │
│  │ Evaluator (LoRA 15 epochs)             │                │
│  │ Returns: [acc0, acc1, ..., acc9]       │                │
│  └────────┬───────────────────────────────┘                │
│           │ tell()                                           │
│           ▼                                                  │
│  ┌──────────────────────────────────────┐                  │
│  │ Pareto Archive                        │                  │
│  │ - Non-dominated solutions             │                  │
│  │ - Hypervolume tracking                │                  │
│  └──────────────────────────────────────┘                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Optuna Multi-Objective Configuration

```python
import optuna
from optuna.samplers import NSGAIISampler

study = optuna.create_study(
    directions=["maximize"] * 10,  # 10 objectives: one per class
    sampler=NSGAIISampler(
        population_size=20,        # Larger population for 10D Pareto frontier
        mutation_prob=0.1,
        crossover_prob=0.9,
        seed=42
    )
)
```

**Key Parameters**:
- `directions`: `["maximize"] * 10` - Optimize all 10 class accuracies
- `population_size=20`: NSGA-II maintains 20 solutions in Pareto frontier
- `mutation_prob=0.1`: Small mutations to explore nearby weight configurations
- `crossover_prob=0.9`: High crossover to exploit successful weight combinations

### 2.3 Evaluation Function

```python
def evaluate_class_accuracies(weights: List[float]) -> List[float]:
    """
    Train LoRA with given class_weights and return per-class accuracy.
    
    Returns:
        accuracies: [acc_class0, acc_class1, ..., acc_class9]
                    Each in range [0.0, 1.0]
    """
    metrics = train_lora(weights)
    per_class_recall = metrics["per_class_recall"]  # [10] values
    return per_class_recall  # Use recall as proxy for class accuracy
```

**Rationale**:
- **Per-class recall** directly measures each class's prediction quality
- Already available from existing evaluator (no code change needed)
- Range [0.0, 1.0] is natural for Optuna to maximize

### 2.4 Pareto Archive Management

```python
class ParetoArchive:
    def __init__(self):
        self.solutions = []  # List of (weights, accuracies, trial_number)
    
    def add(self, weights, accuracies, trial_number):
        """Add solution if non-dominated, remove dominated solutions."""
        if self._is_dominated(accuracies):
            return False  # Dominated, reject
        
        # Remove solutions dominated by new entry
        self.solutions = [s for s in self.solutions if not self._dominates(accuracies, s[1])]
        self.solutions.append((weights, accuracies, trial_number))
        return True
    
    def _dominates(self, acc_a, acc_b):
        """acc_a dominates acc_b if acc_a >= acc_b in all dimensions and > in at least one."""
        return all(a >= b for a, b in zip(acc_a, acc_b)) and any(a > b for a, b in zip(acc_a, acc_b))
    
    def get_hypervolume(self):
        """Compute hypervolume indicator (quality of Pareto frontier)."""
        from optuna.study import StudyDirection
        import optuna
        return optuna.visualization._hypervolume(
            self.solutions,
            reference_point=[0.0] * 10  # Worst case: all classes 0% accuracy
        )
```

---

## 3. Implementation Plan

### 3.1 File Structure

```
dgm_exp/
  multi_objective_agent/
    __init__.py
    multi_objective_sampler.py   # NSGA-II wrapper
    multi_objective_agent.py     # LLM validator (similar to adapt_agent.py)
    multi_objective_loop.py      # Main DGM loop with Pareto archive
    pareto_archive.py            # Pareto frontier management
  scripts/
    run_multi_objective_dgm.ps1  # Launcher script
  results/
    dgm_archive_multiobj_v7.jsonl       # Standard archive
    dgm_pareto_frontier_v7.jsonl        # Pareto-optimal solutions
    dgm_multiobj_v7_log.jsonl           # Iteration log with 10D objectives
```

### 3.2 Key Differences from v0.5

| Component         | v0.5 Optuna Adaptive              | v0.7 Multi-Objective               |
| ----------------- | --------------------------------- | ---------------------------------- |
| **Objectives**    | 1 (macro F1)                      | 10 (per-class accuracy)            |
| **Sampler**       | TPESampler                        | NSGAIISampler                      |
| **Archive**       | Best F1 solution                  | Pareto frontier (20+ solutions)    |
| **Acceptance**    | delta_f1 >= 0.001                 | Non-dominated only                 |
| **Output**        | Single best model                 | Multiple trade-off models          |
| **Visualization** | F1 trajectory                     | Pareto frontier plot (2D slices)   |
| **Iterations**    | 32 (standard)                     | 50-100 (more needed for 10D space) |

---

## 4. Expected Outcomes

### 4.1 Pareto Frontier Example

After 50 iterations, we expect a Pareto frontier like:

```
Solution A: [1.0, 0.73, 1.0, 1.0, 1.0, 0.92, 1.0, 1.0, 1.0, 1.0]  # Balanced
Solution B: [1.0, 0.60, 1.0, 1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0]  # Class 1 sacrificed
Solution C: [1.0, 0.80, 1.0, 1.0, 0.95, 0.88, 1.0, 1.0, 1.0, 1.0] # Classes 4,5 sacrificed
...
```

**Trade-off Analysis**:
- Solution A: Best for class 1, but class 5 slightly lower
- Solution B: Perfect on class 5, but class 1 suffers
- Solution C: Most balanced, slight loss on classes 4 and 5

User can select based on application requirements (e.g., if class 1 is critical, choose Solution C).

### 4.2 Comparison with v0.5

| Metric                      | v0.5 (Single-Objective)  | v0.7 (Multi-Objective)          |
| --------------------------- | ------------------------ | ------------------------------- |
| **Best macro F1**           | 0.9433                   | ~0.94 (estimated)               |
| **Worst class accuracy**    | 0.733 (class 1)          | > 0.80 (all classes, estimated) |
| **Number of solutions**     | 1                        | 20+ (Pareto frontier)           |
| **Class imbalance**         | High (0.733 vs 1.0)      | Low (<0.2 gap between classes)  |
| **Hypervolume**             | N/A                      | 0.85+ (estimated)               |
| **Iterations**              | 32                       | 50-100                          |

**Hypothesis**: v0.7 will produce more balanced models with **higher minimum class accuracy**, even if macro F1 is slightly lower than v0.5.

---

## 5. Experimental Protocol

### 5.1 Baseline: v0.5 Best Solution

```python
# v0.5 Best (F1=0.9433)
weights = [0.55, 0.76, 1.29, 1.22, 0.99, 0.84, 0.92, 1.26, 1.03, 1.12]
per_class_recall = [1.0, 0.733, 1.0, 1.0, 1.0, 0.917, 1.0, 1.0, 1.0, 1.0]
macro_f1 = 0.9433
```

### 5.2 v0.7 Experiment

**Settings**:
- Model: codestral:latest
- Iterations: 50 (conservative for 10D space)
- Population size: 20 (NSGA-II)
- Archive: Pareto frontier only (non-dominated solutions)
- Evaluation: 15-epoch LoRA (same as v0.5)

**Success Criteria**:
1. **Class balance**: min(per_class_recall) > 0.80
2. **Pareto frontier size**: 15-25 non-dominated solutions
3. **Hypervolume**: > 0.80
4. **Macro F1**: ≥ 0.93 (slight trade-off acceptable for balance)

---

## 6. Visualization

### 6.1 Pareto Frontier Scatter Plots

Since 10D is hard to visualize, we use **pairwise 2D projections**:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Example: Class 1 vs Class 5 (two classes with lowest recall in v0.5)
plt.scatter(
    [s[1][1] for s in pareto_frontier],  # Class 1 accuracy
    [s[1][5] for s in pareto_frontier],  # Class 5 accuracy
    c='blue', alpha=0.6
)
plt.xlabel("Class 1 Accuracy")
plt.ylabel("Class 5 Accuracy")
plt.title("Pareto Frontier: Class 1 vs Class 5 Trade-off")
plt.show()
```

### 6.2 Hypervolume Trajectory

```python
# Track hypervolume improvement over iterations
plt.plot(iterations, hypervolumes)
plt.xlabel("Iteration")
plt.ylabel("Hypervolume")
plt.title("Pareto Frontier Quality (Hypervolume Indicator)")
plt.show()
```

---

## 7. Risks and Mitigations

### 7.1 Risk: Curse of Dimensionality

**Risk**: 10-objective optimization is computationally expensive (50-100 iterations × 6.8 min = 5.6-11.3 hours)

**Mitigation**:
- Start with 50 iterations (conservative)
- Monitor hypervolume convergence; stop early if plateau reached
- Use dry-run mode to validate implementation before full run

### 7.2 Risk: Non-Convergence

**Risk**: NSGA-II may not converge in 10D space with limited iterations

**Mitigation**:
- Increase population size to 30 if needed
- Use warm-start from v0.5 best solution
- Fall back to single-objective (macro F1) if Pareto frontier quality is poor

### 7.3 Risk: Solution Selection Difficulty

**Risk**: 20+ Pareto-optimal solutions may overwhelm users

**Mitigation**:
- Provide **solution ranking** based on:
  1. Macro F1 (for direct comparison with v0.5)
  2. Min class accuracy (for balance-focused use cases)
  3. Hypervolume contribution (for diversity)
- Generate summary report with top 5 recommended solutions

---

## 8. Next Steps

1. **Implement** multi_objective_sampler.py (NSGA-II wrapper)
2. **Adapt** multi_objective_agent.py from v0.5's adapt_agent.py
3. **Create** multi_objective_loop.py with Pareto archive management
4. **Validate** with dry-run (7 iterations)
5. **Execute** full run (50 iterations)
6. **Analyze** Pareto frontier and compare with v0.5
7. **Document** results in RESULT_v07_MultiObjective.md

---

**Status**: Ready for implementation  
**Estimated Implementation Time**: 2-3 hours  
**Estimated Execution Time**: 5.6 hours (50 iterations × 6.8 min/iter)
