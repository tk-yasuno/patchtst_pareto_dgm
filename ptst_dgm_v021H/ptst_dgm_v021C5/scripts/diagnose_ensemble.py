"""
Diagnose ensemble performance by evaluating each model individually
"""
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

def evaluate_model(pred_file: Path) -> dict:
    """Evaluate a single model's predictions"""
    data = np.load(pred_file)
    probs = data['probs']  # (N, 3)
    labels = data['labels']  # (N, 3)
    
    metrics = {}
    horizons = ['30d', '60d', '90d']
    
    for i, h in enumerate(horizons):
        y_true = labels[:, i].astype(int)
        y_prob = probs[:, i]
        y_pred = (y_prob > 0.5).astype(int)
        
        auc = roc_auc_score(y_true, y_prob)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        metrics[f'f1_{h}'] = f1
        metrics[f'fpr_{h}'] = fpr
        metrics[f'auc_{h}'] = auc
    
    metrics['macro_f1'] = np.mean([metrics[f'f1_{h}'] for h in horizons])
    metrics['mean_fpr'] = np.mean([metrics[f'fpr_{h}'] for h in horizons])
    
    return metrics

# Evaluate each model
for i in range(1, 4):
    pred_file = Path(f"ptst_dgm/results/ensemble_models/model_{i}/predictions.npz")
    if pred_file.exists():
        metrics = evaluate_model(pred_file)
        print(f"\nModel #{i}:")
        print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
        print(f"  Mean FPR:  {metrics['mean_fpr']:.4f}")
        print(f"  30d: F1={metrics['f1_30d']:.4f}, FPR={metrics['fpr_30d']:.4f}, AUC={metrics['auc_30d']:.4f}")
        print(f"  60d: F1={metrics['f1_60d']:.4f}, FPR={metrics['fpr_60d']:.4f}, AUC={metrics['auc_60d']:.4f}")
        print(f"  90d: F1={metrics['f1_90d']:.4f}, FPR={metrics['fpr_90d']:.4f}, AUC={metrics['auc_90d']:.4f}")

# Evaluate ensemble
print("\n" + "="*60)
print("Ensemble (soft voting):")
all_probs = []
for i in range(1, 4):
    pred_file = Path(f"ptst_dgm/results/ensemble_models/model_{i}/predictions.npz")
    data = np.load(pred_file)
    all_probs.append(data['probs'])

ensemble_probs = np.mean(all_probs, axis=0)
labels = np.load(Path("ptst_dgm/results/ensemble_models/model_1/predictions.npz"))['labels']

horizons = ['30d', '60d', '90d']
metrics = {}
for i, h in enumerate(horizons):
    y_true = labels[:, i].astype(int)
    y_prob = ensemble_probs[:, i]
    y_pred = (y_prob > 0.5).astype(int)
    
    auc = roc_auc_score(y_true, y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    metrics[f'f1_{h}'] = f1
    metrics[f'fpr_{h}'] = fpr

metrics['macro_f1'] = np.mean([metrics[f'f1_{h}'] for h in horizons])
metrics['mean_fpr'] = np.mean([metrics[f'fpr_{h}'] for h in horizons])

print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
print(f"  Mean FPR:  {metrics['mean_fpr']:.4f}")
