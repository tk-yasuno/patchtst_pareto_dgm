"""
train_patchtst_dgm.py
PatchTST DGM training script with Horizon-specific 9D Optimization (v0.2.1H).

Control variables (v0.2.1H):
  - Focal Loss (per-horizon): --focal-alpha-XXd, --focal-gamma-XXd, --w-normal-XXd (constraint: w_normal + w_anomal = 5.88 per horizon)
  - Architecture: --patch-len, --stride (FIXED at v0.2.3 best: 26, 16)
  - LoRA: --lora-rank, --lora-alpha (FIXED at v0.2.3 best: 16, 47)
Output: JSON with 15 metrics (AUC/Precision/Recall/F1/FPR x 30d/60d/90d)

Usage:
    python ptst_dgm_v021H/training/train_patchtst_dgm.py \
        --focal-alpha-30d 0.75 --focal-gamma-30d 1.0 --w-normal-30d 1.0 --w-anomal-30d 1.0 \
        --focal-alpha-60d 0.75 --focal-gamma-60d 1.0 --w-normal-60d 1.0 --w-anomal-60d 1.0 \
        --focal-alpha-90d 0.75 --focal-gamma-90d 1.0 --w-normal-90d 1.0 --w-anomal-90d 1.0 \
        --patch-len 26 --stride 16 \
        --lora-rank 16 --lora-alpha 47 \
        --output-json ptst_dgm_v021H/results/eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)

# Resolve paths relative to workspace root (two levels up from this file)
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BASELINE_SRC = WORKSPACE_ROOT / "0_LogBAK" / "v4-1-3_tst" / "src"
sys.path.insert(0, str(WORKSPACE_ROOT / "0_LogBAK" / "v4-1-3_tst"))

from src.losses.focal_loss import FocalLoss
from src.models.lora_config import LoRAParams
from src.models.patch_tst_lora import build_patch_tst, PatchTSTWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HORIZONS = [30, 60, 90]

# v0.4+v0.5: All parameters now as command-line args (no hardcoded values)


class WeightedFocalLoss(nn.Module):
    """Horizon-specific FocalLoss with per-sample class weights for binary anomaly detection (v0.2.1H)."""

    def __init__(
        self,
        focal_alpha_30d: float, focal_gamma_30d: float, w_normal_30d: float, w_anomal_30d: float,
        focal_alpha_60d: float, focal_gamma_60d: float, w_normal_60d: float, w_anomal_60d: float,
        focal_alpha_90d: float, focal_gamma_90d: float, w_normal_90d: float, w_anomal_90d: float,
    ) -> None:
        super().__init__()
        # Create 3 separate Focal Loss instances (one per horizon)
        self.focal_30d = FocalLoss(alpha=focal_alpha_30d, gamma=focal_gamma_30d, reduction="none")
        self.focal_60d = FocalLoss(alpha=focal_alpha_60d, gamma=focal_gamma_60d, reduction="none")
        self.focal_90d = FocalLoss(alpha=focal_alpha_90d, gamma=focal_gamma_90d, reduction="none")
        # Class weights per horizon
        self.w_normal_30d, self.w_anomal_30d = w_normal_30d, w_anomal_30d
        self.w_normal_60d, self.w_anomal_60d = w_normal_60d, w_anomal_60d
        self.w_normal_90d, self.w_anomal_90d = w_normal_90d, w_anomal_90d

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits/targets: [B, 3] for 30d/60d/90d horizons
        # Compute per-horizon losses
        loss_30d = self.focal_30d(logits[:, 0:1], targets[:, 0:1])  # [B, 1]
        loss_60d = self.focal_60d(logits[:, 1:2], targets[:, 1:2])  # [B, 1]
        loss_90d = self.focal_90d(logits[:, 2:3], targets[:, 2:3])  # [B, 1]
        
        # Apply per-horizon class weights
        class_w_30d = self.w_anomal_30d * targets[:, 0:1] + self.w_normal_30d * (1.0 - targets[:, 0:1])
        class_w_60d = self.w_anomal_60d * targets[:, 1:2] + self.w_normal_60d * (1.0 - targets[:, 1:2])
        class_w_90d = self.w_anomal_90d * targets[:, 2:3] + self.w_normal_90d * (1.0 - targets[:, 2:3])
        
        # Weighted losses per horizon
        weighted_30d = (class_w_30d * loss_30d).mean()
        weighted_60d = (class_w_60d * loss_60d).mean()
        weighted_90d = (class_w_90d * loss_90d).mean()
        
        # Return average across horizons
        return (weighted_30d + weighted_60d + weighted_90d) / 3.0


class GoldenDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.seqs = np.stack(df["value_norm_seq"].to_numpy()).astype(np.float32)
        self.labels = df[[f"label_{h}d" for h in HORIZONS]].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.seqs[idx]), torch.from_numpy(self.labels[idx])


def _train_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def _eval_auc(model, loader):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy())
        labels.append(y.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    aucs = []
    for i in range(labels.shape[1]):
        try:
            aucs.append(roc_auc_score(labels[:, i], probs[:, i]))
        except Exception:
            aucs.append(0.5)
    return float(np.mean(aucs))


def _find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Grid-search threshold on [0.1, 0.9] that maximises F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.91, 0.01):
        pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


@torch.no_grad()
def _compute_all_metrics(
    model, loader, val_loader
) -> tuple:
    """Compute 15 metrics on test loader using thresholds found on val loader.
    
    Returns:
        metrics: dict with 15 metrics
        test_probs: (N, 3) array of test prediction probabilities
        test_labels: (N, 3) array of test ground truth labels
    """
    model.eval()

    def _collect(ld):
        ps, ls = [], []
        for x, y in ld:
            ps.append(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy())
            ls.append(y.numpy())
        return np.concatenate(ps), np.concatenate(ls)

    val_probs, val_labels = _collect(val_loader)
    test_probs, test_labels = _collect(loader)

    results = {}
    for i, h in enumerate(HORIZONS):
        yt_val = val_labels[:, i].astype(int)
        yt_test = test_labels[:, i].astype(int)
        yp_test = test_probs[:, i]

        # Threshold found on val set
        thresh = _find_best_threshold(yt_val, val_probs[:, i])
        pred = (yp_test >= thresh).astype(int)

        try:
            auc = roc_auc_score(yt_test, yp_test)
        except Exception:
            auc = 0.5

        prec = precision_score(yt_test, pred, zero_division=0)
        rec  = recall_score(yt_test, pred, zero_division=0)
        f1   = f1_score(yt_test, pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(yt_test, pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        results[f"auc_{h}d"]       = round(auc, 6)
        results[f"precision_{h}d"] = round(prec, 6)
        results[f"recall_{h}d"]    = round(rec, 6)
        results[f"f1_{h}d"]        = round(f1, 6)
        results[f"fpr_{h}d"]       = round(fpr, 6)

    return results, test_probs, test_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    # v0.2.1H: Horizon-specific 9D control variables
    # 30d horizon
    parser.add_argument("--focal-alpha-30d", type=float, default=0.75)
    parser.add_argument("--focal-gamma-30d", type=float, default=1.0)
    parser.add_argument("--w-normal-30d", type=float, default=1.0)
    parser.add_argument("--w-anomal-30d", type=float, default=1.0)
    # 60d horizon
    parser.add_argument("--focal-alpha-60d", type=float, default=0.75)
    parser.add_argument("--focal-gamma-60d", type=float, default=1.0)
    parser.add_argument("--w-normal-60d", type=float, default=1.0)
    parser.add_argument("--w-anomal-60d", type=float, default=1.0)
    # 90d horizon
    parser.add_argument("--focal-alpha-90d", type=float, default=0.75)
    parser.add_argument("--focal-gamma-90d", type=float, default=1.0)
    parser.add_argument("--w-normal-90d", type=float, default=1.0)
    parser.add_argument("--w-anomal-90d", type=float, default=1.0)
    # Architecture and LoRA (FIXED at v0.2.3 best)
    parser.add_argument("--patch-len",    type=int,   default=26)
    parser.add_argument("--stride",       type=int,   default=16)
    parser.add_argument("--lora-rank",    type=int,   default=16)
    parser.add_argument("--lora-alpha",   type=int,   default=47)
    # Training parameters
    parser.add_argument("--seed",         type=int,   default=None,
                        help="Random seed for reproducibility (PyTorch, NumPy, CUDA)")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--patience",     type=int,   default=20)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--data-path",    type=Path,
                        default=WORKSPACE_ROOT / "data" / "golden_testset")
    parser.add_argument("--output-dir",   type=Path,
                        default=WORKSPACE_ROOT / "ptst_dgm" / "results" / "temp_model")
    parser.add_argument("--output-json",  type=Path, required=False)
    parser.add_argument("--output-predictions", action="store_true",
                        help="Save test predictions (probs and labels) as npz file")
    args = parser.parse_args()

    # Set random seed for reproducibility (v0.3.5 + deterministic mode)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        # Enable deterministic mode for full reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[Train] Seed: {args.seed} (deterministic mode enabled)")
    else:
        print(f"[Train] Seed: None (random initialization)")

    print(f"[Train] Horizon-specific Focal Loss:")
    print(f"  30d: α={args.focal_alpha_30d:.3f}  γ={args.focal_gamma_30d:.3f}  "
          f"w_n={args.w_normal_30d:.3f}  w_a={args.w_anomal_30d:.3f}")
    print(f"  60d: α={args.focal_alpha_60d:.3f}  γ={args.focal_gamma_60d:.3f}  "
          f"w_n={args.w_normal_60d:.3f}  w_a={args.w_anomal_60d:.3f}")
    print(f"  90d: α={args.focal_alpha_90d:.3f}  γ={args.focal_gamma_90d:.3f}  "
          f"w_n={args.w_normal_90d:.3f}  w_a={args.w_anomal_90d:.3f}")
    print(f"[Train] Architecture: patch_len={args.patch_len}  stride={args.stride}")
    print(f"[Train] LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"[Train] Epochs={args.epochs}  device={DEVICE}")

    # ── Data ──────────────────────────────────────────────────────────────
    parquet = args.data_path / "golden_testset.parquet"
    df = pd.read_parquet(parquet)

    # 70% train / 15% val / 15% test (fixed seed for reproducibility)
    train_df, tmp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label_30d"]
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.50, random_state=42, stratify=tmp_df["label_30d"]
    )
    print(f"[Train] Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_loader = DataLoader(GoldenDataset(train_df), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(GoldenDataset(val_df),   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(GoldenDataset(test_df),  batch_size=args.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    lora_params = LoRAParams(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none"
    )
    base = build_patch_tst(
        context_len=90,
        patch_len=args.patch_len,
        stride=args.stride,
        lora_params=lora_params
    )
    model = PatchTSTWrapper(base, dropout=0.1).to(DEVICE)

    # ── Loss / Optimizer ──────────────────────────────────────────────────
    criterion = WeightedFocalLoss(
        focal_alpha_30d=args.focal_alpha_30d, focal_gamma_30d=args.focal_gamma_30d,
        w_normal_30d=args.w_normal_30d, w_anomal_30d=args.w_anomal_30d,
        focal_alpha_60d=args.focal_alpha_60d, focal_gamma_60d=args.focal_gamma_60d,
        w_normal_60d=args.w_normal_60d, w_anomal_60d=args.w_anomal_60d,
        focal_alpha_90d=args.focal_alpha_90d, focal_gamma_90d=args.focal_gamma_90d,
        w_normal_90d=args.w_normal_90d, w_anomal_90d=args.w_anomal_90d,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_auc = 0.0
    patience_ctr = 0
    ckpt_path = args.output_dir / "best_ptst_dgm.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer)
        val_auc = _eval_auc(model, val_loader)
        scheduler.step(val_auc)

        print(f"Epoch {epoch:3d} | loss={train_loss:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Final evaluation ──────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    metrics, test_probs, test_labels = _compute_all_metrics(model, test_loader, val_loader)

    print("\n[Eval] 15-metric results:")
    for h in HORIZONS:
        print(f"  {h}d | AUC={metrics[f'auc_{h}d']:.4f}  "
              f"P={metrics[f'precision_{h}d']:.4f}  "
              f"R={metrics[f'recall_{h}d']:.4f}  "
              f"F1={metrics[f'f1_{h}d']:.4f}  "
              f"FPR={metrics[f'fpr_{h}d']:.4f}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[Eval] Saved → {args.output_json}")
    
    if args.output_predictions:
        pred_file = args.output_dir / "predictions.npz"
        pred_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(pred_file, probs=test_probs, labels=test_labels)
        print(f"predictions_saved: {pred_file}")


if __name__ == "__main__":
    main()
