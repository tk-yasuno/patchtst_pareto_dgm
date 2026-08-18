"""
train_patchtst_dgm.py
PatchTST DGM training script with Focal Loss + Class Weights.

Fixed: LoRA r=16, alpha=32
Control variables: --focal-alpha, --focal-gamma, --w-normal, --w-anomal
Output: JSON with 15 metrics (AUC/Precision/Recall/F1/FPR x 30d/60d/90d)

Usage:
    python ptst_dgm/training/train_patchtst_dgm.py \
        --focal-alpha 0.5 --focal-gamma 2.0 \
        --w-normal 1.0 --w-anomal 3.0 \
        --output-json ptst_dgm/results/eval.json
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
LORA_R = 16
LORA_ALPHA = 32


class WeightedFocalLoss(nn.Module):
    """FocalLoss with per-sample class weights for binary anomaly detection."""

    def __init__(
        self,
        focal_alpha: float = 0.5,
        focal_gamma: float = 1.0,
        w_normal: float = 1.0,
        w_anomal: float = 1.0,
    ) -> None:
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="none")
        self.w_normal = w_normal
        self.w_anomal = w_anomal

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits/targets: [B, 3] for 30d/60d/90d horizons
        per_sample = self.focal(logits, targets)          # [B, 3]
        class_w = self.w_anomal * targets + self.w_normal * (1.0 - targets)
        return (class_w * per_sample).mean()


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
) -> dict:
    """Compute 15 metrics on test loader using thresholds found on val loader."""
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

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--w-normal",    type=float, default=1.0)
    parser.add_argument("--w-anomal",    type=float, default=1.0)
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--patience",    type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--data-path",   type=Path,
                        default=WORKSPACE_ROOT / "data" / "golden_testset")
    parser.add_argument("--output-dir",  type=Path,
                        default=WORKSPACE_ROOT / "ptst_dgm" / "results" / "temp_model")
    parser.add_argument("--output-json", type=Path, required=False)
    args = parser.parse_args()

    print(f"[Train] focal_alpha={args.focal_alpha}  focal_gamma={args.focal_gamma}  "
          f"w_normal={args.w_normal}  w_anomal={args.w_anomal}")
    print(f"[Train] LoRA r={LORA_R}, alpha={LORA_ALPHA}  epochs={args.epochs}  device={DEVICE}")

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
    lora_params = LoRAParams(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none")
    base = build_patch_tst(lora_params=lora_params)
    model = PatchTSTWrapper(base, dropout=0.1).to(DEVICE)

    # ── Loss / Optimizer ──────────────────────────────────────────────────
    criterion = WeightedFocalLoss(
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        w_normal=args.w_normal,
        w_anomal=args.w_anomal,
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
    metrics = _compute_all_metrics(model, test_loader, val_loader)

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


if __name__ == "__main__":
    main()
