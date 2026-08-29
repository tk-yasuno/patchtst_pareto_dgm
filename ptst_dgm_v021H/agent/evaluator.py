"""
evaluator.py
Subprocess-based evaluator for PatchTST DGM (v0.4+v0.5).

Calls train_patchtst_dgm.py in .venv-ptstf with:
- Focal Loss parameters (focal_alpha, focal_gamma, w_normal, w_anomal)
- Architecture parameters (patch_len, stride)
- LoRA parameters (lora_rank, lora_alpha)

Returns 15 objective values (5 metrics × 3 horizons).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

from ptst_dgm_v021C5.agent.archive import OBJECTIVE_KEYS


class PatchTSTEvaluator:
    def __init__(
        self,
        python_exe: Path,
        script_path: Path,
        data_path: Path,
        output_dir: Path,
        epochs: int = 100,
    ) -> None:
        self.python_exe = python_exe
        self.script_path = script_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.epochs = epochs
        # Workspace root for cwd (2 levels up from this file)
        self.workspace_root = Path(__file__).resolve().parents[3]

    def evaluate(
        self,
        focal_alpha_30d: float, focal_gamma_30d: float, w_normal_30d: float, w_anomal_30d: float,
        focal_alpha_60d: float, focal_gamma_60d: float, w_normal_60d: float, w_anomal_60d: float,
        focal_alpha_90d: float, focal_gamma_90d: float, w_normal_90d: float, w_anomal_90d: float,
        patch_len: int,
        stride: int,
        lora_rank: int,
        lora_alpha: int,
        seed: int = None,
    ) -> Dict[str, float]:
        """Train PatchTST with given 9D horizon-specific params + seed and return 15 objective values."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            suffix=".json", dir=self.output_dir, delete=False
        ) as f:
            json_path = Path(f.name)

        cmd = [
            str(self.python_exe),
            str(self.script_path),
            "--focal-alpha-30d", str(focal_alpha_30d),
            "--focal-gamma-30d", str(focal_gamma_30d),
            "--w-normal-30d", str(w_normal_30d),
            "--w-anomal-30d", str(w_anomal_30d),
            "--focal-alpha-60d", str(focal_alpha_60d),
            "--focal-gamma-60d", str(focal_gamma_60d),
            "--w-normal-60d", str(w_normal_60d),
            "--w-anomal-60d", str(w_anomal_60d),
            "--focal-alpha-90d", str(focal_alpha_90d),
            "--focal-gamma-90d", str(focal_gamma_90d),
            "--w-normal-90d", str(w_normal_90d),
            "--w-anomal-90d", str(w_anomal_90d),
            "--patch-len", str(patch_len),
            "--stride", str(stride),
            "--lora-rank", str(lora_rank),
            "--lora-alpha", str(lora_alpha),
            "--epochs", str(self.epochs),
            "--data-path", str(self.data_path),
            "--output-dir", str(self.output_dir),
            "--output-json", str(json_path),
        ]
        
        # Add seed if provided (v0.3.6+)
        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        seed_info = f"seed={seed}" if seed is not None else "seed=random"
        print(f"[Evaluator] Running horizon-specific:")
        print(f"  30d: α={focal_alpha_30d:.3f} γ={focal_gamma_30d:.3f} w_n={w_normal_30d:.3f} w_a={w_anomal_30d:.3f}")
        print(f"  60d: α={focal_alpha_60d:.3f} γ={focal_gamma_60d:.3f} w_n={w_normal_60d:.3f} w_a={w_anomal_60d:.3f}")
        print(f"  90d: α={focal_alpha_90d:.3f} γ={focal_gamma_90d:.3f} w_n={w_normal_90d:.3f} w_a={w_anomal_90d:.3f}")
        print(f"  Architecture: patch={patch_len} stride={stride}  LoRA: r={lora_rank} α={lora_alpha}  {seed_info}")

        result = subprocess.run(cmd, capture_output=False, text=True, cwd=self.workspace_root)
        if result.returncode != 0:
            print(f"[Evaluator] Training failed (exit={result.returncode}), returning baseline-like zeros")
            return {k: 0.0 for k in OBJECTIVE_KEYS}

        if not json_path.exists():
            print("[Evaluator] Output JSON missing, returning zeros")
            return {k: 0.0 for k in OBJECTIVE_KEYS}

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        json_path.unlink(missing_ok=True)

        # Validate all 15 keys are present
        objectives = {}
        for key in OBJECTIVE_KEYS:
            objectives[key] = float(data.get(key, 0.0))

        print(f"[Evaluator] AUC: 30d={objectives['auc_30d']:.4f}  "
              f"60d={objectives['auc_60d']:.4f}  90d={objectives['auc_90d']:.4f}")
        print(f"[Evaluator] F1:  30d={objectives['f1_30d']:.4f}  "
              f"60d={objectives['f1_60d']:.4f}  90d={objectives['f1_90d']:.4f}")
        print(f"[Evaluator] FPR: 30d={objectives['fpr_30d']:.4f}  "
              f"60d={objectives['fpr_60d']:.4f}  90d={objectives['fpr_90d']:.4f}")

        return objectives
