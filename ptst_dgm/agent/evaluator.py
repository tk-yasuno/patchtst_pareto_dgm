"""
evaluator.py
Subprocess-based evaluator for PatchTST DGM.

Calls train_patchtst_dgm.py in .venv-ptstf and parses the
output JSON to return 15 objective values.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

from ptst_dgm.agent.archive import OBJECTIVE_KEYS


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

    def evaluate(
        self,
        focal_alpha: float,
        focal_gamma: float,
        w_normal: float,
        w_anomal: float,
    ) -> Dict[str, float]:
        """Train PatchTST with given params and return 15 objective values."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            suffix=".json", dir=self.output_dir, delete=False
        ) as f:
            json_path = Path(f.name)

        cmd = [
            str(self.python_exe),
            str(self.script_path),
            "--focal-alpha", str(focal_alpha),
            "--focal-gamma", str(focal_gamma),
            "--w-normal", str(w_normal),
            "--w-anomal", str(w_anomal),
            "--epochs", str(self.epochs),
            "--data-path", str(self.data_path),
            "--output-dir", str(self.output_dir),
            "--output-json", str(json_path),
        ]

        print(f"[Evaluator] Running: focal_alpha={focal_alpha:.3f} gamma={focal_gamma:.3f} "
              f"w_normal={w_normal:.3f} w_anomal={w_anomal:.3f}")

        result = subprocess.run(cmd, capture_output=False, text=True)
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
