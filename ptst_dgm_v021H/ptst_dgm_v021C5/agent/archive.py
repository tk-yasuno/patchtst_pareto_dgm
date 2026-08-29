"""
archive.py
JSONL-based archive for PatchTST DGM agents.

v0.2.1C: Loss function optimization with constraint
         Architecture + LoRA fixed at v0.2.3 best
      
Each entry stores 8 parameters and 15 evaluation objectives
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


HORIZONS = [30, 60, 90]
METRICS = ["auc", "precision", "recall", "f1", "fpr"]

# All 15 objective keys in canonical order
OBJECTIVE_KEYS: List[str] = [
    f"{m}_{h}d" for m in ["auc", "precision", "recall", "f1"] for h in HORIZONS
] + [f"fpr_{h}d" for h in HORIZONS]

# v0.2.1C: Baseline defaults (simple Focal Loss + v0.2.3 Architecture/LoRA)
DEFAULT_FOCAL_ALPHA = 0.75   # Baseline starting point
DEFAULT_FOCAL_GAMMA = 1.0
DEFAULT_W_NORMAL = 1.0
DEFAULT_W_ANOMAL = 1.0
DEFAULT_PATCH_LEN = 26       # v0.2.3 best (from 8D optimization)
DEFAULT_STRIDE = 16
DEFAULT_LORA_RANK = 16       # v0.2.3 best (from 8D optimization)
DEFAULT_LORA_ALPHA = 47      # v0.2.3 best (from 8D optimization)


@dataclass
class PatchTSTAgentEntry:
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    coding_model: str = "baseline"

    # v0.4+v0.5: All 8 control variables (joint optimization)
    focal_alpha: float = DEFAULT_FOCAL_ALPHA
    focal_gamma: float = DEFAULT_FOCAL_GAMMA
    w_normal: float = DEFAULT_W_NORMAL
    w_anomal: float = DEFAULT_W_ANOMAL
    patch_len: int = DEFAULT_PATCH_LEN
    stride: int = DEFAULT_STRIDE
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA

    # 15 objective values: auc×3, precision×3, recall×3, f1×3, fpr×3
    objectives: Dict[str, float] = field(default_factory=dict)

    # Summary scalar used for archive sorting (mean F1 across horizons)
    macro_f1: float = 0.0

    rationale: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_dict(cls, d: dict) -> "PatchTSTAgentEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def create_baseline(cls, objectives: Dict[str, float]) -> "PatchTSTAgentEntry":
        """Create baseline entry (v0.2.1C: default Focal Loss + v0.2.3 best Architecture/LoRA)."""
        macro_f1 = sum(objectives[f"f1_{h}d"] for h in HORIZONS) / 3
        return cls(
            parent_id=None,
            coding_model="baseline",
            focal_alpha=DEFAULT_FOCAL_ALPHA,
            focal_gamma=DEFAULT_FOCAL_GAMMA,
            w_normal=DEFAULT_W_NORMAL,
            w_anomal=DEFAULT_W_ANOMAL,
            patch_len=DEFAULT_PATCH_LEN,
            stride=DEFAULT_STRIDE,
            lora_rank=DEFAULT_LORA_RANK,
            lora_alpha=DEFAULT_LORA_ALPHA,
            objectives=objectives,
            macro_f1=macro_f1,
            rationale="Baseline: default Focal Loss + v0.2.3 Architecture/LoRA",
        )

    @classmethod
    def create_child(
        cls,
        parent_id: str,
        coding_model: str,
        focal_alpha: float,
        focal_gamma: float,
        w_normal: float,
        w_anomal: float,
        patch_len: int,
        stride: int,
        lora_rank: int,
        lora_alpha: int,
        objectives: Dict[str, float],
        rationale: str,
    ) -> "PatchTSTAgentEntry":
        """Create child entry (v0.2.1C: Loss function optimization with constraint)."""
        macro_f1 = sum(objectives[f"f1_{h}d"] for h in HORIZONS) / 3
        return cls(
            parent_id=parent_id,
            coding_model=coding_model,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            w_normal=w_normal,
            w_anomal=w_anomal,
            patch_len=patch_len,
            stride=stride,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            objectives=objectives,
            macro_f1=macro_f1,
            rationale=rationale,
        )


class PatchTSTArchive:
    """JSONL-backed archive with incremental persistence."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: List[PatchTSTAgentEntry] = []
        if path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(PatchTSTAgentEntry.from_dict(json.loads(line)))

    def add(self, entry: PatchTSTAgentEntry) -> None:
        self.entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def get_best(self) -> PatchTSTAgentEntry:
        if not self.entries:
            raise RuntimeError("Archive is empty")
        return max(self.entries, key=lambda e: e.macro_f1)

    def __len__(self) -> int:
        return len(self.entries)
