"""
archive.py
JSONL-based archive for PatchTST DGM agents.

v0.3: Architecture parameters (patch_len, stride) as control variables
      Focal Loss parameters fixed (v0.2 best solution)
      
Each entry stores 2 architecture parameters and 15 evaluation objectives
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

# v0.3: Fixed Focal Loss parameters (v0.2 best - Trial #381)
FIXED_FOCAL_ALPHA = 0.866
FIXED_FOCAL_GAMMA = 1.156
FIXED_W_NORMAL = 1.851
FIXED_W_ANOMAL = 4.035


@dataclass
class PatchTSTAgentEntry:
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    coding_model: str = "baseline"

    # v0.3: Architecture control variables
    patch_len: int = 16
    stride: int = 8

    # v0.1/v0.2: Focal Loss control variables (kept for backward compatibility)
    focal_alpha: Optional[float] = None
    focal_gamma: Optional[float] = None
    w_normal: Optional[float] = None
    w_anomal: Optional[float] = None

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
        """Create baseline entry with default architecture params."""
        macro_f1 = sum(objectives[f"f1_{h}d"] for h in HORIZONS) / 3
        return cls(
            parent_id=None,
            coding_model="baseline",
            patch_len=16,
            stride=8,
            objectives=objectives,
            macro_f1=macro_f1,
            rationale="Baseline: patch_len=16, stride=8",
        )

    @classmethod
    def create_child(
        cls,
        parent_id: str,
        coding_model: str,
        patch_len: int,
        stride: int,
        objectives: Dict[str, float],
        rationale: str,
    ) -> "PatchTSTAgentEntry":
        """Create child entry with architecture params."""
        macro_f1 = sum(objectives[f"f1_{h}d"] for h in HORIZONS) / 3
        return cls(
            parent_id=parent_id,
            coding_model=coding_model,
            patch_len=patch_len,
            stride=stride,
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
