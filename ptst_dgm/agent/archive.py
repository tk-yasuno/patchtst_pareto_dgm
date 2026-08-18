"""
archive.py
JSONL-based archive for PatchTST DGM agents.

Each entry stores 4 Focal Loss / class-weight parameters
and 15 evaluation objectives (5 metrics × 3 horizons).
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


@dataclass
class PatchTSTAgentEntry:
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    coding_model: str = "baseline"

    # 4 control variables
    focal_alpha: float = 0.5
    focal_gamma: float = 1.0
    w_normal: float = 1.0
    w_anomal: float = 1.0

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
        macro_f1 = sum(objectives[f"f1_{h}d"] for h in HORIZONS) / 3
        return cls(
            parent_id=None,
            coding_model="baseline",
            focal_alpha=0.5,
            focal_gamma=1.0,
            w_normal=1.0,
            w_anomal=1.0,
            objectives=objectives,
            macro_f1=macro_f1,
            rationale="Baseline: FocalLoss(alpha=0.5, gamma=1.0), equal class weights",
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
        objectives: Dict[str, float],
        rationale: str,
    ) -> "PatchTSTAgentEntry":
        macro_f1 = sum(objectives[f"f1_{h}d"] for h in HORIZONS) / 3
        return cls(
            parent_id=parent_id,
            coding_model=coding_model,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            w_normal=w_normal,
            w_anomal=w_anomal,
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
