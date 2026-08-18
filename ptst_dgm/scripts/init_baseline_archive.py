"""
init_baseline_archive.py
Create the initial JSONL archive from a baseline evaluation JSON.

Usage:
    python ptst_dgm/scripts/init_baseline_archive.py \
        --eval-json  ptst_dgm/results/baseline_eval.json \
        --archive    ptst_dgm/results/ptst_archive.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

from ptst_dgm.agent.archive import PatchTSTAgentEntry, PatchTSTArchive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--archive",   type=Path, required=True)
    args = parser.parse_args()

    with open(args.eval_json, encoding="utf-8") as f:
        objectives = json.load(f)

    entry = PatchTSTAgentEntry.create_baseline(objectives)

    archive = PatchTSTArchive(args.archive)
    if len(archive) > 0:
        print(f"[InitArchive] Archive already exists with {len(archive)} entries — skipping")
        return

    archive.add(entry)
    print(f"[InitArchive] Baseline entry added: id={entry.id[:8]}  macro_F1={entry.macro_f1:.4f}")
    print(f"[InitArchive] Archive saved → {args.archive}")


if __name__ == "__main__":
    main()
