"""
ptst_agent.py
LLM-based validator for NSGA-II proposals in PatchTST DGM.

Flow:
  1. PatchTSTSampler.suggest() → NSGA-II proposes 4 parameters
  2. codestral:latest validates the proposal in anomaly detection context
  3. Returns validated (or NSGA-II raw) parameters
"""
from __future__ import annotations

import json
import re
from typing import Dict, Optional, Tuple

import requests

from ptst_dgm.agent.archive import PatchTSTAgentEntry, HORIZONS
from ptst_dgm.multi_objective_agent.ptst_sampler import PatchTSTSampler, PARAM_BOUNDS


class PatchTSTMultiObjectiveAgent:
    OLLAMA_API_URL = "http://localhost:11434/api/generate"

    def __init__(
        self,
        model_name: str = "codestral:latest",
        population_size: int = 20,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.sampler = PatchTSTSampler(
            study_name=f"ptst_dgm_{seed}",
            seed=seed,
            population_size=population_size,
        )

    def generate(
        self,
        parent: PatchTSTAgentEntry,
        max_retries: int = 2,
    ) -> Tuple[Dict, int]:
        """Return (proposal_dict, trial_number). Always succeeds via NSGA-II fallback."""
        trial_number, nsgaii_params = self.sampler.suggest()
        print(f"[PatchTSTAgent] [TOOL: get_nsgaii_suggestion] trial=#{trial_number}  "
              f"patch_len={nsgaii_params['patch_len']}  "
              f"stride={nsgaii_params['stride']}")

        self._preload_model()
        result = None
        try:
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.OLLAMA_API_URL,
                        json={
                            "model": self.model_name,
                            "prompt": self._build_prompt(parent, nsgaii_params),
                            "stream": False,
                            "options": {"temperature": 0.3, "top_p": 0.9},
                        },
                        timeout=300,
                    )
                    response.raise_for_status()
                    text = response.json().get("response", "")
                    parsed = self._parse_json(text)
                    if parsed and self._validate(parsed):
                        result = parsed
                        break
                    print(f"[PatchTSTAgent] Parse/validate failed (attempt {attempt+1}) — using NSGA-II raw")
                except Exception as e:
                    print(f"[PatchTSTAgent] LLM error: {e}")
        finally:
            self._unload_model()

        if result is None:
            result = {**nsgaii_params, "rationale": f"NSGA-II trial #{trial_number} (LLM fallback)"}

        return result, trial_number

    def tell_result(self, trial_number: int, objectives: Dict[str, float]) -> None:
        self.sampler.tell(trial_number, objectives)
        mean_f1  = sum(objectives[f"f1_{h}d"]  for h in HORIZONS) / 3
        mean_fpr = sum(objectives[f"fpr_{h}d"] for h in HORIZONS) / 3
        print(f"[PatchTSTAgent] [TOOL: tell_nsgaii_result] trial=#{trial_number}  "
              f"mean_F1={mean_f1:.4f}  mean_FPR={mean_fpr:.4f}")
        print(f"[PatchTSTAgent] {self.sampler.summary()}")

    def _build_prompt(
        self, parent: PatchTSTAgentEntry, nsgaii: Dict[str, float]
    ) -> str:
        objs = parent.objectives
        parent_str = "\n".join(
            f"  {h}d | AUC={objs.get(f'auc_{h}d', 0):.4f}  "
            f"P={objs.get(f'precision_{h}d', 0):.4f}  "
            f"R={objs.get(f'recall_{h}d', 0):.4f}  "
            f"F1={objs.get(f'f1_{h}d', 0):.4f}  "
            f"FPR={objs.get(f'fpr_{h}d', 0):.4f}"
            for h in HORIZONS
        )
        return f"""You are an ML expert validating an Optuna NSGA-II suggestion for
PatchTST architecture optimization (v0.3: patch_len/stride tuning).

=== TOOL CALL RESULT ===
Tool: get_nsgaii_suggestion (NSGA-II Pareto 15-Objective Optimizer)
Suggested architecture parameters:
  patch_len = {nsgaii['patch_len']}  (range 8–32, integer)
  stride    = {nsgaii['stride']}  (range 4–24, integer)
========================

Context (90-day time series → 3 horizons):
  - patch_len: Length of each patch extracted from time series
    - Smaller (8-12): More granular features, more patches → higher model capacity but risk of overfitting
    - Larger (24-32): More context per patch, fewer patches → lower capacity but more robust
  - stride: Step size between consecutive patches
    - Smaller (4-8): High overlap, redundant features → better local details but slower training
    - Larger (16-24): Low overlap, less redundancy → faster training but may miss transitions

Objectives: Maximize AUC/Precision/Recall/F1 × 3 horizons, Minimize FPR × 3 horizons.

Current best parent (macro_F1={parent.macro_f1:.4f}):
  patch_len={parent.patch_len}  stride={parent.stride}
{parent_str}

Focal Loss FIXED (v0.2 best): alpha=0.866, gamma=1.156, w_normal=1.851, w_anomal=4.035

YOUR TASK:
1. Evaluate if NSGA-II suggestion is physically reasonable for time series patching:
   - patch_len ≥ stride (required: no gaps between patches)
   - stride ≤ patch_len/2 often optimal (50%+ overlap for smooth transitions)
   - For 90-day sequences: ~90/stride patches generated
2. Accept NSGA-II's Pareto suggestion if it looks reasonable
3. Micro-correct (±2 max) only if clearly outside sensible range (e.g., stride > patch_len)
4. Trust NSGA-II — it explores Pareto trade-offs across all 15 objectives

Output (JSON only, no other text):
{{"patch_len":<int>,"stride":<int>,"rationale":"<one sentence>"}}"""

    def _parse_json(self, text: str) -> Optional[Dict]:
        m = re.search(r'\{[^{}]*"patch_len"[^{}]*\}', text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except Exception:
            return None

    def _validate(self, cfg: Dict) -> bool:
        required = {"patch_len", "stride", "rationale"}
        if not required.issubset(cfg):
            return False
        for name, (lo, hi) in PARAM_BOUNDS.items():
            v = cfg.get(name)
            if not isinstance(v, (int, float)) or not (lo <= v <= hi):
                return False
        return True

    def _preload_model(self) -> None:
        """Preload model into Ollama GPU memory before inference."""
        print(f"[PatchTSTAgent] Loading {self.model_name} to GPU...")
        try:
            requests.post(
                self.OLLAMA_API_URL,
                json={"model": self.model_name, "prompt": "", "stream": False},
                timeout=30,
            )
            print(f"[PatchTSTAgent] ✓ Model loaded")
        except Exception as e:
            print(f"[PatchTSTAgent] Preload warning: {e}")

    def _unload_model(self) -> None:
        """Unload model from Ollama GPU memory after inference (free VRAM for training)."""
        print(f"[PatchTSTAgent] Unloading {self.model_name} from GPU...")
        try:
            requests.post(
                self.OLLAMA_API_URL,
                json={"model": self.model_name, "prompt": "", "keep_alive": 0},
                timeout=10,
            )
            print(f"[PatchTSTAgent] ✓ Model unloaded (VRAM freed for PatchTST training)")
        except Exception as e:
            print(f"[PatchTSTAgent] Unload warning: {e}")
