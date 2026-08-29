"""
ptst_agent.py
LLM-based validator for NSGA-II proposals in PatchTST DGM (v0.2.1C5).

Flow:
  1. PatchTSTSampler.suggest() → NSGA-II proposes 3 parameters (Focal Loss with relaxed constraint)
  2. codestral:latest validates the proposal in Focal Loss context
  3. Returns validated (or NSGA-II raw) parameters
  4. Architecture fixed at v0.2.3 best (patch_len=26, stride=16)
  5. LoRA fixed at v0.2.3 best (rank=16, alpha=47)
  6. Constraint: w_normal + w_anomal = 5.88 (v0.2 best sum)
  7. Seed: None (random init for diversity)
"""
from __future__ import annotations

import json
import re
from typing import Dict, Optional, Tuple

import requests

from ptst_dgm_v021C5.agent.archive import PatchTSTAgentEntry, HORIZONS
from ptst_dgm_v021C5.multi_objective_agent.ptst_sampler import PatchTSTSampler, PARAM_BOUNDS


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
        seed_info = f" seed={nsgaii_params.get('seed')}" if nsgaii_params.get('seed') is not None else ""
        print(f"[PatchTSTAgent] [TOOL: get_nsgaii_suggestion] trial=#{trial_number}")
        print(f"  Architecture: patch={nsgaii_params['patch_len']} stride={nsgaii_params['stride']} (Discovered at v0.2.3){seed_info}")
        print(f"  LoRA: rank={nsgaii_params['lora_rank']} alpha={nsgaii_params['lora_alpha']} (Discovered at v0.2.3)")
        print(f"  Focal Loss: α={nsgaii_params['focal_alpha']:.3f} γ={nsgaii_params['focal_gamma']:.3f} "
              f"w_n={nsgaii_params['w_normal']:.3f} w_a={nsgaii_params['w_anomal']:.3f} (constraint: w_n+w_a=5.88)")

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
                        # Add fixed Architecture, LoRA, and seed=None to LLM result (v0.2.1C5)
                        from ptst_dgm_v021C.multi_objective_agent.ptst_sampler import FIXED_ARCH_PARAMS, FIXED_LORA_PARAMS
                        result = {**parsed, **FIXED_ARCH_PARAMS, **FIXED_LORA_PARAMS, "seed": None}
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
Focal Loss optimization in PatchTST (v0.2.1C5).

=== TOOL CALL RESULT ===
Tool: get_nsgaii_suggestion (NSGA-II Pareto 15-Objective Optimizer)
Suggested Focal Loss parameters (proposal):
  focal_alpha={nsgaii['focal_alpha']:.3f}, focal_gamma={nsgaii['focal_gamma']:.3f}
  w_normal={nsgaii['w_normal']:.3f}, w_anomal={nsgaii['w_anomal']:.3f} (constraint: w_normal + w_anomal = 5.88)
  Architecture: FIXED v0.2.3 best (patch_len={nsgaii['patch_len']}, stride={nsgaii['stride']})
  LoRA: FIXED v0.2.3 best (rank={nsgaii['lora_rank']}, alpha={nsgaii['lora_alpha']})
  Seed: None (random init for diversity)
========================

Context (Focal Loss for imbalanced anomaly detection - v0.2.1C5 with relaxed constraint):
  - focal_alpha [0.5-0.9]: Weighting factor for hard/easy examples
    * Lower → focus on easy examples → safer but may ignore hard cases
    * Higher → focus on hard examples → aggressive learning but unstable
  - focal_gamma [0.6-2.0]: Modulation factor controlling focus strength
    * Lower → weaker modulation → closer to standard cross-entropy
    * Higher → stronger modulation → heavily downweight easy examples
  - w_normal [0.3-5.5]: Class weight for normal samples
    * CONSTRAINT: w_normal + w_anomal = 5.88 (v0.2 best: w_n=1.85, w_a=4.03)
    * Dataset: 12.9% anomaly → 87.1% normal (imbalanced)
    * v0.2 best ratio: w_anomal/w_normal = 2.18
    * Lower w_normal → higher w_anomal → prioritize anomaly detection
    * Higher w_normal → lower w_anomal → reduce false positives

Objectives: Maximize AUC/Precision/Recall/F1 × 3 horizons, Minimize FPR × 3 horizons.

Current best parent (macro_F1={parent.macro_f1:.4f}):
  Focal: α={parent.focal_alpha:.3f} γ={parent.focal_gamma:.3f} w_n={parent.w_normal:.3f} w_a={parent.w_anomal:.3f} (sum={parent.w_normal+parent.w_anomal:.2f}, ratio={parent.w_anomal/parent.w_normal:.2f})
  Arch: patch={parent.patch_len} stride={parent.stride} (FIXED v0.2.3 best)
  LoRA: rank={parent.lora_rank} alpha={parent.lora_alpha} (FIXED v0.2.3 best)
  Seed: None (random)
{parent_str}

YOUR TASK:
1. Evaluate if NSGA-II suggestion is reasonable for Focal Loss:
   - CONSTRAINT CHECK: w_normal + w_anomal must equal 5.88 (already applied by sampler)
   - Weight ratio: v0.2 best was w_anomal/w_normal = 2.18 (w_n=1.85, w_a=4.03)
   - Alpha-gamma balance: high gamma needs lower alpha for stability
   - Parent F1={parent.macro_f1:.4f} from α={parent.focal_alpha:.3f} γ={parent.focal_gamma:.3f}
2. Accept NSGA-II's Pareto suggestion if it looks reasonable
3. Micro-adjust only if clearly problematic (but maintain constraint w_n + w_a = 5.88)
4. Trust NSGA-II — it explores Pareto trade-offs across all 15 objectives

Output (JSON only, no other text):
{{"focal_alpha":<float 0.5-0.9>,"focal_gamma":<float 0.6-2.0>,"w_normal":<float 0.3-5.5>,"rationale":"<one sentence>"}}

Note: w_anomal will be computed as 5.88 - w_normal automatically."""

    def _parse_json(self, text: str) -> Optional[Dict]:
        m = re.search(r'\{[^{{}}]*"focal_alpha"[^{{}}]*\}', text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except Exception:
            return None

    def _validate(self, cfg: Dict) -> bool:
        # v0.2.1C5: LLM only returns focal_alpha, focal_gamma, w_normal (constraint applies w_anomal)
        required = {"focal_alpha", "focal_gamma", "w_normal", "rationale"}
        if not required.issubset(cfg):
            return False
        # Check focal_alpha in [0.5, 0.9]
        if not (0.5 <= cfg.get("focal_alpha", 0) <= 0.9):
            return False
        # Check focal_gamma in [0.6, 2.0]
        if not (0.6 <= cfg.get("focal_gamma", 0) <= 2.0):
            return False
        # Check w_normal in [0.3, 5.5]
        if not (0.3 <= cfg.get("w_normal", 0) <= 5.5):
            return False
        # Apply constraint: w_anomal = 5.88 - w_normal
        cfg["w_anomal"] = 5.88 - cfg["w_normal"]
        return True

    def _preload_model(self) -> None:
        """Preload model into Ollama GPU memory before inference."""
        print(f"[PatchTSTAgent] Loading {self.model_name} to GPU...")
        try:
            requests.post(
                self.OLLAMA_API_URL,
                json={"model": self.model_name, "prompt": "", "stream": False},
                timeout=120,  # v0.3.6_repro: increased from 30s to avoid timeout
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
