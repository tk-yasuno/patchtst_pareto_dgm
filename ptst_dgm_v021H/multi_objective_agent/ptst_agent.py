"""
ptst_agent.py
LLM-based validator for NSGA-II proposals in PatchTST DGM (v0.2.1H).

Flow:
  1. PatchTSTSampler.suggest() → NSGA-II proposes 9 parameters (horizon-specific Focal Loss)
  2. codestral:latest validates the proposal in horizon-specific context
  3. Returns validated (or NSGA-II raw) parameters
  4. Architecture fixed at v0.2.3 best (patch_len=26, stride=16)
  5. LoRA fixed at v0.2.3 best (rank=16, alpha=47)
  6. Constraint: w_normal_XXd + w_anomal_XXd = 5.88 (per-horizon)
  7. Seed: None (random init for diversity)
"""
from __future__ import annotations

import json
import re
from typing import Dict, Optional, Tuple

import requests

from ptst_dgm_v021H.agent.archive import PatchTSTAgentEntry, HORIZONS
from ptst_dgm_v021H.multi_objective_agent.ptst_sampler import PatchTSTSampler, PARAM_BOUNDS


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
        for h in [30, 60, 90]:
            hd = f"{h}d"
            print(f"  {hd} Focal Loss: α={nsgaii_params[f'focal_alpha_{hd}']:.3f} γ={nsgaii_params[f'focal_gamma_{hd}']:.3f} "
                  f"w_n={nsgaii_params[f'w_normal_{hd}']:.3f} w_a={nsgaii_params[f'w_anomal_{hd}']:.3f} (constraint: w_n+w_a=5.88)")

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
                        # Add fixed Architecture, LoRA, and seed=None to LLM result (v0.2.1H)
                        from ptst_dgm_v021H.multi_objective_agent.ptst_sampler import FIXED_ARCH_PARAMS, FIXED_LORA_PARAMS
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
        
        # Build horizon-specific parameter display
        nsgaii_params_str = ""
        parent_params_str = ""
        for h in [30, 60, 90]:
            hd = f"{h}d"
            nsgaii_params_str += f"  {hd}: α={nsgaii[f'focal_alpha_{hd}']:.3f} γ={nsgaii[f'focal_gamma_{hd}']:.3f} w_n={nsgaii[f'w_normal_{hd}']:.3f} w_a={nsgaii[f'w_anomal_{hd}']:.3f}\n"
            if hasattr(parent, f'focal_alpha_{hd}'):
                parent_params_str += f"  {hd}: α={getattr(parent, f'focal_alpha_{hd}'):.3f} γ={getattr(parent, f'focal_gamma_{hd}'):.3f} w_n={getattr(parent, f'w_normal_{hd}'):.3f} w_a={getattr(parent, f'w_anomal_{hd}'):.3f}\n"
        
        return f"""You are an ML expert validating an Optuna NSGA-II suggestion for
horizon-specific Focal Loss optimization in PatchTST (v0.2.1H).

=== TOOL CALL RESULT ===
Tool: get_nsgaii_suggestion (NSGA-II Pareto 15-Objective Optimizer)
Suggested horizon-specific Focal Loss parameters (9D proposal):
{nsgaii_params_str}  Architecture: FIXED v0.2.3 best (patch_len={nsgaii['patch_len']}, stride={nsgaii['stride']})
  LoRA: FIXED v0.2.3 best (rank={nsgaii['lora_rank']}, alpha={nsgaii['lora_alpha']})
  Constraint: w_normal_XXd + w_anomal_XXd = 5.88 (per-horizon)
========================

Context (v0.2.1H - Horizon-specific 9D optimization):
  - Each horizon (30d/60d/90d) has independent Focal Loss parameters
  - focal_alpha [0.5-0.9], focal_gamma [0.6-2.0], w_normal [0.3-5.5]
  - Constraint: w_normal_XXd + w_anomal_XXd = 5.88 (applied per-horizon)
  - Dataset: 12.9% anomaly, 87.1% normal (imbalanced)
  - Different horizons may need different loss configurations

Objectives: Maximize AUC/Precision/Recall/F1 × 3 horizons, Minimize FPR × 3 horizons.

Current best parent (macro_F1={parent.macro_f1:.4f}):
{parent_params_str}  Arch: patch={parent.patch_len} stride={parent.stride} (FIXED v0.2.3 best)
  LoRA: rank={parent.lora_rank} alpha={parent.lora_alpha} (FIXED v0.2.3 best)
{parent_str}

YOUR TASK:
1. Validate NSGA-II's horizon-specific suggestions (9D space)
2. Accept if reasonable - trust NSGA-II's Pareto exploration
3. Micro-adjust only if clearly problematic (maintain per-horizon constraint)
4. Each horizon can have different optimal loss parameters

Output (JSON only, no other text):
{{"focal_alpha_30d":<float>,"focal_gamma_30d":<float>,"w_normal_30d":<float>,
  "focal_alpha_60d":<float>,"focal_gamma_60d":<float>,"w_normal_60d":<float>,
  "focal_alpha_90d":<float>,"focal_gamma_90d":<float>,"w_normal_90d":<float>,
  "rationale":"<one sentence>"}}

Note: w_anomal_XXd will be computed as 5.88 - w_normal_XXd automatically per horizon."""

    def _parse_json(self, text: str) -> Optional[Dict]:
        m = re.search(r'\{[^{{}}]*"focal_alpha_30d"[^{{}}]*\}', text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except Exception:
            return None

    def _validate(self, cfg: Dict) -> bool:
        # v0.2.1H: LLM returns 9 parameters (3 per horizon)
        required = {
            "focal_alpha_30d", "focal_gamma_30d", "w_normal_30d",
            "focal_alpha_60d", "focal_gamma_60d", "w_normal_60d",
            "focal_alpha_90d", "focal_gamma_90d", "w_normal_90d",
            "rationale"
        }
        if not required.issubset(cfg):
            return False
        
        # Validate each horizon's parameters
        for h in [30, 60, 90]:
            hd = f"{h}d"
            # Check focal_alpha in [0.5, 0.9]
            if not (0.5 <= cfg.get(f"focal_alpha_{hd}", 0) <= 0.9):
                return False
            # Check focal_gamma in [0.6, 2.0]
            if not (0.6 <= cfg.get(f"focal_gamma_{hd}", 0) <= 2.0):
                return False
            # Check w_normal in [0.3, 5.5]
            if not (0.3 <= cfg.get(f"w_normal_{hd}", 0) <= 5.5):
                return False
            # Apply constraint: w_anomal = 5.88 - w_normal (per-horizon)
            cfg[f"w_anomal_{hd}"] = 5.88 - cfg[f"w_normal_{hd}"]
        
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
