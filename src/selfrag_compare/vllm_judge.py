"""
vLLM-based LLM Judge for rebuttal evaluation.

Drop-in replacement for OllamaJudge (src/evaluation/ollama_judge.py),
backed by vLLM instead of Ollama. Reuses the same domain-specific prompts
and 4-metric rubric (Processing Feasibility / Structure Emergence /
Property Consistency / Causal PSP Reasoning, total 100 points).

Default model: Qwen/Qwen2-7B-Instruct  (same as qwen2:7b in Ollama)
With 4 H100s you can bump to Qwen/Qwen2.5-72B via --judge_model.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional

from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
DEFAULT_JUDGE_MODEL = "Qwen/Qwen2-7B-Instruct"
DEFAULT_TENSOR_PARALLEL = 1  # 7B fits comfortably on 1 H100; leave 3 for SelfRAG

_JUDGE_SP = SamplingParams(
    temperature=0.6,
    top_p=1.0,
    max_tokens=4096,
    skip_special_tokens=True,
)

# Chat template for Qwen-Instruct
_SYSTEM = (
    "You are an expert materials scientist evaluating AI-generated predictions "
    "for 2D materials processing. Output ONLY valid JSON — no markdown, no prose."
)


def _apply_chat_template(system: str, user: str) -> str:
    """Qwen2/Qwen2.5 chat template (no special tokenizer needed)."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _parse_json(text: str) -> Dict:
    """Best-effort JSON extraction from model output."""
    # Try raw parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Try extracting from ```json ... ``` block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first { ... } span
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# VLLMJudge — same interface as OllamaJudge
# ---------------------------------------------------------------------------

class VLLMJudge:
    """
    LLM judge backed by vLLM.

    Evaluates predictions on 4 domain-specific metrics:
      1. Processing Feasibility   (0-40)
      2. Structure Emergence      (0-30)
      3. Property Consistency     (0-20)
      4. Causal PSP Reasoning     (0-10)
    Total: 100 points.

    Interface is identical to OllamaJudge so it can be swapped in.
    """

    METRICS = {
        "processing_feasibility": {
            "name": "Processing Feasibility",
            "max_score": 40,
            "description": "Thermodynamic & kinetic viability of predicted processing conditions",
            "rubric": """
Score 35-40: All conditions thermodynamically/kinetically viable, realistic equipment
  - Temperature ranges appropriate for material phase stability
  - Pressure conditions compatible with synthesis method
  - Time scales realistic for defect formation/annealing
  - Atmosphere correct for doping/oxidation control
Score 25-34: Minor feasibility issues (slightly high/low temperature, extended but possible timescales)
Score 15-24: Significant issues (wrong atmosphere, unrealistic pressure, incompatible temperature)
Score 0-14: Fundamentally impossible (violates phase diagram, destroys material, physically impossible)
""",
        },
        "structure_emergence": {
            "name": "Structure Emergence",
            "max_score": 30,
            "description": "Accuracy of predicted structural outcomes from processing",
            "rubric": """
Score 25-30: Correct defect type, realistic density, accurate lattice strain, correct stacking order
Score 18-24: Correct defect type but density slightly off, strain directionally correct
Score 10-17: Right defect family but wrong specific type, major uncertainty in density
Score 0-9: Wrong defect type, structurally impossible outcomes
""",
        },
        "property_consistency": {
            "name": "Property Consistency",
            "max_score": 20,
            "description": "Coherence between predicted properties and structure/processing",
            "rubric": """
Score 17-20: Electronic/mechanical/optical/magnetic properties fully consistent with structure
Score 12-16: Minor inconsistencies, band gap off <0.5 eV, qualitative trends correct
Score 6-11: Wrong carrier type (n vs p), mechanical properties inconsistent with defects
Score 0-5: Metallic predicted for insulating, physically impossible values, contradictions
""",
        },
        "causal_psp_reasoning": {
            "name": "Causal PSP Reasoning",
            "max_score": 10,
            "description": "Quality of Processing→Structure→Property causal chain",
            "rubric": """
Score 8-10: Clear P→S→P chain, mechanistic explanations, physical justifications, uncertainty acknowledged
Score 5-7: P→S→P present but incomplete, some mechanistic explanations
Score 2-4: Mentions connections but lacks detail, empirical correlations without mechanisms
Score 0-1: No P→S→P chain, pure correlation or guessing
""",
        },
    }

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model_name = model
        logger.info(f"Loading judge model: {model} (tp={tensor_parallel_size}) ...")
        self._llm = LLM(model, dtype="half", tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization)
        logger.info("Judge model ready.")

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: Dict,
        prediction: Dict,
        ground_truth: Dict,
        metric_key: str,
    ) -> str:
        m = self.METRICS[metric_key]
        user = f"""Evaluate this materials science prediction on **{m['name']}**.

METRIC: {m['name']} (0-{m['max_score']} points)
{m['description']}

RUBRIC:
{m['rubric']}

INPUT QUERY:
{json.dumps(query, indent=2)}

GROUND TRUTH:
{json.dumps(ground_truth, indent=2)}

MODEL PREDICTION:
{json.dumps(prediction, indent=2)}

Return ONLY a JSON object with this exact schema:
{{
  "metric": "{metric_key}",
  "score": <float 0-{m['max_score']}>,
  "justification": "<1-2 sentence explanation>",
  "failure_modes": ["<issue 1>", ...],
  "strengths": ["<strength 1>", ...]
}}"""
        return _apply_chat_template(_SYSTEM, user)

    def score_prediction(
        self,
        query: Dict,
        prediction: Dict,
        ground_truth: Dict,
        metric_key: str,
    ) -> Dict:
        if metric_key not in self.METRICS:
            raise ValueError(f"Unknown metric: {metric_key}")

        prompt = self._build_prompt(query, prediction, ground_truth, metric_key)
        outputs = self._llm.generate([prompt], _JUDGE_SP)
        raw = outputs[0].outputs[0].text

        result = _parse_json(raw)
        if not result:
            logger.warning(f"JSON parse failed for {metric_key}. Raw: {raw[:200]}")
            result = {
                "metric": metric_key,
                "score": 0.0,
                "justification": f"Parse failed: {raw[:100]}",
                "failure_modes": ["judge parse error"],
                "strengths": [],
            }

        # Clip score to valid range
        max_s = self.METRICS[metric_key]["max_score"]
        result["score"] = max(0.0, min(float(result.get("score", 0)), max_s))
        return result

    def evaluate_all_metrics(
        self,
        query: Dict,
        prediction: Dict,
        ground_truth: Dict,
    ) -> Dict:
        """Score on all 4 metrics. Returns results dict with overall_score (0-100)."""
        # Batch all 4 prompts in one vLLM call for efficiency
        prompts = [
            self._build_prompt(query, prediction, ground_truth, mk)
            for mk in self.METRICS
        ]
        outputs = self._llm.generate(prompts, _JUDGE_SP)

        metric_scores = {}
        total = 0.0
        for mk, out in zip(self.METRICS, outputs):
            raw = out.outputs[0].text
            res = _parse_json(raw)
            if not res:
                res = {
                    "metric": mk,
                    "score": 0.0,
                    "justification": f"Parse failed",
                    "failure_modes": ["judge parse error"],
                    "strengths": [],
                }
            max_s = self.METRICS[mk]["max_score"]
            res["score"] = max(0.0, min(float(res.get("score", 0)), max_s))
            metric_scores[mk] = res
            total += res["score"]

        return {
            "query": query,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "metric_scores": metric_scores,
            "overall_score": total,
            "overall_score_normalized": total,  # already /100
        }

    def batch_evaluate(self, test_cases: List[Dict]) -> List[Dict]:
        results = []
        for i, tc in enumerate(test_cases):
            logger.info(f"Evaluating {i+1}/{len(test_cases)} ...")
            r = self.evaluate_all_metrics(
                tc.get("query", {}),
                tc.get("prediction", {}),
                tc.get("ground_truth", {}),
            )
            r["test_case_id"] = tc.get("id", i)
            results.append(r)
        return results
