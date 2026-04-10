"""
vLLM drop-in replacement for src/ollama_client.py.

Same public interface as OllamaClient (generate / generate_json /
batch_generate / embed / test_connection) but backed by a locally-loaded
vLLM engine instead of the Ollama subprocess.

Default model: Qwen/Qwen2-7B-Instruct  (same capability as qwen2:7b in Ollama)
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2-7B-Instruct"
_DEFAULT_SYSTEM = (
    "You are an expert materials scientist specialising in 2D material "
    "doping and synthesis. Follow instructions precisely."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat_prompt(system: str, user: str, json_mode: bool = False) -> str:
    """Apply Qwen2-Instruct chat template."""
    msg = user
    if json_mode:
        msg += "\n\nIMPORTANT: Respond with ONLY valid JSON. No other text."
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _extract_json(text: str) -> Optional[str]:
    """Best-effort JSON extraction (raw → ```json block → first {…})."""
    text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            json.loads(m.group(1))
            return m.group(1)
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            json.loads(m.group(0))
            return m.group(0)
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# VLLMClient
# ---------------------------------------------------------------------------

class VLLMClient:
    """
    Drop-in replacement for OllamaClient that uses a local vLLM engine.

    All public methods match OllamaClient's signatures so existing ARIA
    variant code (src/variants/*.py) works without modification.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.85,
        **_kwargs,          # absorb unused OllamaClient kwargs (base_url, etc.)
    ):
        self.model = model
        self._embedding_model_cached = None
        logger.info(f"Loading vLLM model: {model}  tp={tensor_parallel_size}")
        self._llm = LLM(
            model,
            dtype="half",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        logger.info("vLLM model ready.")

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = True,
        system_message: Optional[str] = None,
    ) -> str:
        system = system_message or _DEFAULT_SYSTEM
        full_prompt = _chat_prompt(system, prompt, json_mode)
        sp = SamplingParams(
            temperature=max(float(temperature), 1e-4),
            top_p=1.0,
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )
        outputs = self._llm.generate([full_prompt], sp)
        raw = outputs[0].outputs[0].text.strip()

        if json_mode:
            extracted = _extract_json(raw)
            if extracted:
                return extracted
            logger.warning("JSON extraction failed. Returning raw output.")
        return raw

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        system_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        raw = self.generate(
            prompt, temperature, max_tokens,
            json_mode=True, system_message=system_message,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON parse error: {exc}\nRaw: {raw[:300]}") from exc

    def batch_generate(
        self,
        prompts: List[str],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = True,
        system_message: Optional[str] = None,
    ) -> List[str]:
        """Batch all prompts in a single vLLM call for maximum throughput."""
        system = system_message or _DEFAULT_SYSTEM
        full_prompts = [_chat_prompt(system, p, json_mode) for p in prompts]
        sp = SamplingParams(
            temperature=max(float(temperature), 1e-4),
            top_p=1.0,
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )
        outputs = self._llm.generate(full_prompts, sp)
        results = []
        for out in outputs:
            raw = out.outputs[0].text.strip()
            if json_mode:
                extracted = _extract_json(raw)
                results.append(extracted if extracted else raw)
            else:
                results.append(raw)
        return results

    # ------------------------------------------------------------------
    # Embeddings — keep sentence-transformers (same as OllamaClient)
    # ------------------------------------------------------------------

    def embed(
        self, text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        from sentence_transformers import SentenceTransformer
        if self._embedding_model_cached is None:
            logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
            self._embedding_model_cached = SentenceTransformer("all-MiniLM-L6-v2")
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        vecs = self._embedding_model_cached.encode(texts, convert_to_numpy=True).tolist()
        return vecs[0] if is_single else vecs

    # ------------------------------------------------------------------
    # Misc compatibility shims
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        try:
            r = self.generate(
                'Test. Respond with exactly: {"status": "ok"}',
                json_mode=True, max_tokens=50,
            )
            return json.loads(r).get("status") == "ok"
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return {"model": self.model, "backend": "vllm"}


# ---------------------------------------------------------------------------
# Singleton factory — mirrors get_ollama_client() in src/ollama_client.py
# ---------------------------------------------------------------------------

_default_client: Optional[VLLMClient] = None


def get_vllm_client(
    model: str = DEFAULT_MODEL,
    tensor_parallel_size: int = 4,
    **kwargs,
) -> VLLMClient:
    global _default_client
    if _default_client is None or _default_client.model != model:
        _default_client = VLLMClient(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs,
        )
    return _default_client
