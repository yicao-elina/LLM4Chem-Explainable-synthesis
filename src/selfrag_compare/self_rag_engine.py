"""
SelfRAG Baseline Engine for 2D Material Doping Tasks

Uses selfrag/selfrag_llama2_13b (vLLM) with TF-IDF retrieval over
cleaned/train/ paper paragraphs — the same knowledge base ARIA's KG
was built from.

Inference follows the proper SelfRAG retrieval loop:
  1. Start with just the instruction (no pre-injected paragraph).
  2. Generate until the model emits [Retrieval] (stop token) or EOS.
  3. If [Retrieval], retrieve the most relevant chunk and inject it.
  4. Continue generating. Repeat up to MAX_RETRIEVAL_ITERS times.

This faithfully implements SelfRAG's adaptive retrieval, matching the
described approach in the original paper.

Compatible with src/project/test/evaluation_multi.py via the
CausalReasoningEngine interface.

Retrieval corpus: cleaned/train/ (105 papers, chunked by paragraph)
Test data:        data/KG/outputs/test_doping_data.json (23 experiments)
DO NOT include:   cleaned/test/ (sources of test experiments — oracle leakage)
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parents[2]
TRAIN_PAPER_DIR = REPO_ROOT / "cleaned" / "train"
TRAINING_GRAPH_DEFAULT = REPO_ROOT / "data" / "KG" / "outputs" / "combined_doping_data.json"

# ---------------------------------------------------------------------------
# SelfRAG inference settings
# ---------------------------------------------------------------------------
SELFRAG_MODEL = "selfrag/selfrag_llama2_13b"

# Stop at [Retrieval] so we can inject a real passage before continuing
_SP_STOP_AT_RETRIEVAL = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=300,
    skip_special_tokens=False,
    stop=["[Retrieval]"],
)
# Final generation after last retrieval (no early stop)
_SP_FINAL = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=300,
    skip_special_tokens=False,
)

MAX_RETRIEVAL_ITERS = 3  # cap the retrieval loop
MAX_PARAGRAPH_CHARS = 500  # keep injected paragraphs short

# Reflection tokens emitted by SelfRAG
_REFLECTION_RE = re.compile(
    r"\[(Retrieval|No Retrieval|Relevant|Irrelevant"
    r"|Fully supported|Partially supported|No support[^]]*"
    r"|Continue to Use Evidence"
    r"|Utility:\d+)\]",
    re.IGNORECASE,
)

# Heuristics for non-content chunks to exclude from corpus
_JUNK_RE = re.compile(
    r"^\s*(\d+\s*$"                        # lone page numbers
    r"|[A-Z][a-z]+ [A-Z][a-z]+\d"         # author line: "Smith J1,"
    r"|\*?\w+@\w+\.\w+"                    # email address
    r"|doi:|http[s]?://"                   # DOI / URL lines
    r"|©|copyright|\bISSN\b|\bDOI\b"       # copyright line
    r"|^\s*\d+\.\s+Introduction"           # section header
    r"|^\s*References\s*$"                 # references header
    r"|\[\d+\])",                          # reference list entries
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Prompt helpers  (strict SelfRAG format)
# ---------------------------------------------------------------------------

def format_prompt(instruction: str) -> str:
    """Initial prompt — no paragraph, let the model decide to retrieve."""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def inject_paragraph(prompt_so_far: str, generated_so_far: str, paragraph: str) -> str:
    """Append [Retrieval]<paragraph>...</paragraph> to continue generation."""
    return (
        prompt_so_far
        + generated_so_far
        + "[Retrieval]<paragraph>"
        + paragraph
        + "</paragraph>"
    )


def strip_reflection_tokens(text: str) -> str:
    """Remove SelfRAG special tokens; return clean readable text."""
    clean = _REFLECTION_RE.sub("", text)
    clean = re.sub(r"</?paragraph>", "", clean)
    return clean.strip()


# ---------------------------------------------------------------------------
# Retrieval corpus: chunk papers into content paragraphs
# ---------------------------------------------------------------------------

def _is_junk(chunk: str) -> bool:
    """Return True for non-content chunks (author lines, refs, page numbers)."""
    if _JUNK_RE.search(chunk):
        return True
    # Skip chunks that are mostly numbers / citations
    alpha = sum(c.isalpha() for c in chunk)
    if len(chunk) > 0 and alpha / len(chunk) < 0.4:
        return True
    return False


def _chunk_paper(text: str, min_chars: int = 120, max_chars: int = MAX_PARAGRAPH_CHARS) -> List[str]:
    """Split a paper into clean content paragraphs."""
    raw_chunks = [p.strip() for p in re.split(r"\n{2,}", text)]
    good = []
    for c in raw_chunks:
        if len(c) < min_chars:
            continue
        if _is_junk(c):
            continue
        good.append(c[:max_chars])
    return good


def build_corpus(paper_dir: Path) -> List[Dict[str, str]]:
    """Load all .txt train papers → list of {source, text} paragraph dicts."""
    corpus = []
    if not paper_dir.exists():
        print(f"[SelfRAG] Warning: paper dir not found: {paper_dir}")
        return corpus
    for fpath in sorted(paper_dir.glob("*.txt")):
        raw = fpath.read_text(errors="replace")
        for chunk in _chunk_paper(raw):
            corpus.append({"source": fpath.name, "text": chunk})
    print(f"[SelfRAG] Corpus: {len(corpus)} content paragraphs from {paper_dir.name}/")
    return corpus


# ---------------------------------------------------------------------------
# TF-IDF retriever
# ---------------------------------------------------------------------------

class TFIDFRetriever:
    def __init__(self, corpus: List[Dict[str, str]]):
        self.corpus = corpus
        texts = [item["text"] for item in corpus]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=30000)
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 1) -> List[Dict[str, str]]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.corpus[i] for i in top_idx]


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def _synthesis_to_query(synthesis_inputs: Dict[str, Any], host: str = "", dopant: str = "") -> str:
    """Concise natural-language forward-prediction question."""
    method = synthesis_inputs.get("method", "")
    temp = synthesis_inputs.get("temperature_c")
    atm = synthesis_inputs.get("atmosphere")

    q = f"What are the electronic, thermal, and structural properties of {host}"
    if dopant:
        q += f" after doping with {dopant}"
    if method:
        q += f" synthesized by {method}"
    if temp:
        q += f" at {temp}°C"
    if atm:
        q += f" in {atm} atmosphere"
    q += "?"
    return q


def _properties_to_query(desired_properties: Dict[str, Any], host: str = "", dopant: str = "") -> str:
    """Concise natural-language inverse-design question."""
    elec = desired_properties.get("electronic", {})
    carrier = elec.get("carrier_type", "")
    bg_after = (elec.get("band_gap_ev") or {}).get("after")
    thermal = desired_properties.get("thermal", {}).get("thermal_stability", "")

    q = "What synthesis conditions (method, temperature, atmosphere, dopant) are needed"
    if host:
        q += f" for {host}"
    q += " to achieve"
    if carrier:
        q += f" {carrier} conductivity"
    if bg_after is not None:
        q += f" with a band gap of {bg_after} eV"
    if thermal:
        q += f" and {thermal}"
    if not any([carrier, bg_after is not None, thermal]):
        q += " the desired material properties"
    q += "?"
    return q


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class CausalReasoningEngine:
    """
    SelfRAG engine for 2D material doping tasks.

    Implements the adaptive retrieval loop:
      - Model generates freely until it emits [Retrieval] or EOS.
      - On [Retrieval], we retrieve the best-matching paragraph and inject it.
      - Continue up to MAX_RETRIEVAL_ITERS times.

    Interface matches ARIA variants — compatible with evaluation_multi.py.

    Args:
        training_graph_file: Path to combined_doping_data.json.
            Pass None or use_kg_passages=False to skip.
        use_kg_passages: Also index serialized KG experiments as passages.
        top_k_retrieve: Passages to retrieve per [Retrieval] call.
    """

    def __init__(
        self,
        training_graph_file: str = str(TRAINING_GRAPH_DEFAULT),
        use_kg_passages: bool = False,
        top_k_retrieve: int = 1,
        tensor_parallel_size: int = 4,
    ):
        self.top_k = top_k_retrieve

        corpus = build_corpus(TRAIN_PAPER_DIR)
        if use_kg_passages and training_graph_file:
            corpus += self._kg_passages(training_graph_file)

        if not corpus:
            raise RuntimeError(
                f"Empty retrieval corpus. Check that {TRAIN_PAPER_DIR} exists."
            )
        self.retriever = TFIDFRetriever(corpus)

        print(f"[SelfRAG] Loading model: {SELFRAG_MODEL} (tensor_parallel_size={tensor_parallel_size}) ...")
        self.model = LLM(SELFRAG_MODEL, dtype="half", tensor_parallel_size=tensor_parallel_size)
        print("[SelfRAG] Model ready.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward_prediction(self, synthesis_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Predict material properties from synthesis conditions."""
        query = _synthesis_to_query(synthesis_inputs)
        raw = self._selfrag_loop(query)
        clean = strip_reflection_tokens(raw)
        return {
            "predicted_properties": {"description": clean},
            "reasoning": clean,
            "raw_selfrag_output": raw,
            "confidence": 0.5,
        }

    def inverse_design(self, desired_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest synthesis conditions to achieve desired properties."""
        query = _properties_to_query(desired_properties)
        raw = self._selfrag_loop(query)
        clean = strip_reflection_tokens(raw)
        return {
            "suggested_synthesis_conditions": {"description": clean},
            "reasoning": clean,
            "raw_selfrag_output": raw,
            "confidence": 0.5,
        }

    # ------------------------------------------------------------------
    # SelfRAG retrieval loop
    # ------------------------------------------------------------------

    def _selfrag_loop(self, query: str) -> str:
        """
        Adaptive retrieval loop as described in the SelfRAG paper.

        The model generates until it requests [Retrieval] or finishes.
        On each [Retrieval] request we fetch the best-matching paragraph
        and inject it so the model can continue with grounded context.
        """
        base_prompt = format_prompt(query)
        accumulated = ""   # everything the model has generated so far
        current_prompt = base_prompt

        for iteration in range(MAX_RETRIEVAL_ITERS):
            is_last = (iteration == MAX_RETRIEVAL_ITERS - 1)
            sp = _SP_FINAL if is_last else _SP_STOP_AT_RETRIEVAL

            outputs = self.model.generate([current_prompt], sp)
            out = outputs[0].outputs[0]
            chunk = out.text
            accumulated += chunk

            # Finished normally (EOS or max_tokens without [Retrieval] stop)
            if out.finish_reason != "stop" or is_last:
                break

            # Model hit [Retrieval] stop — inject a real passage and continue
            context = query + " " + strip_reflection_tokens(accumulated)
            passage = self.retriever.retrieve(context, top_k=self.top_k)[0]["text"]
            current_prompt = inject_paragraph(base_prompt, accumulated, passage)

        return accumulated

    # ------------------------------------------------------------------
    # KG passages (optional corpus augmentation)
    # ------------------------------------------------------------------

    @staticmethod
    def _kg_passages(graph_file: str) -> List[Dict[str, str]]:
        passages = []
        try:
            with open(graph_file) as f:
                data = json.load(f)
            exps = data.get("doping_experiments", []) if isinstance(data, dict) else data
            for exp in exps:
                passages.append({
                    "source": "kg:" + exp.get("experiment_id", "?"),
                    "text": _exp_to_text(exp),
                })
            print(f"[SelfRAG] Added {len(passages)} KG experiment passages")
        except Exception as e:
            print(f"[SelfRAG] Warning: could not load KG passages: {e}")
        return passages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exp_to_text(exp: Dict) -> str:
    """Serialize a KG experiment to a retrieval-friendly sentence."""
    host = exp.get("host_material", "unknown material")
    dopant = exp.get("dopant", {})
    elem = dopant.get("element", "unknown")
    conc = dopant.get("concentration", "")
    sc = exp.get("synthesis_conditions", {})
    pc = exp.get("property_changes", {})

    parts = [f"Doping {host} with {elem}"]
    if conc:
        parts.append(f"concentration {conc}")
    if sc.get("method"):
        parts.append(f"method: {sc['method']}")
    if sc.get("temperature_c"):
        parts.append(f"temperature: {sc['temperature_c']}°C")
    if sc.get("atmosphere"):
        parts.append(f"atmosphere: {sc['atmosphere']}")

    elec = pc.get("electronic", {})
    if elec.get("carrier_type"):
        parts.append(f"carrier type: {elec['carrier_type']}")
    bg = elec.get("band_gap_ev", {})
    if isinstance(bg, dict) and bg.get("after") is not None:
        parts.append(f"band gap after: {bg['after']} eV")

    thermal = pc.get("thermal", {})
    if thermal.get("thermal_stability"):
        parts.append(f"thermal stability: {thermal['thermal_stability']}")

    site = (exp.get("doping_outcome", {})
               .get("site_distribution", {})
               .get("primary_site", ""))
    if site:
        parts.append(f"primary doping site: {site}")

    return ". ".join(parts) + "."
