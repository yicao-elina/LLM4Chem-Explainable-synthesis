"""
Run the full ARIA evaluation pipeline with vLLM instead of Ollama.

Uses kg_normalized_full.json (405 experiments / 1237 relationships,
normalized from the complete main-branch corpus) as the knowledge graph.

Replicates run_phase1_evaluation.py exactly — same 4 variants, same
embedding metrics, same in-domain / out-of-domain split logic — but
replaces OllamaClient with VLLMClient so no Ollama server is needed.

How it works
------------
1. Monkey-patches src.ollama_client.get_ollama_client so every variant
   automatically gets a VLLMClient instead of OllamaClient.
2. Splits kg_normalized_full.json 80/20 by paper (no paper overlap) to
   get proper in-domain (train) and out-of-domain (test) KGs.
3. Runs all 4 variants on first --n_tests cases from each split.
4. Scores with evaluate_prediction() (embedding-based, no API).
5. Writes results/phase2_vllm/vllm_raw_results.csv and prints a
   leaderboard.

Usage
-----
    python src/rebuttal/run_vllm_aria_eval.py
    python src/rebuttal/run_vllm_aria_eval.py --n_tests 10 --gpus 4
    python src/rebuttal/run_vllm_aria_eval.py \\
        --kg data/KG/outputs/kg_normalized_full.json \\
        --output results/phase2_vllm/vllm_raw_results.csv
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sentence_transformers import SentenceTransformer

# ── repo root on sys.path ────────────────────────────────────────────────────
REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_KG   = REPO / "data/KG/outputs/kg_normalized_full.json"
DEFAULT_OUT  = REPO / "results/phase2_vllm/vllm_raw_results.csv"
EMBED_MODEL  = "all-MiniLM-L6-v2"
RANDOM_SEED  = 42


# ---------------------------------------------------------------------------
# 1. Monkey-patch get_ollama_client → VLLMClient
# ---------------------------------------------------------------------------

def patch_ollama_with_vllm(model: str, tensor_parallel_size: int):
    """
    Replace get_ollama_client() globally so every ARIA variant
    transparently uses vLLM without code changes.
    """
    from src.rebuttal.vllm_client import get_vllm_client
    import src.ollama_client as _oc_mod

    _client_cache = {}

    def _patched_get(model: str = model, **kwargs):
        if model not in _client_cache:
            logger.info(f"[patch] Creating VLLMClient for model={model}")
            _client_cache[model] = get_vllm_client(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
            )
        return _client_cache[model]

    _oc_mod.get_ollama_client = _patched_get
    logger.info("Patched get_ollama_client → VLLMClient")


# ---------------------------------------------------------------------------
# 2. KG splitting
# ---------------------------------------------------------------------------

def split_kg(kg_path: Path, seed: int = RANDOM_SEED, train_frac: float = 0.8):
    """
    80/20 split by source_file (paper) with no paper overlap.

    Returns (train_data, test_data) each as dicts with
    'doping_experiments' and 'causal_relationships'.
    """
    with open(kg_path) as f:
        data = json.load(f)

    experiments   = data.get("doping_experiments", [])
    relationships = data.get("causal_relationships", [])

    # Collect unique papers from experiments
    papers = list({e.get("source_file", "") for e in experiments if e.get("source_file")})
    random.seed(seed)
    random.shuffle(papers)
    split_at = max(1, int(len(papers) * train_frac))
    train_papers = set(papers[:split_at])
    test_papers  = set(papers[split_at:])

    def _filter(items, paper_set):
        return [x for x in items if x.get("source_file", "") in paper_set]

    train_data = {
        "doping_experiments":   _filter(experiments,   train_papers),
        "causal_relationships": _filter(relationships, train_papers),
    }
    test_data = {
        "doping_experiments":   _filter(experiments,   test_papers),
        "causal_relationships": _filter(relationships, test_papers),
    }

    logger.info(
        f"Split: train={len(train_data['doping_experiments'])} exp / "
        f"{len(train_data['causal_relationships'])} rels  |  "
        f"test={len(test_data['doping_experiments'])} exp / "
        f"{len(test_data['causal_relationships'])} rels"
    )
    return train_data, test_data


def write_tmp_kg(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# 3. Test-case extraction  (same logic as Phase1Evaluator._load_test_data)
# ---------------------------------------------------------------------------

def extract_test_cases(kg_data: dict, domain: str, n: int) -> List[Dict]:
    """
    Pull first n causal_relationships and wrap them as test cases.
    Follows the exact same format as run_phase1_evaluation.py.
    """
    relationships = kg_data.get("causal_relationships", [])
    cases = []
    for i, rel in enumerate(relationships[:n]):
        if not isinstance(rel, dict):
            continue
        synthesis = {
            "cause_parameter": rel.get("cause_parameter", ""),
            "method":          rel.get("method", ""),
            "temperature":     rel.get("temperature", ""),
        }
        props = {
            "effect_on_doping":  rel.get("effect_on_doping", ""),
            "affected_property": rel.get("affected_property", ""),
            "property_changes":  rel.get("property_changes", {}),
        }
        cases.append({
            "id":                      f"{domain}_{i}",
            "domain":                  domain,
            "synthesis_conditions":    synthesis,
            "properties":              props,
            "ground_truth_synthesis":  synthesis,
            "ground_truth_properties": props,
        })
    return cases


# ---------------------------------------------------------------------------
# 4. System initialisation
# ---------------------------------------------------------------------------

def init_systems(kg_file_id: str) -> Dict:
    """Initialise all 4 ARIA variants with the in-domain KG path."""
    from src.variants.baseline_ollama   import BaselineOllama
    from src.variants.naive_kg_ollama   import NaiveKGOllama
    from src.variants.aria_search_ollama import ARIASearchOllama
    from src.variants.aria_full_ollama  import ARIAFullOllama

    systems = {}
    for name, cls, kwargs in [
        ("Baseline LLM",   BaselineOllama,   {"model": "Qwen/Qwen2-7B-Instruct"}),
        ("Naive KG+LLM",   NaiveKGOllama,    {"kg_file": kg_file_id, "model": "Qwen/Qwen2-7B-Instruct"}),
        ("Online KG+LLM",  ARIASearchOllama, {"kg_file": kg_file_id, "model": "Qwen/Qwen2-7B-Instruct"}),
        ("ARIA",           ARIAFullOllama,   {"kg_file": kg_file_id, "model": "Qwen/Qwen2-7B-Instruct"}),
    ]:
        try:
            systems[name] = cls(**kwargs)
            logger.info(f"  ✅ {name} initialised")
        except Exception as exc:
            logger.error(f"  ❌ {name} failed: {exc}")
            systems[name] = None

    return systems


# ---------------------------------------------------------------------------
# 5. Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_system(system_name, system, test_cases, task, emb_model):
    from src.evaluation.metrics import evaluate_prediction

    if system is None:
        logger.warning(f"Skipping {system_name} — not initialised")
        return []

    results = []
    for i, tc in enumerate(test_cases):
        logger.info(f"  [{i+1}/{len(test_cases)}] {system_name} | {task} | {tc['domain']}")
        try:
            if task == "forward":
                pred = system.forward_prediction(tc["synthesis_conditions"])
                gt   = tc["ground_truth_properties"]
            else:
                pred = system.inverse_design(tc["properties"])
                gt   = tc["ground_truth_synthesis"]

            scores = evaluate_prediction(pred, gt, emb_model)
            results.append({
                "system": system_name,
                "domain": tc["domain"],
                "task":   task,
                "test_id": tc["id"],
                **scores,
            })
        except Exception as exc:
            logger.error(f"    ❌ {exc}")
            results.append({
                "system": system_name, "domain": tc["domain"],
                "task": task, "test_id": tc["id"],
                "scientific_accuracy": 0.0, "functional_equivalence": 0.0,
                "reasoning_quality": 0.0, "completeness": 0.0,
                "interpretability": 0.0, "overall": 0.0,
                "error": str(exc),
            })
    return results


# ---------------------------------------------------------------------------
# 6. Leaderboard printer
# ---------------------------------------------------------------------------

METRICS = ["scientific_accuracy", "functional_equivalence",
           "reasoning_quality", "completeness", "interpretability", "overall"]

def print_leaderboard(df: pd.DataFrame):
    print("\n" + "=" * 75)
    print("VLLM ARIA EVALUATION LEADERBOARD  (kg_normalized_full, embedding judge)")
    print("=" * 75)
    for task in ["forward", "inverse"]:
        print(f"\n  {'FORWARD PREDICTION' if task == 'forward' else 'INVERSE DESIGN'}")
        print(f"  {'System':<22} {'Sci.Acc':>8} {'Funct.Eq':>9} "
              f"{'Reason':>8} {'Compl.':>8} {'Interp':>7} {'Overall':>8}")
        print(f"  {'-'*74}")
        sub = df[df["task"] == task]
        g = (sub.groupby("system")[METRICS].mean()
               .sort_values("overall", ascending=False))
        for sys_name, row in g.iterrows():
            print(f"  {sys_name:<22} {row['scientific_accuracy']:>8.3f} "
                  f"{row['functional_equivalence']:>9.3f} "
                  f"{row['reasoning_quality']:>8.3f} "
                  f"{row['completeness']:>8.3f} "
                  f"{row['interpretability']:>7.3f} "
                  f"{row['overall']:>8.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ARIA evaluation via vLLM")
    p.add_argument("--kg",      default=str(DEFAULT_KG),
                   help="Path to normalized full KG JSON")
    p.add_argument("--output",  default=str(DEFAULT_OUT),
                   help="Output CSV path")
    p.add_argument("--n_tests", type=int, default=10,
                   help="Test cases per domain (default 10)")
    p.add_argument("--gpus",    type=int, default=4,
                   help="tensor_parallel_size for vLLM")
    p.add_argument("--model",   default="Qwen/Qwen2-7B-Instruct",
                   help="HuggingFace model ID")
    p.add_argument("--seed",    type=int, default=RANDOM_SEED)
    return p.parse_args()


def main():
    args = parse_args()

    # ── patch Ollama → vLLM (must happen before any variant import) ──────────
    patch_ollama_with_vllm(args.model, args.gpus)

    # ── load & split KG ──────────────────────────────────────────────────────
    logger.info(f"\nLoading KG: {args.kg}")
    train_kg, test_kg = split_kg(Path(args.kg), seed=args.seed)

    # Write temp files so variant constructors can read them by path
    tmp_dir = REPO / "outputs" / "_tmp_vllm_eval"
    train_path = tmp_dir / "train_kg.json"
    test_path  = tmp_dir / "test_kg.json"
    write_tmp_kg(train_kg, train_path)
    write_tmp_kg(test_kg,  test_path)

    # ── embedding model (shared) ─────────────────────────────────────────────
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    emb_model = SentenceTransformer(EMBED_MODEL)

    # ── test cases ───────────────────────────────────────────────────────────
    cases_id  = extract_test_cases(train_kg, "in-domain",     args.n_tests)
    cases_ood = extract_test_cases(test_kg,  "out-of-domain", args.n_tests)
    logger.info(f"Test cases — in-domain: {len(cases_id)}, out-of-domain: {len(cases_ood)}")

    # ── init systems ─────────────────────────────────────────────────────────
    logger.info("\nInitialising ARIA variants...")
    systems = init_systems(str(train_path))

    # ── evaluation loop ──────────────────────────────────────────────────────
    all_results = []
    for domain_label, cases in [("in-domain", cases_id), ("out-of-domain", cases_ood)]:
        logger.info(f"\n{'='*60}\nDomain: {domain_label}\n{'='*60}")
        for task in ["forward", "inverse"]:
            logger.info(f"\n  Task: {task.upper()}")
            for sys_name, system in systems.items():
                all_results.extend(
                    evaluate_system(sys_name, system, cases, task, emb_model)
                )

    # ── save & report ─────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"\nSaved {len(df)} rows → {out}")

    print_leaderboard(df)

    # Clean up temp KG files
    for p in [train_path, test_path]:
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
