"""
Fair comparison: SelfRAG vs ARIA variants on identical conditions.

Uses EXACTLY the same pipeline as src/evaluation/run_phase1_evaluation.py:
  - Same test data  : causal_relationships from combined_doping_data.json (first 10)
  - Same metrics    : evaluate_prediction() from src/evaluation/metrics.py
                      (embedding-based, all-MiniLM-L6-v2, no API key needed)
  - Same format     : matches phase1_raw_results_clean.csv column-for-column

Adds SelfRAG as a new row and appends to the existing ARIA results so the
final DataFrame can be used directly in the visualisation / table scripts.

Both generation model (qwen2:7b → selfrag_llama2_13b via vLLM) and
judge (embedding metrics) are deployed locally — no external APIs required.

Usage:
    python src/rebuttal/run_fair_comparison.py
    python src/rebuttal/run_fair_comparison.py --n_tests 10 --gpus 4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO))

from src.evaluation.metrics import evaluate_prediction
from src.rebuttal.self_rag_engine import CausalReasoningEngine

KG_FILE        = REPO / "data/KG/outputs/combined_doping_data.json"
ARIA_RESULTS   = REPO / "results/phase1/phase1_raw_results_clean.csv"
OUTPUT_CSV     = REPO / "outputs/fair_comparison_results.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Load test cases — identical to Phase1Evaluator._load_test_data()
# ---------------------------------------------------------------------------

def load_test_cases(kg_file: Path, domain: str, n: int) -> list:
    """
    Replicate the exact test-case extraction used in run_phase1_evaluation.py.
    Pulls the first `n` causal_relationships and formats them as
    {id, domain, synthesis_conditions, properties,
     ground_truth_synthesis, ground_truth_properties}.
    """
    with open(kg_file) as f:
        data = json.load(f)

    relationships = data.get("causal_relationships", [])

    cases = []
    for i, rel in enumerate(relationships[:n]):
        if not isinstance(rel, dict):
            continue

        synthesis = {
            "cause_parameter": rel.get("cause_parameter", ""),
            "method":          rel.get("method", ""),
            "temperature":     rel.get("temperature", ""),
        }
        properties = {
            "effect_on_doping":  rel.get("effect_on_doping", ""),
            "affected_property": rel.get("affected_property", ""),
            "property_changes":  rel.get("property_changes", {}),
        }

        cases.append({
            "id":                    f"{domain}_{i}",
            "domain":                domain,
            "synthesis_conditions":  synthesis,
            "properties":            properties,
            "ground_truth_synthesis":  synthesis,
            "ground_truth_properties": properties,
        })

    return cases


# ---------------------------------------------------------------------------
# Query builders for the causal_relationships format
# ---------------------------------------------------------------------------

def _synthesis_query(sc: dict) -> str:
    """Forward-prediction query from causal_relationship synthesis fields."""
    cause  = sc.get("cause_parameter", "")
    method = sc.get("method", "")
    temp   = sc.get("temperature", "")

    q = "What are the structural and electronic property changes that result from"
    if cause:
        q += f" {cause}"
    if method:
        q += f" using {method}"
    if temp:
        q += f" at {temp}"
    q += " in a 2D material doping experiment?"
    return q


def _properties_query(props: dict) -> str:
    """Inverse-design query from causal_relationship property fields."""
    effect   = props.get("effect_on_doping", "")
    affected = props.get("affected_property", "")

    q = "What synthesis conditions (processing method, temperature, cause parameter) are needed to achieve"
    if effect:
        q += f" {effect}"
    if affected:
        q += f" affecting {affected}"
    q += " in a 2D material?"
    return q


# ---------------------------------------------------------------------------
# Run SelfRAG on the test cases
# ---------------------------------------------------------------------------

def run_selfrag(
    engine: CausalReasoningEngine,
    test_cases: list,
    embedding_model: SentenceTransformer,
    domain: str,
) -> list:
    results = []
    n = len(test_cases)

    for i, tc in enumerate(test_cases):
        print(f"  [{i+1}/{n}] SelfRAG — {tc['task']} — {domain}")

        # Override the internal query builder for this data format
        sc    = tc["synthesis_conditions"]
        props = tc["properties"]

        if tc["task"] == "forward":
            # Temporarily monkey-patch the query for this format
            query = _synthesis_query(sc)
            raw   = engine._selfrag_loop(query)
            from src.rebuttal.self_rag_engine import strip_reflection_tokens
            clean = strip_reflection_tokens(raw)
            prediction = {
                "predicted_properties": {"description": clean},
                "reasoning":            clean,
                "confidence":           0.5,
            }
            ground_truth = tc["ground_truth_properties"]

        else:  # inverse
            query = _properties_query(props)
            raw   = engine._selfrag_loop(query)
            from src.rebuttal.self_rag_engine import strip_reflection_tokens
            clean = strip_reflection_tokens(raw)
            prediction = {
                "suggested_synthesis_conditions": {"description": clean},
                "reasoning":                      clean,
                "confidence":                     0.5,
            }
            ground_truth = tc["ground_truth_synthesis"]

        scores = evaluate_prediction(prediction, ground_truth, embedding_model)

        results.append({
            "system":  "SelfRAG",
            "domain":  domain,
            "task":    tc["task"],
            "test_id": tc["id"],
            **scores,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_tests", type=int, default=10,
                   help="Test cases per domain (default 10 — matches ARIA)")
    p.add_argument("--gpus", type=int, default=4,
                   help="tensor_parallel_size for SelfRAG vLLM")
    p.add_argument("--output", default=str(OUTPUT_CSV))
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    emb_model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"\nLoading test cases from {KG_FILE.name}  (n={args.n_tests} per domain)")
    # ARIA phase1 uses the same KG file for both in-domain and out-of-domain
    # (it's a limitation of the current setup — same data, different label)
    cases_id  = load_test_cases(KG_FILE, "in-domain",     args.n_tests)
    cases_ood = load_test_cases(KG_FILE, "out-of-domain",  args.n_tests)
    print(f"  in-domain: {len(cases_id)} cases, out-of-domain: {len(cases_ood)} cases")

    # Attach task label — evaluate both forward and inverse
    all_cases = []
    for domain_cases in [cases_id, cases_ood]:
        for tc in domain_cases:
            for task in ["forward", "inverse"]:
                all_cases.append({**tc, "task": task})

    print(f"\nInitialising SelfRAG engine (gpus={args.gpus}) ...")
    engine = CausalReasoningEngine(tensor_parallel_size=args.gpus)

    print(f"\nRunning SelfRAG on {len(all_cases)} test cases ...")
    selfrag_results = []
    for tc in all_cases:
        domain = tc["domain"]
        selfrag_results.extend(
            run_selfrag(engine, [tc], emb_model, domain)
        )

    selfrag_df = pd.DataFrame(selfrag_results)

    # Load existing ARIA results and append SelfRAG
    print(f"\nLoading existing ARIA results from {ARIA_RESULTS.name}")
    aria_df = pd.read_csv(ARIA_RESULTS)

    combined = pd.concat([aria_df, selfrag_df], ignore_index=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"Saved {len(combined)} rows → {out}")

    # Print leaderboard
    print("\n" + "=" * 65)
    print("FAIR COMPARISON LEADERBOARD  (same test data + same metrics)")
    print("=" * 65)
    for task in ["forward", "inverse"]:
        print(f"\n  {'FORWARD PREDICTION' if task=='forward' else 'INVERSE DESIGN'}")
        print(f"  {'System':<20} {'Sci.Acc':>8} {'Funct.Eq':>9} "
              f"{'Reason':>8} {'Complete':>9} {'Interp':>7} {'Overall':>8}")
        print(f"  {'-'*72}")
        sub = combined[combined["task"] == task]
        g = sub.groupby("system")[
            ["scientific_accuracy","functional_equivalence",
             "reasoning_quality","completeness","interpretability","overall"]
        ].mean().sort_values("overall", ascending=False)
        for sys_name, row in g.iterrows():
            marker = " ◄" if sys_name == "SelfRAG" else ""
            print(f"  {sys_name:<20} {row['scientific_accuracy']:>8.3f} "
                  f"{row['functional_equivalence']:>9.3f} "
                  f"{row['reasoning_quality']:>8.3f} "
                  f"{row['completeness']:>9.3f} "
                  f"{row['interpretability']:>7.3f} "
                  f"{row['overall']:>8.3f}{marker}")


if __name__ == "__main__":
    main()
