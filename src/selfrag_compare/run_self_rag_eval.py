"""
Standalone evaluation runner for the SelfRAG baseline.

Runs forward_prediction and inverse_design on all 23 test experiments
and saves raw outputs to a CSV — no Gemini API required at this stage.

To also score with the LLM judge, use evaluation_multi.py afterwards:

    cd src/project/test
    python evaluation_multi.py \\
        --test_data ../../../data/KG/outputs/test_doping_data.json \\
        --training_graph ../../../data/KG/outputs/combined_doping_data.json \\
        --engines "SelfRAG:src.rebuttal.self_rag_engine" \\
        --output_prefix self_rag_vs_baseline

Usage:
    cd <repo_root>
    python src/rebuttal/run_self_rag_eval.py
    python src/rebuttal/run_self_rag_eval.py --test_data path/to/test.json --output outputs/self_rag_raw.csv
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rebuttal.self_rag_engine import CausalReasoningEngine

DEFAULT_TEST_DATA = REPO_ROOT / "data" / "KG" / "outputs" / "test_doping_data.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "self_rag_raw_outputs.csv"


def load_test_experiments(path: Path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "doping_experiments" in data:
        return data["doping_experiments"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected format in {path}")


def run_eval(args):
    test_path = Path(args.test_data)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Test data : {test_path}")
    print(f"Output    : {out_path}")

    experiments = load_test_experiments(test_path)
    print(f"Loaded {len(experiments)} test experiments\n")

    engine = CausalReasoningEngine(
        training_graph_file=args.training_graph,
        use_kg_passages=args.use_kg_passages,
        top_k_retrieve=args.top_k,
        tensor_parallel_size=args.gpus,
    )

    records = []
    for i, exp in enumerate(experiments):
        exp_id = exp.get("experiment_id", f"exp_{i}")
        host = exp.get("host_material", "")
        dopant = exp.get("dopant", {}).get("element", "")
        synthesis = exp.get("synthesis_conditions", {})
        properties = exp.get("property_changes", {})

        print(f"[{i+1}/{len(experiments)}] {exp_id}  {host} + {dopant}")

        # Forward prediction
        t0 = time.time()
        try:
            fwd = engine.forward_prediction(synthesis)
            fwd_text = fwd.get("reasoning", "")
            fwd_raw = fwd.get("raw_selfrag_output", "")
            fwd_ok = True
        except Exception as e:
            print(f"  forward_prediction failed: {e}")
            fwd_text = fwd_raw = ""
            fwd_ok = False
        fwd_time = time.time() - t0

        # Inverse design
        t0 = time.time()
        try:
            inv = engine.inverse_design(properties)
            inv_text = inv.get("reasoning", "")
            inv_raw = inv.get("raw_selfrag_output", "")
            inv_ok = True
        except Exception as e:
            print(f"  inverse_design failed: {e}")
            inv_text = inv_raw = ""
            inv_ok = False
        inv_time = time.time() - t0

        records.append({
            "experiment_id": exp_id,
            "host_material": host,
            "dopant": dopant,
            # ground truth (serialized for reference)
            "gt_synthesis_conditions": json.dumps(synthesis),
            "gt_property_changes": json.dumps(properties),
            # forward prediction
            "fwd_selfrag_raw": fwd_raw,
            "fwd_selfrag_clean": fwd_text,
            "fwd_ok": fwd_ok,
            "fwd_time_s": round(fwd_time, 2),
            # inverse design
            "inv_selfrag_raw": inv_raw,
            "inv_selfrag_clean": inv_text,
            "inv_ok": inv_ok,
            "inv_time_s": round(inv_time, 2),
        })

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    # Quick summary
    print(f"\nForward prediction success rate : {df['fwd_ok'].mean():.0%}")
    print(f"Inverse design success rate     : {df['inv_ok'].mean():.0%}")
    print(f"Avg forward time (s)            : {df['fwd_time_s'].mean():.1f}")
    print(f"Avg inverse time (s)            : {df['inv_time_s'].mean():.1f}")


def parse_args():
    p = argparse.ArgumentParser(description="SelfRAG baseline evaluation")
    p.add_argument("--test_data", default=str(DEFAULT_TEST_DATA),
                   help="Path to test_doping_data.json")
    p.add_argument("--training_graph", default=str(REPO_ROOT / "data/KG/outputs/combined_doping_data.json"),
                   help="Path to combined_doping_data.json (for optional KG passages)")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Output CSV path")
    p.add_argument("--use_kg_passages", action="store_true",
                   help="Also add serialized KG experiments to retrieval corpus")
    p.add_argument("--top_k", type=int, default=1,
                   help="Number of passages to retrieve per query")
    p.add_argument("--gpus", type=int, default=4,
                   help="tensor_parallel_size for vLLM (default: 4)")
    return p.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
