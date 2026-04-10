"""
Rebuttal evaluation: score SelfRAG (and optionally other engines) with VLLMJudge.

Reads self_rag_raw_outputs.csv produced by run_self_rag_eval.py,
scores each prediction on 4 domain-specific metrics using a vLLM judge,
and writes a summary CSV + printed leaderboard.

Usage:
    # Score SelfRAG only (reads existing CSV):
    python src/rebuttal/run_rebuttal_eval.py

    # Use a bigger judge model:
    python src/rebuttal/run_rebuttal_eval.py --judge_model Qwen/Qwen2.5-72B-Instruct --judge_gpus 4

    # Score additional engine outputs too (CSV must have same schema):
    python src/rebuttal/run_rebuttal_eval.py --extra_csvs outputs/aria_outputs.csv:ARIA
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rebuttal.vllm_judge import VLLMJudge

DEFAULT_SELFRAG_CSV = REPO_ROOT / "outputs" / "self_rag_raw_outputs.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "rebuttal_eval_scores.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_experiments(path: Path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "doping_experiments" in data:
        return {e["experiment_id"]: e for e in data["doping_experiments"]}
    if isinstance(data, list):
        return {e["experiment_id"]: e for e in data}
    raise ValueError(f"Unexpected format: {path}")


def score_engine_outputs(
    df: pd.DataFrame,
    label: str,
    judge: VLLMJudge,
    exp_map: dict,
) -> pd.DataFrame:
    """
    Score forward_prediction and inverse_design outputs for one engine.

    df must contain columns: experiment_id, fwd_selfrag_clean, inv_selfrag_clean
    (or fwd_clean / inv_clean for non-selfrag engines).
    """
    fwd_col = next((c for c in ["fwd_selfrag_clean", "fwd_clean"] if c in df.columns), None)
    inv_col = next((c for c in ["inv_selfrag_clean", "inv_clean"] if c in df.columns), None)

    if not fwd_col or not inv_col:
        raise ValueError(f"Cannot find fwd/inv output columns in {label} CSV. "
                         f"Columns: {df.columns.tolist()}")

    records = []
    n = len(df)
    for i, row in df.iterrows():
        eid = row["experiment_id"]
        exp = exp_map.get(eid)
        if exp is None:
            print(f"  [warn] experiment_id '{eid}' not found in test data, skipping")
            continue

        synthesis = exp.get("synthesis_conditions", {})
        properties = exp.get("property_changes", {})

        print(f"  [{i+1}/{n}] {eid} | {row.get('host_material','')} + {row.get('dopant','')}")

        base = {
            "experiment_id": eid,
            "host_material": row.get("host_material", ""),
            "dopant": row.get("dopant", ""),
            "engine": label,
        }

        # --- Forward prediction ---
        fwd_pred = {"reasoning": str(row[fwd_col])}
        fwd_result = judge.evaluate_all_metrics(
            query=synthesis,
            prediction=fwd_pred,
            ground_truth=properties,
        )
        fwd_scores = fwd_result["metric_scores"]
        rec_fwd = {**base, "task": "forward_prediction",
                   "overall_score": fwd_result["overall_score"]}
        for mk, v in fwd_scores.items():
            rec_fwd[f"{mk}_score"] = v["score"]
            rec_fwd[f"{mk}_justification"] = v.get("justification", "")
        records.append(rec_fwd)

        # --- Inverse design ---
        inv_pred = {"reasoning": str(row[inv_col])}
        inv_result = judge.evaluate_all_metrics(
            query=properties,
            prediction=inv_pred,
            ground_truth=synthesis,
        )
        inv_scores = inv_result["metric_scores"]
        rec_inv = {**base, "task": "inverse_design",
                   "overall_score": inv_result["overall_score"]}
        for mk, v in inv_scores.items():
            rec_inv[f"{mk}_score"] = v["score"]
            rec_inv[f"{mk}_justification"] = v.get("justification", "")
        records.append(rec_inv)

    return pd.DataFrame(records)


def print_leaderboard(scores_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("REBUTTAL EVALUATION LEADERBOARD")
    print("=" * 70)

    metrics = ["processing_feasibility", "structure_emergence",
               "property_consistency", "causal_psp_reasoning"]
    max_scores = [40, 30, 20, 10]

    for task in ["forward_prediction", "inverse_design"]:
        print(f"\n{task.replace('_', ' ').upper()}")
        print("-" * 50)
        task_df = scores_df[scores_df["task"] == task]

        rows = []
        for engine, grp in task_df.groupby("engine"):
            row = {"Engine": engine,
                   "Overall /100": f"{grp['overall_score'].mean():.1f}"}
            for mk, mx in zip(metrics, max_scores):
                col = f"{mk}_score"
                if col in grp.columns:
                    row[mk[:8]] = f"{grp[col].mean():.1f}/{mx}"
            rows.append(row)

        if rows:
            header = list(rows[0].keys())
            print("  ".join(f"{h:<20}" for h in header))
            for r in sorted(rows, key=lambda x: float(x["Overall /100"]), reverse=True):
                print("  ".join(f"{str(r[h]):<20}" for h in header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Score SelfRAG (and other engines) with VLLMJudge")
    p.add_argument("--selfrag_csv", default=str(DEFAULT_SELFRAG_CSV),
                   help="Path to self_rag_raw_outputs.csv")
    p.add_argument("--test_data",
                   default=str(REPO_ROOT / "data/KG/outputs/test_doping_data.json"),
                   help="Path to test_doping_data.json")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Output CSV path for scores")
    p.add_argument("--judge_model", default="Qwen/Qwen2-7B-Instruct",
                   help="HuggingFace model ID for the judge")
    p.add_argument("--judge_gpus", type=int, default=4,
                   help="tensor_parallel_size for judge model")
    p.add_argument("--extra_csvs", nargs="*", default=[],
                   metavar="FILE:LABEL",
                   help="Extra engine output CSVs to score. Format: path.csv:EngineName")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Test data  : {args.test_data}")
    print(f"SelfRAG CSV: {args.selfrag_csv}")
    print(f"Output     : {out_path}")
    print(f"Judge      : {args.judge_model} (tp={args.judge_gpus})\n")

    exp_map = load_test_experiments(Path(args.test_data))
    print(f"Loaded {len(exp_map)} test experiments\n")

    judge = VLLMJudge(model=args.judge_model, tensor_parallel_size=args.judge_gpus)

    all_dfs = []

    # Score SelfRAG
    print("=== Scoring SelfRAG ===")
    selfrag_df = pd.read_csv(args.selfrag_csv)
    all_dfs.append(score_engine_outputs(selfrag_df, "SelfRAG", judge, exp_map))

    # Score any extra engines
    for spec in args.extra_csvs:
        if ":" not in spec:
            print(f"[warn] Skipping '{spec}' — format must be path.csv:Label")
            continue
        fpath, label = spec.rsplit(":", 1)
        print(f"\n=== Scoring {label} ===")
        extra_df = pd.read_csv(fpath)
        all_dfs.append(score_engine_outputs(extra_df, label, judge, exp_map))

    scores_df = pd.concat(all_dfs, ignore_index=True)
    scores_df.to_csv(out_path, index=False)
    print(f"\nScores saved to {out_path}")

    print_leaderboard(scores_df)


if __name__ == "__main__":
    main()
