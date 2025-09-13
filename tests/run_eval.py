# tests/run_eval.py
from pathlib import Path
import sys, json, yaml, csv, glob
from metrics import edge_prf, path_exact_f1, inverse_topk, calib_bins, coverage

# ---------- set up paths ----------
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "script"
sys.path.insert(0, str(SCRIPTS_DIR))  # so we can import your engine module

# ---------- Import engine ----------
from causal_engine_stable import CausalReasoningEngine as Engine
#from causal_engine import CausalReasoningEngine as Engine

# If your engine needs a data file (adjust as needed)
ENGINE_JSON = REPO_ROOT / "outputs" / "combined_doping_data.json"

# ---------- Helper functions ----------
import re

def _canon(s: str) -> str:
    s = str(s)
    s = s.lower().strip()
    s = s.replace("° c", "°c").replace("ºc", "°c")
    s = re.sub(r"\s+", " ", s)
    return s

def _load_aliases(cid: str):
    p = TESTS_DIR / "cases" / "aliases"  / f"{cid}_aliases.yaml"
    if p.exists():
        import yaml as _yaml
        return _yaml.safe_load(open(p)) or {}
    return {}

def _map_node(name: str, aliases) -> str:
    name_c = _canon(name)
    nodes = (aliases.get("nodes") or {})
    return nodes.get(name_c, name_c)

def _norm_edge_list(edges, aliases):
    out = []
    for e in edges:
        out.append({
            **e,
            "source": _map_node(e.get("source",""), aliases),
            "target": _map_node(e.get("target",""), aliases),
        })
    return out

def _norm_path_list(paths, aliases):
    return [[_map_node(n, aliases) for n in path] for path in paths]

def _norm_params(d: dict, aliases) -> dict:
    # normalize recipe param keys/values so they match gold
    key_map = { _canon(k): v for k,v in (aliases.get("param_keys") or {}).items() }
    val_map = { _canon(k): v for k,v in (aliases.get("param_values") or {}).items() }
    out = {}
    for k, v in (d or {}).items():
        k_c = _canon(k)
        k_norm = key_map.get(k_c, k_c)
        if isinstance(v, str):
            v_c = _canon(v)
            out[k_norm] = val_map.get(v_c, v)  # keep original if no alias
        else:
            out[k_norm] = v
    return out
# ---------- Helper functions ENDS ----------

def _load_yaml_one(pattern: str):
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No YAML found for pattern: {pattern}")
    with open(paths[0], "r") as f:
        return yaml.safe_load(f)

def _gold_bool_alignment(pred_edges, gold_edges):
    """Map each predicted edge to correctness (for calibration)."""
    def key(e): return (str(e.get("source","")).lower(), str(e.get("target","")).lower())
    gold_set = {key(e) for e in gold_edges}
    return [key(e) in gold_set for e in pred_edges]

def eval_case(case_yaml_path: Path, engine: Engine):
    meta = yaml.safe_load(open(case_yaml_path, "r"))
    cid = meta["id"]

    gold_edges = json.load(open(TESTS_DIR / "gold" / f"{cid}_edges.json"))
    gold_paths = json.load(open(TESTS_DIR / "gold" / f"{cid}_paths.json"))
    gold_recipe = json.load(open(TESTS_DIR / "gold" / f"{cid}_recipe.json"))

    # ---------- Forward evaluation ----------
    fwd_pred = engine.forward_prediction(meta["targets"]["forward"]["synthesis"])
    
    # dump preidction for checks
    (PRED_DIR := (TESTS_DIR / "preds")).mkdir(exist_ok=True)
    with open(PRED_DIR / f"{cid}_forward.json", "w") as f:
        json.dump(fwd_pred, f, indent=2)
    
    # scoring begins (normalize for name matching before scoring)
    aliases = _load_aliases(cid) 
    pred_edges = fwd_pred.get("causal_edges", [])
    pred_paths = fwd_pred.get("causal_paths", [])
    conf_edges = fwd_pred.get(
        "edge_confidences",
        [e.get("confidence", 0.5) for e in pred_edges]
    )
    
    pred_edges_n = _norm_edge_list(pred_edges, aliases)
    pred_paths_n = _norm_path_list(pred_paths, aliases)
    
    gold_edges_n = _norm_edge_list(gold_edges, aliases)
    gold_paths_n = _norm_path_list(gold_paths, aliases)
    
    eP, eR, eF1 = edge_prf(pred_edges_n, gold_edges_n)
    pExact, pF1 = path_exact_f1(pred_paths_n, gold_paths_n)
    gold_set = {(g["source"], g["target"]) for g in gold_edges_n}
    #ece_pack = calib_bins(conf_edges, _gold_bool_alignment(pred_edges, gold_edges))
    ece_pack = calib_bins(conf_edges, [ (e["source"], e["target"]) in gold_set for e in pred_edges_n ])
    cov = coverage(pred_paths_n)

    # ---------- Inverse evaluation ----------
    inv_pred = engine.inverse_design(meta["targets"]["inverse"]["properties"])
     
    with open(PRED_DIR / f"{cid}_inverse.json", "w") as f:
        json.dump(inv_pred, f, indent=2)
    
    raw_cands = inv_pred.get("recipe_candidates", [])
    pred_recipes = []
    for c in raw_cands:
        params = c.get("params", c)
        pred_recipes.append({"params": _norm_params(params, aliases), "score": c.get("score", 0.0)})
    #pred_recipes = inv_pred.get("recipe_candidates", [])
    topk = inverse_topk(pred_recipes, gold_recipe)

    out = {
        "id": cid,
        "edge_P": eP, "edge_R": eR, "edge_F1": eF1,
        "path_exact": pExact, "path_F1": pF1,
        "coverage": cov, "ECE": ece_pack["ECE"],
    }
    for k, v in topk.items():
        out[f"inverse{k}"] = v
    return out

def main():
    # ---- Instantiate your engine ----
    # If your constructor expects the JSON path:
    engine = Engine(str(ENGINE_JSON))
    # If your constructor takes no args, use instead:
    # engine = Engine()

    # ---- Collect cases ----
    case_files = sorted((TESTS_DIR / "cases").glob("CASE*.yaml"))
    if not case_files:
        raise SystemExit("No cases found in tests/cases/*.yaml")

    rows = [eval_case(p, engine) for p in case_files]

    # ---- Write results ----
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "benchmark_results.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_csv} with {len(rows)} case(s).")

    # ---- Small leaderboard ----
    rows_sorted = sorted(rows, key=lambda r: r.get("edge_F1", 0.0), reverse=True)
    print("\nTop cases by Edge F1:")
    for r in rows_sorted[:5]:
        print(f"{r['id']}: Edge F1={r['edge_F1']}, Path F1={r['path_F1']}, inverse@3={r.get('inverse@3', False)}")

if __name__ == "__main__":
    main()
