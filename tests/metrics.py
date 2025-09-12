
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import math, re, numpy as np

def _edge_key(e: Dict[str, Any]) -> Tuple[str,str]:
    return (str(e.get("source","")).strip().lower(), str(e.get("target","")).strip().lower())

def edge_prf(pred_edges: List[Dict[str, Any]], gold_edges: List[Dict[str, Any]]) -> Tuple[float,float,float]:
    """Compute precision/recall/F1 on edges ignoring mechanism text (exact source/target match)."""
    P = {_edge_key(e) for e in pred_edges}
    G = {_edge_key(e) for e in gold_edges}
    tp = len(P & G); fp = len(P - G); fn = len(G - P)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return round(prec,3), round(rec,3), round(f1,3)

def path_exact_f1(pred_paths: List[List[str]], gold_paths: List[List[str]]) -> Tuple[float,float]:
    """Exact path match rate + relaxed F1 over constituent edges (best match)."""
    def edges(path): return {(a.strip().lower(), b.strip().lower()) for a,b in zip(path[:-1], path[1:])}
    exact = 0
    f1s = []
    GP = [edges(p) for p in gold_paths]
    for p in pred_paths:
        if p in gold_paths:
            exact += 1
        Pe = edges(p)
        # best F1 vs any gold path
        best = 0.0
        for Ge in GP:
            tp = len(Pe & Ge); fp = len(Pe - Ge); fn = len(Ge - Pe)
            pr = tp/(tp+fp) if (tp+fp) else 0.0
            rc = tp/(tp+fn) if (tp+fn) else 0.0
            f1 = 2*pr*rc/(pr+rc) if (pr+rc) else 0.0
            best = max(best, f1)
        f1s.append(best)
    exact_rate = exact/len(pred_paths) if pred_paths else 0.0
    mean_f1 = sum(f1s)/len(f1s) if f1s else 0.0
    return round(exact_rate,3), round(mean_f1,3)

def coverage(pred_paths: List[List[str]]) -> float:
    """Fraction of cases with any non-empty path."""
    return 1.0 if any(pred_paths) else 0.0

def _norm_num(x):
    try: return float(x)
    except: return None

def _match_param(key, gold_val, cand_val) -> bool:
    """Tolerance-aware parameter matcher. Supports numeric ranges and regex for strings."""
    if isinstance(gold_val, list) and len(gold_val)==2:
        lo, hi = gold_val
        x = _norm_num(cand_val)
        return (x is not None) and (lo <= x <= hi)
    if isinstance(gold_val, str):
        # treat as regex OR pipe-separated enums
        pattern = gold_val if any(ch in gold_val for ch in ".*?|[]()") else "^(" + gold_val + ")$"
        return re.search(pattern, str(cand_val), flags=re.IGNORECASE) is not None
    return str(gold_val).lower() == str(cand_val).lower()

def inverse_topk(recipe_candidates: List[Dict[str,Any]], gold_recipe: Dict[str,Any]) -> Dict[str,bool]:
    """Return hit@k dictionary e.g., {'@1': True, '@3': False, '@5': False}"""
    params = gold_recipe.get("params", {})
    ks = sorted({1,3,5, int(gold_recipe.get("match_top_k",3))})
    hits = {f"@{k}": False for k in ks}
    for rank, cand in enumerate(recipe_candidates[:max(ks)], start=1):
        cand_params = cand.get("params", cand)  # allow flat dict
        ok = all(_match_param(k, v, cand_params.get(k)) for k,v in params.items())
        if ok:
            for k in ks:
                if rank <= k: hits[f"@{k}"] = True
            break
    return hits

def calib_bins(confidences: List[float], gold_is_correct: List[bool], num_bins: int=10) -> Dict[str, Any]:
    """Simple ECE. gold_is_correct is a boolean list aligned to confidences."""
    if not confidences:
        return {"ECE": 0.0, "bins": []}
    bins = [[] for _ in range(num_bins)]
    for c, ok in zip(confidences, gold_is_correct):
        i = min(num_bins-1, max(0, int(c * num_bins)))
        bins[i].append((c, 1.0 if ok else 0.0))
    ece = 0.0
    out_bins = []
    for i, bucket in enumerate(bins):
        if not bucket: 
            out_bins.append({"bin": i, "count": 0, "conf": 0.0, "acc": 0.0})
            continue
        conf = sum(c for c,_ in bucket)/len(bucket)
        acc  = sum(ok for _,ok in bucket)/len(bucket)
        ece += (len(bucket)/len(confidences))*abs(acc-conf)
        out_bins.append({"bin": i, "count": len(bucket), "conf": round(conf,3), "acc": round(acc,3)})
    return {"ECE": round(ece,3), "bins": out_bins}
