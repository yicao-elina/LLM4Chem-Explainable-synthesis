"""
Visualization: SelfRAG vs ARIA variants comparison.

Produces outputs/rebuttal_comparison.pdf  (and .png)

Layout:
  (A) Overall score comparison across all systems (forward + inverse)
  (B) SelfRAG per-metric breakdown (normalized 0-1)
  (C) ARIA per-metric breakdown for context

Note on comparability:
  ARIA scores come from phase1_raw_results_clean.csv (5 metrics, 0-1 scale, Gemini judge).
  SelfRAG scores come from rebuttal_eval_scores.csv (4 metrics, 0-100 scale, Qwen2-7B judge).
  Overall scores are normalized to 0-1 for the main bar chart. Per-metric panels
  show each system's own rubric separately to avoid misleading cross-rubric comparison.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).parents[2]
OUT_DIR = REPO / "outputs"

# ── colour palette (JHU-inspired, colourblind-safe) ─────────────────────────
C = {
    "ARIA":          "#002D72",   # Heritage Blue
    "Baseline LLM":  "#68ACE5",   # Spirit Blue
    "Online KG+LLM": "#9EA2A2",   # Grey
    "Naive KG+LLM":  "#C8C9C7",   # Light grey
    "SelfRAG":       "#E03C31",   # Sizzling Red
}
TASK_ALPHA = {"forward": 1.0, "inverse": 0.55}

SYSTEMS_ORDER = ["SelfRAG", "Naive KG+LLM", "Baseline LLM", "Online KG+LLM", "ARIA"]


# ── load data ────────────────────────────────────────────────────────────────

def load_aria():
    df = pd.read_csv(REPO / "results/phase1/phase1_raw_results_clean.csv")
    # average over in-domain + out-of-domain
    g = (df.groupby(["system", "task"])
           [["scientific_accuracy", "functional_equivalence",
             "reasoning_quality", "completeness", "interpretability", "overall"]]
           .agg(["mean", "sem"])
           .reset_index())
    g.columns = ["_".join(c).strip("_") for c in g.columns]
    return g


def load_selfrag():
    df = pd.read_csv(REPO / "outputs/rebuttal_eval_scores.csv")
    metrics_raw = {
        "processing_feasibility_score": 40,
        "structure_emergence_score":    30,
        "property_consistency_score":   20,
        "causal_psp_reasoning_score":   10,
        "overall_score":               100,
    }
    # normalise to 0-1
    for col, mx in metrics_raw.items():
        df[col + "_norm"] = df[col] / mx

    g = (df.groupby("task")
           [[c + "_norm" for c in metrics_raw]]
           .agg(["mean", "sem"])
           .reset_index())
    g.columns = ["_".join(c).strip("_") for c in g.columns]
    # rename task values to match ARIA convention
    g["task"] = g["task"].str.replace("forward_prediction", "forward")\
                          .str.replace("inverse_design",     "inverse")
    g["system"] = "SelfRAG"
    return g


# ── panel A: overall score bar chart ────────────────────────────────────────

def plot_overall(ax, aria_g, selfrag_g):
    tasks   = ["forward", "inverse"]
    task_labels = ["Forward Prediction", "Inverse Design"]
    systems = SYSTEMS_ORDER
    n_sys   = len(systems)
    width   = 0.35
    x       = np.arange(n_sys)

    # build lookup: (system, task) -> (mean, sem)
    lookup = {}
    for _, row in aria_g.iterrows():
        lookup[(row["system"], row["task"])] = (row["overall_mean"], row["overall_sem"])
    for _, row in selfrag_g.iterrows():
        col_mean = "overall_score_norm_mean"
        col_sem  = "overall_score_norm_sem"
        lookup[("SelfRAG", row["task"])] = (row[col_mean], row[col_sem])

    offsets = [-width/2, width/2]
    hatches = ["", "///"]
    for t_idx, (task, label) in enumerate(zip(tasks, task_labels)):
        means = [lookup.get((s, task), (0, 0))[0] for s in systems]
        sems  = [lookup.get((s, task), (0, 0))[1] for s in systems]
        bars = ax.bar(x + offsets[t_idx], means, width,
                      color=[C[s] for s in systems],
                      alpha=TASK_ALPHA[task],
                      hatch=hatches[t_idx],
                      edgecolor="white", linewidth=0.5,
                      yerr=sems, capsize=3, error_kw=dict(elinewidth=0.8))
        for bar, m in zip(bars, means):
            if m > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                        f"{m:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Overall Score (0–1)", fontsize=9)
    ax.set_title("(A)  Overall Score Comparison", fontsize=10, fontweight="bold", loc="left")
    ax.set_ylim(0, 0.82)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # legend for tasks
    patch_fwd = mpatches.Patch(facecolor="grey", alpha=1.0, label="Forward Prediction")
    patch_inv = mpatches.Patch(facecolor="grey", alpha=0.55, hatch="///",
                               edgecolor="grey", label="Inverse Design")
    ax.legend(handles=[patch_fwd, patch_inv], fontsize=8, loc="upper right",
              framealpha=0.8)

    # annotate evaluation source
    ax.text(0.01, 0.97,
            "ARIA/Baseline/Online/Naive: evaluated with Gemini judge (5 metrics, 0–1)\n"
            "SelfRAG: evaluated with Qwen2-7B judge (4 metrics, normalised to 0–1)",
            transform=ax.transAxes, fontsize=6.5, va="top",
            color="#555555", style="italic")


# ── panel B: SelfRAG metric breakdown ───────────────────────────────────────

def plot_selfrag_metrics(ax, selfrag_g):
    metric_cols = [
        ("processing_feasibility_score_norm_mean", "processing_feasibility_score_norm_sem",
         "Processing\nFeasibility\n(/40)"),
        ("structure_emergence_score_norm_mean",    "structure_emergence_score_norm_sem",
         "Structure\nEmergence\n(/30)"),
        ("property_consistency_score_norm_mean",   "property_consistency_score_norm_sem",
         "Property\nConsistency\n(/20)"),
        ("causal_psp_reasoning_score_norm_mean",   "causal_psp_reasoning_score_norm_sem",
         "Causal PSP\nReasoning\n(/10)"),
    ]
    tasks = ["forward", "inverse"]
    x = np.arange(len(metric_cols))
    width = 0.35
    offsets = [-width/2, width/2]
    alphas  = [1.0, 0.55]
    hatches = ["", "///"]
    labels  = ["Forward Prediction", "Inverse Design"]

    row_dict = {row["task"]: row for _, row in selfrag_g.iterrows()}

    for t_idx, (task, label) in enumerate(zip(tasks, labels)):
        row = row_dict.get(task)
        if row is None:
            continue
        means = [row[m] for m, _, _ in metric_cols]
        sems  = [row[s] for _, s, _ in metric_cols]
        ax.bar(x + offsets[t_idx], means, width,
               color=C["SelfRAG"], alpha=alphas[t_idx],
               hatch=hatches[t_idx], edgecolor="white", linewidth=0.5,
               yerr=sems, capsize=3, error_kw=dict(elinewidth=0.8),
               label=label)
        for xi, (m, _) in zip(x + offsets[t_idx], zip(means, sems)):
            if m > 0.02:
                ax.text(xi, m + 0.012, f"{m:.2f}",
                        ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, _, lbl in metric_cols], fontsize=8)
    ax.set_ylabel("Normalised Score (0–1)", fontsize=9)
    ax.set_title("(B)  SelfRAG — Per-Metric Breakdown", fontsize=10,
                 fontweight="bold", loc="left")
    ax.set_ylim(0, 0.65)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.8)


# ── panel C: ARIA metric breakdown ──────────────────────────────────────────

def plot_aria_metrics(ax, aria_g):
    metric_cols = [
        ("scientific_accuracy_mean",     "scientific_accuracy_sem",     "Scientific\nAccuracy"),
        ("functional_equivalence_mean",  "functional_equivalence_sem",  "Functional\nEquiv."),
        ("reasoning_quality_mean",       "reasoning_quality_sem",       "Reasoning\nQuality"),
        ("completeness_mean",            "completeness_sem",            "Completeness"),
        ("interpretability_mean",        "interpretability_sem",        "Interpretability"),
    ]
    systems = ["ARIA", "Online KG+LLM", "Baseline LLM", "Naive KG+LLM"]
    tasks   = ["forward", "inverse"]
    x = np.arange(len(metric_cols))
    n = len(systems)
    width_total = 0.7
    w = width_total / n
    hatches = ["", "///"]

    for t_idx, task in enumerate(tasks):
        subset = aria_g[aria_g["task"] == task]
        for s_idx, sys in enumerate(systems):
            row = subset[subset["system"] == sys]
            if row.empty:
                continue
            row = row.iloc[0]
            means = [row[m] for m, _, _ in metric_cols]
            sems  = [row[s] for _, s, _ in metric_cols]
            offset = (s_idx - n/2 + 0.5) * w + (t_idx - 0.5) * (w * 0.15)
            ax.bar(x + offset, means, w * 0.9,
                   color=C[sys], alpha=TASK_ALPHA[task],
                   hatch=hatches[t_idx], edgecolor="white", linewidth=0.4,
                   yerr=sems, capsize=2, error_kw=dict(elinewidth=0.6))

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, _, lbl in metric_cols], fontsize=8)
    ax.set_ylabel("Score (0–1)", fontsize=9)
    ax.set_title("(C)  ARIA Variants — Per-Metric Breakdown (Gemini judge)",
                 fontsize=10, fontweight="bold", loc="left")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(facecolor=C[s], label=s) for s in systems]
    patch_fwd = mpatches.Patch(facecolor="grey", alpha=1.0,        label="Forward")
    patch_inv = mpatches.Patch(facecolor="grey", alpha=0.55, hatch="///",
                               edgecolor="grey",                    label="Inverse")
    ax.legend(handles=handles + [patch_fwd, patch_inv],
              fontsize=7.5, loc="upper right", ncol=2, framealpha=0.8)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    aria_g    = load_aria()
    selfrag_g = load_selfrag()

    fig = plt.figure(figsize=(14, 11))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                   height_ratios=[1, 1])

    ax_a = fig.add_subplot(gs[0, :])   # top row, full width
    ax_b = fig.add_subplot(gs[1, 0])   # bottom left
    ax_c = fig.add_subplot(gs[1, 1])   # bottom right

    plot_overall(ax_a, aria_g, selfrag_g)
    plot_selfrag_metrics(ax_b, selfrag_g)
    plot_aria_metrics(ax_c, aria_g)

    fig.suptitle(
        "SelfRAG vs ARIA Variants — Rebuttal Comparison\n"
        r"$\bf{Note:}$"
        " ARIA evaluated with Gemini judge; SelfRAG with Qwen2-7B judge."
        " Overall scores normalised to 0–1 for panel (A).",
        fontsize=10, y=1.01
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out = OUT_DIR / f"rebuttal_comparison.{fmt}"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
