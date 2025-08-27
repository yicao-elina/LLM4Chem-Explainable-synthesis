import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jhu_colors
from jhu_colors import get_jhu_color

def plot_comprehensive_results(results_df: pd.DataFrame):
    """Creates a comprehensive visualization of evaluation results using the JHU color scheme."""
    # Set the style and color palette for the plots
    # jhu_colors.use_style()
    task_colors = {
        "forward_prediction": get_jhu_color('Heritage Blue'),
        "inverse_design": get_jhu_color('Spirit Blue')
    }
    task_markers = {
        "forward_prediction": 'o',
        "inverse_design": '^'
    }

    # Maintain a (6,4) aspect ratio for each subplot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Confidence vs Similarity for DAG-LLM
    ax = axes[0, 0]
    for task in ["forward_prediction", "inverse_design"]:
        task_data = results_df[results_df["task"] == task]
        ax.scatter(task_data["dag_confidence"], task_data["dag_similarity"],
                   alpha=0.7, label=task.replace("_", " ").title(), s=60,
                   color=task_colors[task], marker=task_markers[task], edgecolors='w', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Ideal Correlation (y=x)')
    ax.set_xlabel("DAG-LLM Confidence Score")
    ax.set_ylabel("Semantic Similarity to Ground Truth")
    ax.set_title("DAG-LLM: Confidence vs. Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Perturbation Level vs Similarity (both models)
    ax = axes[0, 1]
    for task in ["forward_prediction", "inverse_design"]:
        task_data = results_df[results_df["task"] == task]
        grouped = task_data.groupby("perturbation_level").agg({
            "dag_similarity": "mean",
            "baseline_similarity": "mean"
        })
        ax.plot(grouped.index, grouped["dag_similarity"], marker=task_markers[task], linestyle='-',
                label=f"DAG-LLM ({task.split('_')[0].title()})", markersize=8, color=task_colors[task])
        ax.plot(grouped.index, grouped["baseline_similarity"], marker=task_markers[task], linestyle='--',
                label=f"Baseline ({task.split('_')[0].title()})", markersize=8, color=get_jhu_color('Red'))
    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("Average Semantic Similarity")
    ax.set_title("Model Robustness to Input Perturbations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Improvement over baseline
    ax = axes[0, 2]
    improvement_data = results_df.groupby(["task", "perturbation_level"])["dag_improvement"].mean().unstack(level=0)
    width = 0.04
    x = improvement_data.index
    ax.bar(x - width/2, improvement_data["forward_prediction"], width=width,
           label="Forward Prediction", color=task_colors["forward_prediction"])
    ax.bar(x + width/2, improvement_data["inverse_design"], width=width,
           label="Inverse Design", color=task_colors["inverse_design"])
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("Improvement over Baseline")
    ax.set_title("Relative Performance Gain")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Confidence distribution by perturbation level
    ax = axes[1, 0]
    sns.boxplot(data=results_df, x="perturbation_level", y="dag_confidence",
                hue="task", ax=ax, palette=task_colors)

    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("DAG-LLM Confidence")
    ax.set_title("Confidence Distribution by Perturbation")
    
    # 5. Performance heatmap
    ax = axes[1, 1]
    pivot_data = results_df.pivot_table(
        values="dag_similarity",
        index="perturbation_level",
        columns="task",
        aggfunc="mean"
    )
    sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="jhu", ax=ax, linewidths=.5)
    # maually change the x tick and tilt it
    ax.set_xticklabels(["Forward", "Inverse"], rotation=45, ha='right')
    ax.set_title("DAG-LLM Performance Heatmap")
    
    # 6. Scatter: DAG vs Baseline similarity
    ax = axes[1, 2]
    scatter = ax.scatter(results_df["baseline_similarity"], results_df["dag_similarity"],
                         alpha=0.6, c=results_df["perturbation_level"], cmap="jhu", s=50)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("Baseline Similarity")
    ax.set_ylabel("DAG-LLM Similarity")
    ax.set_title("Head-to-Head Model Comparison")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Perturbation Level")
    
    plt.tight_layout()
    plt.savefig("comprehensive_evaluation_results.png", dpi=300, bbox_inches='tight')
    plt.show()

def print_key_findings(results_df: pd.DataFrame):
    """Print key findings from the evaluation results"""
    print("\n" + "="*60)
    print("KEY FINDINGS FROM STATISTICAL ANALYSIS")
    print("="*60)
    
    # Average performance by model
    avg_dag = results_df["dag_similarity"].mean()
    avg_baseline = results_df["baseline_similarity"].mean()
    print(f"\nAverage Semantic Similarity (Overall):")
    print(f"  DAG-LLM: {avg_dag:.3f}")
    print(f"  Baseline: {avg_baseline:.3f}")
    print(f"  Improvement: {avg_dag - avg_baseline:.3f} ({(avg_dag - avg_baseline)/avg_baseline*100:.1f}%)")
    
    # Performance by perturbation level
    perturbation_levels = sorted(results_df["perturbation_level"].unique())
    print("\nDAG-LLM Advantage by Perturbation Level:")
    for pert_level in perturbation_levels:
        level_data = results_df[results_df["perturbation_level"] == pert_level]
        if len(level_data) > 0:
            avg_improvement = level_data["dag_improvement"].mean()
            win_rate = (level_data["dag_improvement"] > 0).sum() / len(level_data)
            print(f"  Level {pert_level:.2f}: +{avg_improvement:.3f} avg. similarity ({win_rate*100:.0f}% win rate)")
    
    # Confidence calibration
    print("\nConfidence Calibration (Pearson correlation of Confidence vs. Similarity):")
    for task in ["forward_prediction", "inverse_design"]:
        task_data = results_df[results_df["task"] == task]
        if len(task_data) > 1: # Correlation requires more than one data point
            correlation = task_data["dag_confidence"].corr(task_data["dag_similarity"])
            print(f"  {task.replace('_', ' ').title()}: {correlation:.3f}")
    
    # Task-specific performance
    print("\nPerformance by Task:")
    task_summary = results_df.groupby("task").agg({
        "dag_similarity": ["mean", "std"],
        "baseline_similarity": ["mean", "std"],
        "dag_improvement": "mean"
    }).round(3)
    print(task_summary)

def create_additional_plots(results_df: pd.DataFrame):
    """Create additional insightful plots using the JHU color scheme."""
    # jhu_colors.use_style()
    task_colors = {
        "forward_prediction": get_jhu_color('Heritage Blue'),
        "inverse_design": get_jhu_color('Spirit Blue')
    }
    
    # Maintain a (6,4) aspect ratio for each subplot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Violin plot of improvements by task
    ax = axes[0]
    sns.violinplot(data=results_df, x="task", y="dag_improvement", ax=ax, palette=task_colors)
    ax.axhline(y=0, color=get_jhu_color('Red'), linestyle='--', alpha=0.7)
    ax.set_xlabel("Task")
    ax.set_ylabel("Improvement over Baseline")
    ax.set_title("Distribution of Performance Gains")
    ax.set_xticklabels(["Forward\nPrediction", "Inverse\nDesign"])
    
    # 2. Confidence vs Improvement
    ax = axes[1]
    for task in ["forward_prediction", "inverse_design"]:
        task_data = results_df[results_df["task"] == task]
        ax.scatter(task_data["dag_confidence"], task_data["dag_improvement"],
                   alpha=0.6, label=task.replace("_", " ").title(), s=50, color=task_colors[task])
    ax.axhline(y=0, color=get_jhu_color('Red'), linestyle='--', alpha=0.7)
    ax.set_xlabel("DAG-LLM Confidence")
    ax.set_ylabel("Improvement over Baseline")
    ax.set_title("Confidence vs. Performance Gain")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Bar plot of average similarities by host material (if available)
    ax = axes[2]
    if "host_material" in results_df.columns:
        top_materials = results_df["host_material"].value_counts().nlargest(5).index
        material_data = results_df[results_df["host_material"].isin(top_materials)]
        
        material_summary = material_data.groupby("host_material").agg({
            "dag_similarity": "mean",
            "baseline_similarity": "mean"
        }).reindex(top_materials) # Keep original order
        
        x = np.arange(len(material_summary))
        width = 0.35
        
        ax.bar(x - width/2, material_summary["dag_similarity"], width, label='DAG-LLM', color=get_jhu_color('Heritage Blue'))
        ax.bar(x + width/2, material_summary["baseline_similarity"], width, label='Baseline', color=get_jhu_color('Spirit Blue'))
        
        ax.set_xlabel("Host Material")
        ax.set_ylabel("Average Similarity")
        ax.set_title("Performance by Host Material (Top 5)")
        ax.set_xticks(x)
        ax.set_xticklabels(material_summary.index, rotation=45, ha='right')
        ax.legend()
    else:
        # Alternative plot if host_material not available
        summary_by_pert = results_df.groupby("perturbation_level")["dag_improvement"].agg(["mean", "std"])
        
        ax.errorbar(summary_by_pert.index, summary_by_pert["mean"], yerr=summary_by_pert["std"],
                    fmt='o-', capsize=5, capthick=2, color=get_jhu_color('Heritage Blue'))
        ax.axhline(y=0, color=get_jhu_color('Red'), linestyle='--', alpha=0.7)
        ax.set_xlabel("Perturbation Level")
        ax.set_ylabel("Mean Improvement ± Std")
        ax.set_title("Average Performance Gain")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("additional_evaluation_plots.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load the saved evaluation results
    try:
        results_df = pd.read_csv("detailed_evaluation_results.csv")
        print(f"Loaded {len(results_df)} evaluation results from 'detailed_evaluation_results.csv'")
        
        # Display basic statistics
        print("\nDataset Overview:")
        print(f"  Total records: {len(results_df)}")
        print(f"  Tasks: {list(results_df['task'].unique())}")
        print(f"  Perturbation levels: {sorted(results_df['perturbation_level'].unique())}")
        print(f"  Columns: {list(results_df.columns)}")
        
        # Generate the comprehensive plots
        print("\nGenerating comprehensive evaluation plots...")
        plot_comprehensive_results(results_df)
        
        # Generate additional plots
        print("\nGenerating additional analysis plots...")
        create_additional_plots(results_df)
        
        # Print key findings
        print_key_findings(results_df)
        
    except FileNotFoundError:
        print("\nError: Could not find 'detailed_evaluation_results.csv'")
        print("Please ensure the full evaluation script has been run first to generate this file.")
    except Exception as e:
        print(f"\nAn error occurred during plotting or analysis: {e}")

