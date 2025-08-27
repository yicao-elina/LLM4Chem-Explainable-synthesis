import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jhu_colors
from jhu_colors import get_jhu_color
from pathlib import Path

def plot_comprehensive_results_llm(results_df: pd.DataFrame):
    """
    Creates a robust, comprehensive, and publication-quality visualization of evaluation 
    results from a DataFrame containing multi-dimensional LLM-based scores.

    Args:
        results_df: A pandas DataFrame with the detailed LLM evaluation results.
    """
    # Set the style and color palette for all plots
    task_colors = {
        "forward_prediction": get_jhu_color('Heritage Blue'),
        "inverse_design": get_jhu_color('Spirit Blue')
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # --- Plot 1: Radar Chart for Multi-Dimensional Score Comparison ---
    ax = axes[0, 0]
    ax.remove()
    ax = fig.add_subplot(2, 3, 1, polar=True)
    
    score_dims = ['Scientific Accuracy', 'Functional Equivalence', 'Reasoning Quality', 'Completeness']
    score_cols = [f"{dim.lower().replace(' ', '_')}_score" for dim in score_dims]
    labels = np.array(score_dims)
    
    # Check if score columns exist before calculating means to prevent KeyErrors
    dag_score_cols_exist = [f"dag_{col}" for col in score_cols if f"dag_{col}" in results_df.columns]
    baseline_score_cols_exist = [f"baseline_{col}" for col in score_cols if f"baseline_{col}" in results_df.columns]
    
    if not dag_score_cols_exist or not baseline_score_cols_exist:
        ax.text(0.5, 0.5, 'Score data missing', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    else:
        dag_means = results_df[dag_score_cols_exist].mean().values
        baseline_means = results_df[baseline_score_cols_exist].mean().values
        
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        stats_dag = np.concatenate((dag_means, [dag_means[0]]))
        stats_baseline = np.concatenate((baseline_means, [baseline_means[0]]))
        angles_closed = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles_closed, stats_dag, 'o-', linewidth=2, label='DAG-LLM', color=get_jhu_color('Heritage Blue'))
        ax.fill(angles_closed, stats_dag, alpha=0.25, color=get_jhu_color('Heritage Blue'))
        ax.plot(angles_closed, stats_baseline, 'o-', linewidth=2, label='Baseline', color=get_jhu_color('Red'))
        ax.fill(angles_closed, stats_baseline, alpha=0.25, color=get_jhu_color('Red'))
        
        ax.set_thetagrids(angles * 180 / np.pi, labels)
        ax.set_title("Multi-Dimensional Performance Profile")
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # --- Plot 2: Perturbation Level vs Overall Score ---
    ax = axes[0, 1]
    sns.lineplot(data=results_df, x="perturbation_level", y="dag_overall_score_score", 
                 hue="task", style="task", markers=True, dashes=False, ax=ax, palette=task_colors)
    sns.lineplot(data=results_df, x="perturbation_level", y="baseline_overall_score_score", 
                 hue="task", style="task", markers=True, dashes=True, ax=ax, 
                 palette={"forward_prediction": get_jhu_color('Orange'), "inverse_design": get_jhu_color('Red')})
    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("Average Overall Score")
    ax.set_title("Model Robustness to Perturbations")
    
    # --- ROBUST LEGEND CREATION ---
    # This dynamically creates the legend, avoiding the index error.
    handles, labels = ax.get_legend_handles_labels()
    new_handles, new_labels = [], []
    label_map = {
        'forward_prediction': ('DAG-LLM (Fwd)', 'Baseline (Fwd)'),
        'inverse_design': ('DAG-LLM (Inv)', 'Baseline (Inv)')
    }
    # This logic correctly maps the generated handles to the desired labels
    # It assumes the first two unique labels are the tasks.
    unique_tasks = pd.unique(labels)
    if len(unique_tasks) > 2: unique_tasks = unique_tasks[2:]

    for handle, label in zip(handles, labels):
        if label in label_map:
            # First time seeing this task, add DAG-LLM label
            new_labels.append(label_map[label][0])
            new_handles.append(handle)
            # Add Baseline label
            baseline_handle_index = labels.index(label, labels.index(label) + 1)
            new_labels.append(label_map[label][1])
            new_handles.append(handles[baseline_handle_index])
            # Prevent re-adding
            label_map.pop(label)

    ax.legend(handles=new_handles, labels=new_labels, title="Model & Task")


    # --- Plot 3: Improvement over Baseline (based on Overall Score) ---
    ax = axes[0, 2]
    sns.barplot(data=results_df, x="perturbation_level", y="dag_improvement", hue="task", ax=ax, palette=task_colors)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("Improvement in Overall Score")
    ax.set_title("Relative Performance Gain")

    # --- Plot 4: Confidence Distribution by Perturbation Level ---
    ax = axes[1, 0]
    sns.boxplot(data=results_df, x="perturbation_level", y="dag_confidence", hue="task", ax=ax, palette=task_colors)
    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("DAG-LLM Confidence")
    ax.set_title("Confidence Distribution")

    # --- Plot 5: Performance Heatmap (DAG-LLM Overall Score) ---
    ax = axes[1, 1]
    pivot_data = results_df.pivot_table(values="dag_overall_score_score", index="perturbation_level", columns="task", aggfunc="mean")
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="jhu", ax=ax, linewidths=.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title("DAG-LLM Performance Heatmap")
    
    # --- Plot 6: Head-to-Head Scatter Plot (Overall Score) ---
    ax = axes[1, 2]
    scatter = ax.scatter(results_df["baseline_overall_score_score"], results_df["dag_overall_score_score"],
                         alpha=0.6, c=results_df["perturbation_level"], cmap="jhu", s=50)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("Baseline Overall Score")
    ax.set_ylabel("DAG-LLM Overall Score")
    ax.set_title("Head-to-Head Model Comparison")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Perturbation Level")
    
    plt.tight_layout()
    output_filename = "comprehensive_llm_evaluation_figure.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nComprehensive evaluation figure saved as '{output_filename}'")

def create_additional_plots_llm(results_df: pd.DataFrame):
    """Create additional insightful plots based on LLM evaluation scores."""
    # jhu_colors.use_style()
    task_colors = {
        "forward_prediction": get_jhu_color('Heritage Blue'),
        "inverse_design": get_jhu_color('Spirit Blue')
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Violin plot of improvements by task
    ax = axes[0]
    sns.violinplot(data=results_df, x="task", y="dag_improvement", ax=ax, palette=task_colors)
    ax.axhline(y=0, color=get_jhu_color('Red'), linestyle='--', alpha=0.7)
    ax.set_xlabel("Task")
    ax.set_ylabel("Improvement in Overall Score")
    ax.set_title("Distribution of Performance Gains")
    ax.set_xticklabels(["Forward\nPrediction", "Inverse\nDesign"])
    
    # Plot 2: Confidence vs Improvement
    ax = axes[1]
    sns.scatterplot(data=results_df, x="dag_confidence", y="dag_improvement", hue="task", palette=task_colors, ax=ax, alpha=0.7, s=60)
    ax.axhline(y=0, color=get_jhu_color('Red'), linestyle='--', alpha=0.7)
    ax.set_xlabel("DAG-LLM Confidence")
    ax.set_ylabel("Improvement over Baseline")
    ax.set_title("Confidence vs. Performance Gain")
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Bar plot of average scores by host material
    ax = axes[2]
    if "host_material" in results_df.columns and not results_df["host_material"].isnull().all():
        top_materials = results_df["host_material"].value_counts().nlargest(5).index
        material_data = results_df[results_df["host_material"].isin(top_materials)]
        
        if not material_data.empty:
            material_summary = material_data.groupby("host_material").agg({
                "dag_overall_score_score": "mean",
                "baseline_overall_score_score": "mean"
            }).reindex(top_materials)
            
            x = np.arange(len(material_summary))
            width = 0.35
            
            ax.bar(x - width/2, material_summary["dag_overall_score_score"], width, label='DAG-LLM', color=get_jhu_color('Heritage Blue'))
            ax.bar(x + width/2, material_summary["baseline_overall_score_score"], width, label='Baseline', color=get_jhu_color('Spirit Blue'))
            
            ax.set_xlabel("Host Material")
            ax.set_ylabel("Average Overall Score")
            ax.set_title("Performance by Host Material (Top 5)")
            ax.set_xticks(x)
            ax.set_xticklabels(material_summary.index, rotation=45, ha='right')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig("additional_llm_evaluation_plots.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Additional analysis plots saved as 'additional_llm_evaluation_plots.png'")


if __name__ == '__main__':
    results_file = Path("llm_evaluation_detailed_results.csv")
    
    if results_file.exists():
        print(f"Loading evaluation data from '{results_file}'...")
        try:
            df = pd.read_csv(results_file)
            
            if df.empty:
                print("Warning: The CSV file is empty. No plots will be generated.")
            else:
                print("\nGenerating comprehensive visualizations...")
                plot_comprehensive_results_llm(df)
                
                print("\nGenerating additional analysis plots...")
                create_additional_plots_llm(df)
            
        except Exception as e:
            print(f"An error occurred while loading or plotting the data: {e}")
    else:
        print(f"Error: The results file was not found at '{results_file}'.")
        print("Please ensure the evaluation script has run successfully to generate this file.")
