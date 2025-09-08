import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import jhu_colors
from jhu_colors import get_jhu_color
import argparse
import sys

# Function to wrap long labels
def wrap_labels(labels, max_length=10):
    wrapped = []
    for label in labels:
        if len(label) > max_length:
            words = label.split()
            if len(words) > 1:
                mid = len(words) // 2
                wrapped.append('\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])]))
            else:
                # If single long word, break at character level
                wrapped.append(label[:max_length] + '\n' + label[max_length:])
        else:
            wrapped.append(label)
    return wrapped

def detect_engines_from_dataframe(results_df: pd.DataFrame) -> list:
    """Automatically detect engine names from dataframe columns"""
    engine_names = []
    
    # Look for columns that end with '_overall_score_score' but aren't 'baseline_overall_score_score'
    for col in results_df.columns:
        if col.endswith('_overall_score_score') and not col.startswith('baseline_'):
            engine_name = col.replace('_overall_score_score', '')
            if engine_name not in engine_names:
                engine_names.append(engine_name)
    
    print(f"Detected engines: {engine_names}")
    return engine_names

def plot_multi_engine_results(results_df: pd.DataFrame, engine_names: list = None, save_path: str = "multi_engine_evaluation_results.png"):
    """Creates comprehensive visualization for multi-engine evaluation."""
    
    # Auto-detect engines if not provided
    if engine_names is None:
        engine_names = detect_engines_from_dataframe(results_df)
    
    if not engine_names:
        print("No engines detected in the dataframe!")
        return
    
    # Define colors for each model
    colors = [get_jhu_color('Heritage Blue'), get_jhu_color('Spirit Blue'), 
              get_jhu_color('Sizzling Red'), get_jhu_color('Academic Blue'),
              get_jhu_color('Homewood Gold'), get_jhu_color('Creekside Green')]
    baseline_color = get_jhu_color('Red')
    
    model_colors = {name: colors[i % len(colors)] for i, name in enumerate(engine_names)}
    model_colors['baseline'] = baseline_color
    
    task_colors = {"forward_prediction": get_jhu_color('Heritage Blue'), 
                   "inverse_design": get_jhu_color('Spirit Blue')}
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Radar Chart for Multi-Dimensional Score Comparison
    ax = axes[0, 0]
    ax.remove()
    ax = fig.add_subplot(2, 3, 1, polar=True)
    
    score_dims = ['Scientific Accuracy', 'Functional Equivalence', 'Reasoning Quality', 
                  'Completeness', 'Overall Score']
    score_cols = [f"{dim.lower().replace(' ', '_')}_score" for dim in score_dims]
    labels = np.array(score_dims)
    wrapped_labels = wrap_labels(labels, max_length=6)

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    # Plot baseline
    baseline_means = results_df[[f"baseline_{col}" for col in score_cols]].mean().values
    stats_baseline = np.concatenate((baseline_means, [baseline_means[0]]))
    ax.plot(angles, stats_baseline, 'o-', linewidth=2, label='Baseline', 
            color=baseline_color, linestyle='--')
    ax.fill(angles, stats_baseline, alpha=0.15, color=baseline_color)

    # Plot each engine
    for engine_name in engine_names:
        try:
            engine_means = results_df[[f"{engine_name}_{col}" for col in score_cols]].mean().values
            stats_engine = np.concatenate((engine_means, [engine_means[0]]))
            ax.plot(angles, stats_engine, 'o-', linewidth=2, label=engine_name, 
                    color=model_colors[engine_name])
            ax.fill(angles, stats_engine, alpha=0.25, color=model_colors[engine_name])
        except KeyError as e:
            print(f"Warning: Missing columns for engine {engine_name}: {e}")
            continue

    # ax.set_thetagrids(angles[:-1] * 180/np.pi, wrapped_labels)
    # Remove default theta labels
    ax.set_thetagrids(angles[:-1] * 180/np.pi, [])

    # Manually place labels with custom distance
    label_distance = 1.35  # Adjust this value to control distance from circle
    for angle, label in zip(angles[:-1], wrapped_labels):
        ax.text(angle, label_distance, label,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transData)
                
    # ax.set_title("Multi-Engine Performance Profile", pad=20)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.6, 1.0])
    ax.set_yticklabels(['0.2', '0.6', '1.0'])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.2))

    # 2. Model comparison by overall score
    ax = axes[0, 1]
    all_models = ['baseline'] + engine_names
    model_scores = []
    model_names = []
    
    for model in all_models:
        try:
            score = results_df[f"{model}_overall_score_score"].mean()
            model_scores.append(score)
            model_names.append(model.replace('_', ' ').title())
        except KeyError:
            print(f"Warning: Missing overall score for {model}")
            continue
    
    if model_scores:
        bars = ax.bar(range(len(model_names)), model_scores, 
                      color=[model_colors.get(model, baseline_color) for model in all_models[:len(model_scores)]], 
                      alpha=0.8)
        
        # Add value labels on bars
        for bar, score in zip(bars, model_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        ax.set_xlabel("Models")
        ax.set_ylabel("Average Overall Score")
        ax.set_title("Overall Model Performance Comparison")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(model_scores) * 1.2)

    # 3. Improvement over baseline distribution
    ax = axes[0, 2]
    for engine_name in engine_names:
        try:
            improvements = results_df[f"{engine_name}_improvement"]
            ax.hist(improvements, bins=15, alpha=0.6, label=engine_name, 
                    color=model_colors[engine_name])
        except KeyError:
            print(f"Warning: Missing improvement data for {engine_name}")
            continue
    
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel("Improvement over Baseline")
    ax.set_ylabel("Count")
    ax.set_title("Performance Improvement Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Box plot comparison across all models
    ax = axes[1, 0]
    score_data = []
    for model in all_models:
        try:
            for _, row in results_df.iterrows():
                score_data.append({
                    'Model': model.replace('_', ' ').title(),
                    'Score': row[f"{model}_overall_score_score"]
                })
        except KeyError:
            continue
    
    if score_data:
        score_df = pd.DataFrame(score_data)
        palette = {model.replace('_', ' ').title(): model_colors.get(model, baseline_color) 
                  for model in all_models}
        sns.boxplot(data=score_df, x="Model", y="Score", ax=ax, palette=palette)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title("Score Distribution by Model")
        ax.grid(True, alpha=0.3, axis='y')

    # 5. Performance by task type
    ax = axes[1, 1]
    task_performance = {}
    for task in ["forward_prediction", "inverse_design"]:
        task_data = results_df[results_df["task"] == task]
        task_performance[task] = {}
        for model in all_models:
            try:
                task_performance[task][model] = task_data[f"{model}_overall_score_score"].mean()
            except KeyError:
                task_performance[task][model] = 0  # Default to 0 if missing
    
    x = np.arange(len(all_models))
    width = 0.35
    
    forward_scores = [task_performance["forward_prediction"][model] for model in all_models]
    inverse_scores = [task_performance["inverse_design"][model] for model in all_models]
    
    bars1 = ax.bar(x - width/2, forward_scores, width, label='Forward Prediction', 
                   color=task_colors["forward_prediction"], alpha=0.8)
    bars2 = ax.bar(x + width/2, inverse_scores, width, label='Inverse Design', 
                   color=task_colors["inverse_design"], alpha=0.8)
    
    ax.set_xlabel("Models")
    ax.set_ylabel("Average Score")
    ax.set_title("Performance by Task Type")
    ax.set_xticks(x)
    ax.set_xticklabels([model.replace('_', ' ').title() for model in all_models], 
                       rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Head-to-head comparison with best engine
    ax = axes[1, 2]
    if engine_names:
        # Find best engine
        best_engine = None
        best_score = -1
        for engine_name in engine_names:
            try:
                score = results_df[f"{engine_name}_overall_score_score"].mean()
                if score > best_score:
                    best_score = score
                    best_engine = engine_name
            except KeyError:
                continue
        
        if best_engine:
            for task in ["forward_prediction", "inverse_design"]:
                task_data = results_df[results_df["task"] == task]
                try:
                    ax.scatter(task_data["baseline_overall_score_score"], 
                              task_data[f"{best_engine}_overall_score_score"],
                              alpha=0.6, label=task.replace("_", " ").title(), 
                              s=50, color=task_colors[task])
                except KeyError:
                    continue
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel("Baseline Overall Score")
            ax.set_ylabel(f"{best_engine} Overall Score")
            ax.set_title(f"Head-to-Head: {best_engine} vs Baseline")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.show()

def generate_multi_engine_summary_statistics(results_df: pd.DataFrame, engine_names: list = None) -> tuple:
    """Generates comprehensive summary statistics for multiple engines"""
    
    # Auto-detect engines if not provided
    if engine_names is None:
        engine_names = detect_engines_from_dataframe(results_df)
    
    overall_stats = {}
    
    score_dims = ['scientific_accuracy_score', 'functional_equivalence_score', 
                  'reasoning_quality_score', 'completeness_score', 'overall_score_score']
    
    all_models = ['baseline'] + engine_names
    
    for model in all_models:
        for dim in score_dims:
            try:
                overall_stats[f'{model}_{dim}_mean'] = results_df[f'{model}_{dim}'].mean()
                overall_stats[f'{model}_{dim}_std'] = results_df[f'{model}_{dim}'].std()
            except KeyError:
                overall_stats[f'{model}_{dim}_mean'] = 0.0
                overall_stats[f'{model}_{dim}_std'] = 0.0
    
    # Calculate improvements over baseline for each engine
    for engine_name in engine_names:
        for dim in score_dims:
            baseline_mean = overall_stats[f'baseline_{dim}_mean']
            engine_mean = overall_stats[f'{engine_name}_{dim}_mean']
            overall_stats[f'{engine_name}_{dim}_improvement'] = engine_mean - baseline_mean
        
        # Win rates
        try:
            overall_stats[f'{engine_name}_win_rate'] = (results_df[f'{engine_name}_improvement'] > 0).mean()
        except KeyError:
            overall_stats[f'{engine_name}_win_rate'] = 0.0
    
    # Task statistics
    task_stats = {}
    for task in ['forward_prediction', 'inverse_design']:
        task_data = results_df[results_df['task'] == task]
        task_stats[task] = {}
        for model in all_models:
            try:
                task_stats[task][f'{model}_mean'] = task_data[f'{model}_overall_score_score'].mean()
                task_stats[task][f'{model}_std'] = task_data[f'{model}_overall_score_score'].std()
            except KeyError:
                task_stats[task][f'{model}_mean'] = 0.0
                task_stats[task][f'{model}_std'] = 0.0
    
    return overall_stats, task_stats, engine_names

def print_multi_engine_summary_statistics(results_df: pd.DataFrame, engine_names: list = None):
    """Print key findings and statistics for multiple engines"""
    overall_stats, task_stats, detected_engines = generate_multi_engine_summary_statistics(results_df, engine_names)
    
    if engine_names is None:
        engine_names = detected_engines
    
    print("\n" + "="*80)
    print("MULTI-ENGINE EVALUATION RESULTS:")
    print("="*80)
    
    print("\nOverall Performance Ranking (by overall score):")
    all_models = ['baseline'] + engine_names
    model_scores = [(model, overall_stats[f'{model}_overall_score_score_mean']) 
                   for model in all_models]
    model_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, score) in enumerate(model_scores, 1):
        print(f"  {i}. {model}: {score:.3f}")
    
    print("\nImprovement over Baseline:")
    for engine_name in engine_names:
        improvement = overall_stats[f'{engine_name}_overall_score_score_improvement']
        win_rate = overall_stats[f'{engine_name}_win_rate'] * 100
        print(f"  {engine_name}: +{improvement:.3f} (Win Rate: {win_rate:.1f}%)")
    
    print("\nPerformance by Task:")
    for task in ['forward_prediction', 'inverse_design']:
        print(f"\n  {task.replace('_', ' ').title()}:")
        task_scores = [(model, task_stats[task][f'{model}_mean']) for model in all_models]
        task_scores.sort(key=lambda x: x[1], reverse=True)
        for model, score in task_scores:
            print(f"    {model}: {score:.3f}")
    
    # Confidence calibration for engines that have confidence data
    print("\nConfidence Calibration (Pearson correlation):")
    for engine_name in engine_names:
        for task in ["forward_prediction", "inverse_design"]:
            task_data = results_df[results_df["task"] == task]
            try:
                if len(task_data) > 0 and f"{engine_name}_confidence" in task_data.columns:
                    correlation = task_data[f"{engine_name}_confidence"].corr(task_data[f"{engine_name}_overall_score_score"])
                    print(f"  {engine_name} - {task.replace('_', ' ').title()}: {correlation:.3f}")
            except KeyError:
                continue

def load_and_plot_multi_engine_results(csv_file: str, save_path: str = None, engine_names: list = None):
    """Load results from CSV and create multi-engine plots"""
    try:
        results_df = pd.read_csv(csv_file)
        print(f"Loaded {len(results_df)} evaluation results from {csv_file}")
        
        # Auto-detect engines if not provided
        if engine_names is None:
            engine_names = detect_engines_from_dataframe(results_df)
        
        if not engine_names:
            print("No engines detected in the data!")
            return
        
        # Set default save path if not provided
        if save_path is None:
            save_path = csv_file.replace('.csv', '.png')
        
        # Validate that we have the required baseline columns
        required_baseline_columns = [
            'task', 'baseline_overall_score_score', 'baseline_scientific_accuracy_score',
            'baseline_functional_equivalence_score', 'baseline_reasoning_quality_score',
            'baseline_completeness_score'
        ]
        
        missing_columns = [col for col in required_baseline_columns if col not in results_df.columns]
        if missing_columns:
            print(f"Warning: Missing baseline columns: {missing_columns}")
        
        # Create plots
        print("Generating multi-engine visualizations...")
        plot_multi_engine_results(results_df, engine_names, save_path)
        
        # Print summary statistics
        print_multi_engine_summary_statistics(results_df, engine_names)
        
        print(f"\nVisualization saved to: {save_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find results file '{csv_file}'")
        print("Please run the main evaluation script first to generate the results.")
    except Exception as e:
        print(f"Error loading or plotting results: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Plot multi-engine evaluation results")
    parser.add_argument("--csv_file", required=True, help="Path to CSV results file")
    parser.add_argument("--save_path", help="Path to save the plot (default: derived from csv_file)")
    parser.add_argument("--engines", nargs='+', help="List of engine names (default: auto-detect)")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Command line interface
    if len(sys.argv) > 1:
        args = parse_arguments()
        load_and_plot_multi_engine_results(args.csv_file, args.save_path, args.engines)
    else:
        # Default behavior for your specific case
        csv_file = "comparison_study_evaluation_results.csv"
        save_path = "comparison_study_evaluation_results.png"
        
        # You can specify engine names explicitly or let it auto-detect
        # engine_names = ["DAG", "DAG+CoT"]  # Uncomment to specify explicitly
        engine_names = None  # Auto-detect from CSV
        
        load_and_plot_multi_engine_results(csv_file, save_path, engine_names)

# Example usage:
# python multi_engine_plot.py --csv_file "comparison_study_evaluation_results.csv" --save_path "comparison_study_plot.png"
# python multi_engine_plot.py --csv_file "comparison_study_evaluation_results.csv" --engines "DAG" "DAG+CoT"