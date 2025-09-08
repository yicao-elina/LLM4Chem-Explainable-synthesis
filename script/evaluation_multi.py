import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
import sys
from typing import Dict, List, Tuple, Any
import pandas as pd
import seaborn as sns
import re
import time
import jhu_colors
from jhu_colors import get_jhu_color
from scipy import stats
import importlib
import argparse

# --- UTILITY FUNCTIONS ---

def flatten_dict(d: dict, parent_key: str = '', sep: str = ' ') -> dict:
    """Flattens nested dictionary for comparison"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, ' '.join(str(item) for item in v)))
        elif v is not None and str(v).lower() not in ["", "not reported", "null", "none"]:
            items.append((new_key, str(v)))
    return dict(items)

def dict_to_string(d: dict) -> str:
    """Converts dictionary to string for embedding or display"""
    if not d or not isinstance(d, dict):
        return ""
    flattened = flatten_dict(d)
    return " and ".join([f"{key} is {value}" for key, value in sorted(flattened.items())])

def extract_json_from_response(text: str) -> Dict:
    """More robust JSON extractor from model responses."""
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if not match:
        return {"error": "No JSON block found in response."}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as e:
        return {"error": f"JSON decoding failed: {e}", "raw_text": match.group(1)}

def get_file_prefix(file_path: str) -> str:
    """Extract basename without extension to use as prefix for output files"""
    return Path(file_path).stem

def load_engine_from_module(module_name: str, training_graph_file: str):
    """Dynamically load CausalReasoningEngine from a module."""
    try:
        module = importlib.import_module(module_name)
        engine_class = getattr(module, 'CausalReasoningEngine')
        return engine_class(training_graph_file)
    except Exception as e:
        print(f"Error loading engine from {module_name}: {e}")
        return None

# --- MODEL DEFINITIONS ---

class BaselineGeminiModel:
    """Baseline Gemini model without DAG constraints for comparison"""
    def __init__(self, model_id: str = "gemini-1.5-pro-latest"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id)
    
    def forward_prediction(self, synthesis_inputs: dict) -> dict:
        prompt = f"""
        You are an expert materials scientist. Based on the following synthesis conditions, 
        predict the resulting material properties.
        
        Synthesis Conditions:
        {json.dumps(synthesis_inputs, indent=2)}
        
        Provide your answer in a JSON block with 'predicted_properties' and 'reasoning' fields.
        """
        response = self.model.generate_content(prompt)
        return extract_json_from_response(response.text)
    
    def inverse_design(self, desired_properties: dict) -> dict:
        prompt = f"""
        You are an expert materials scientist. Suggest synthesis conditions to achieve 
        the following material properties.
        
        Desired Properties:
        {json.dumps(desired_properties, indent=2)}
        
        Provide your answer in a JSON block with 'suggested_synthesis_conditions' and 'reasoning' fields.
        """
        response = self.model.generate_content(prompt)
        return extract_json_from_response(response.text)

# --- LLM-BASED EVALUATOR ---

class LLMEvaluator:
    """Uses a powerful LLM to perform multi-dimensional evaluation."""
    def __init__(self, model_id: str = "gemini-1.5-pro-latest"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY for evaluator.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_id,
            generation_config={"response_mime_type": "application/json"}
        )
        self.evaluation_prompt_template = """
        You are an expert materials scientist serving as an impartial judge. Your task is to evaluate a language model's generated output against a ground truth answer for a materials science problem.

        **Problem Context:**
        - **Task Type:** {task_type}
        - **Input Query:** {input_query}

        **Ground Truth Answer:**
        {ground_truth}

        **Model's Generated Answer:**
        {generated_answer}

        **Evaluation Criteria:**
        Please provide a score from 0 to 10 (integer) for each of the following dimensions. Be critical and rigorous.
        1.  **Scientific Accuracy (0-10):** Is the generated answer scientifically plausible and correct according to known principles of chemistry, physics, and materials science? (0=incorrect/unphysical, 10=perfectly accurate).
        2.  **Functional Equivalence (0-10):** Does the generated answer achieve the same functional outcome or describe the same core scientific concept as the ground truth, even if the wording is different? (0=completely different outcome, 10=functionally identical).
        3.  **Reasoning Quality (0-10):** If reasoning is provided, is it logical, clear, and scientifically sound? Does it correctly justify the conclusion? (0=no reasoning or illogical, 10=clear, correct, and insightful).
        4.  **Completeness (0-10):** Does the generated answer include all key parameters and details present in the ground truth? (0=missing most key details, 10=contains all necessary information).
        5.  **Overall Score (0-10):** Your holistic assessment of the generated answer's quality and usefulness.

        **Your Task:**
        Return a single JSON object with your scores and a brief justification for each score.
        
        **JSON Schema:**
        {{
            "scientific_accuracy": {{ "score": integer, "justification": "string" }},
            "functional_equivalence": {{ "score": integer, "justification": "string" }},
            "reasoning_quality": {{ "score": integer, "justification": "string" }},
            "completeness": {{ "score": integer, "justification": "string" }},
            "overall_score": {{ "score": integer, "justification": "string" }}
        }}
        """

    def evaluate_output(self, task_type: str, input_query: dict, ground_truth: dict, generated_output: dict) -> dict:
        """Performs the evaluation and returns a dictionary of scores."""
        prompt = self.evaluation_prompt_template.format(
            task_type=task_type,
            input_query=json.dumps(input_query, indent=2),
            ground_truth=json.dumps(ground_truth, indent=2),
            generated_answer=json.dumps(generated_output, indent=2)
        )
        
        for attempt in range(3): # Retry mechanism for API calls
            try:
                response = self.model.generate_content(prompt)
                evaluation = json.loads(response.text)
                # Normalize scores to be out of 1.0 for easier plotting
                scores = {f"{key}_score": val.get("score", 0) / 10.0 for key, val in evaluation.items()}
                scores['justifications'] = {key: val.get("justification", "") for key, val in evaluation.items()}
                return scores
            except Exception as e:
                print(f"  - Evaluator attempt {attempt+1} failed: {e}")
                time.sleep(2)
        
        # Return default scores on failure
        return {
            "scientific_accuracy_score": 0.0, "functional_equivalence_score": 0.0,
            "reasoning_quality_score": 0.0, "completeness_score": 0.0,
            "overall_score_score": 0.0, 'justifications': {}
        }

# --- DATA LOADING ---

def load_test_data(test_file: str) -> List[Dict]:
    """Load test dataset"""
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    # Handle both old format (doping_experiments) and new format (direct list)
    if isinstance(test_data, dict) and "doping_experiments" in test_data:
        experiments = test_data["doping_experiments"]
    elif isinstance(test_data, list):
        experiments = test_data
    else:
        print("Warning: Unexpected data format. Attempting to extract experiments...")
        experiments = []
    
    return experiments

# --- EVALUATION WORKFLOW ---

def evaluate_multiple_models_on_test_set(
    engines: Dict[str, Any],
    baseline_model: BaselineGeminiModel,
    llm_evaluator: LLMEvaluator,
    test_experiments: List[Dict]
) -> pd.DataFrame:
    """Evaluates multiple engines and baseline using the LLM-based evaluator on test set."""
    results = []
    
    for i, experiment in enumerate(test_experiments):
        print(f"\nEvaluating test case {i+1}/{len(test_experiments)}")
        
        # Extract synthesis conditions and property changes
        synthesis_conditions = experiment.get("synthesis_conditions", {})
        property_changes = experiment.get("property_changes", {})
        
        # --- Forward Prediction Evaluation ---
        print("  - Task: Forward Prediction")
        
        # Evaluate all engines
        engine_results = {}
        for engine_name, engine in engines.items():
            try:
                pred_result = engine.forward_prediction(synthesis_conditions)
                confidence = pred_result.get("confidence", 0.0)
                engine_results[engine_name] = {
                    'result': pred_result,
                    'confidence': confidence
                }
                print(f"    - Evaluating {engine_name} output...")
                scores = llm_evaluator.evaluate_output("Forward Prediction", synthesis_conditions, 
                                                      property_changes, pred_result)
                engine_results[engine_name]['scores'] = scores
            except Exception as e:
                print(f"    Warning: {engine_name} forward prediction failed: {e}")
                engine_results[engine_name] = {
                    'result': {"predicted_properties": {}, "reasoning": "Error"},
                    'confidence': 0.0,
                    'scores': {
                        "scientific_accuracy_score": 0.0, "functional_equivalence_score": 0.0,
                        "reasoning_quality_score": 0.0, "completeness_score": 0.0,
                        "overall_score_score": 0.0, 'justifications': {}
                    }
                }
        
        # Evaluate baseline
        try:
            baseline_pred_result = baseline_model.forward_prediction(synthesis_conditions)
            print("    - Evaluating Baseline output...")
            baseline_scores = llm_evaluator.evaluate_output("Forward Prediction", synthesis_conditions, 
                                                           property_changes, baseline_pred_result)
        except Exception as e:
            print(f"    Warning: Baseline forward prediction failed: {e}")
            baseline_pred_result = {"predicted_properties": {}, "reasoning": "Error"}
            baseline_scores = {
                "scientific_accuracy_score": 0.0, "functional_equivalence_score": 0.0,
                "reasoning_quality_score": 0.0, "completeness_score": 0.0,
                "overall_score_score": 0.0, 'justifications': {}
            }

        # Create result entry for forward prediction
        result_entry = {
            "task": "forward_prediction",
            "experiment_id": experiment.get("experiment_id", i),
            "host_material": experiment.get("host_material", "unknown"),
            "dopant": experiment.get("dopant", {}).get("element", "unknown"),
        }
        
        # Add baseline scores
        for k, v in baseline_scores.items():
            if k != 'justifications':
                result_entry[f"baseline_{k}"] = v
        
        # Add engine scores and improvements
        for engine_name, engine_data in engine_results.items():
            result_entry[f"{engine_name}_confidence"] = engine_data['confidence']
            for k, v in engine_data['scores'].items():
                if k != 'justifications':
                    result_entry[f"{engine_name}_{k}"] = v
            # Calculate improvement over baseline
            result_entry[f"{engine_name}_improvement"] = (
                engine_data['scores'].get('overall_score_score', 0.0) - 
                baseline_scores.get('overall_score_score', 0.0)
            )
        
        results.append(result_entry)

        # --- Inverse Design Evaluation ---
        print("  - Task: Inverse Design")
        
        # Evaluate all engines
        engine_results = {}
        for engine_name, engine in engines.items():
            try:
                inv_result = engine.inverse_design(property_changes)
                confidence = inv_result.get("confidence", 0.0)
                engine_results[engine_name] = {
                    'result': inv_result,
                    'confidence': confidence
                }
                print(f"    - Evaluating {engine_name} output...")
                scores = llm_evaluator.evaluate_output("Inverse Design", property_changes, 
                                                      synthesis_conditions, inv_result)
                engine_results[engine_name]['scores'] = scores
            except Exception as e:
                print(f"    Warning: {engine_name} inverse design failed: {e}")
                engine_results[engine_name] = {
                    'result': {"suggested_synthesis_conditions": {}, "reasoning": "Error"},
                    'confidence': 0.0,
                    'scores': {
                        "scientific_accuracy_score": 0.0, "functional_equivalence_score": 0.0,
                        "reasoning_quality_score": 0.0, "completeness_score": 0.0,
                        "overall_score_score": 0.0, 'justifications': {}
                    }
                }
        
        # Evaluate baseline
        try:
            baseline_inv_result = baseline_model.inverse_design(property_changes)
            print("    - Evaluating Baseline output...")
            baseline_scores = llm_evaluator.evaluate_output("Inverse Design", property_changes, 
                                                           synthesis_conditions, baseline_inv_result)
        except Exception as e:
            print(f"    Warning: Baseline inverse design failed: {e}")
            baseline_inv_result = {"suggested_synthesis_conditions": {}, "reasoning": "Error"}
            baseline_scores = {
                "scientific_accuracy_score": 0.0, "functional_equivalence_score": 0.0,
                "reasoning_quality_score": 0.0, "completeness_score": 0.0,
                "overall_score_score": 0.0, 'justifications': {}
            }

        # Create result entry for inverse design
        result_entry = {
            "task": "inverse_design",
            "experiment_id": experiment.get("experiment_id", i),
            "host_material": experiment.get("host_material", "unknown"),
            "dopant": experiment.get("dopant", {}).get("element", "unknown"),
        }
        
        # Add baseline scores
        for k, v in baseline_scores.items():
            if k != 'justifications':
                result_entry[f"baseline_{k}"] = v
        
        # Add engine scores and improvements
        for engine_name, engine_data in engine_results.items():
            result_entry[f"{engine_name}_confidence"] = engine_data['confidence']
            for k, v in engine_data['scores'].items():
                if k != 'justifications':
                    result_entry[f"{engine_name}_{k}"] = v
            # Calculate improvement over baseline
            result_entry[f"{engine_name}_improvement"] = (
                engine_data['scores'].get('overall_score_score', 0.0) - 
                baseline_scores.get('overall_score_score', 0.0)
            )
        
        results.append(result_entry)

    return pd.DataFrame(results)

# --- VISUALIZATION ---
def wrap_labels(labels, max_length=10):
    wrapped = []
    for label in labels:
        if len(label) > max_length:
            words = label.split()
            if len(words) > 1:
                mid = len(words) // 2
                wrapped.append('\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])]))
            else:
                wrapped.append(label[:max_length] + '\n' + label[max_length:])
        else:
            wrapped.append(label)
    return wrapped

def plot_multi_model_results(results_df: pd.DataFrame, engine_names: List[str], output_prefix: str):
    """Creates comprehensive visualization for multi-model evaluation."""
    
    # Define colors for each model
    colors = [get_jhu_color('Heritage Blue'), get_jhu_color('Spirit Blue'), 
              get_jhu_color('Sizzling Red'), get_jhu_color('Academic Blue'),
              get_jhu_color('Homewood Gold')]
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
        score = results_df[f"{model}_overall_score_score"].mean()
        model_scores.append(score)
        model_names.append(model.replace('_', ' ').title())
    
    bars = ax.bar(range(len(model_names)), model_scores, 
                  color=[model_colors[model] for model in all_models], alpha=0.8)
    
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
        improvements = results_df[f"{engine_name}_improvement"]
        ax.hist(improvements, bins=15, alpha=0.6, label=engine_name, 
                color=model_colors[engine_name])
    
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
        for _, row in results_df.iterrows():
            score_data.append({
                'Model': model.replace('_', ' ').title(),
                'Score': row[f"{model}_overall_score_score"]
            })
    
    score_df = pd.DataFrame(score_data)
    sns.boxplot(data=score_df, x="Model", y="Score", ax=ax,
                palette={model.replace('_', ' ').title(): model_colors[model] 
                        for model in all_models})
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
            task_performance[task][model] = task_data[f"{model}_overall_score_score"].mean()
    
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
    best_engine = max(engine_names, key=lambda x: results_df[f"{x}_overall_score_score"].mean())
    
    for task in ["forward_prediction", "inverse_design"]:
        task_data = results_df[results_df["task"] == task]
        ax.scatter(task_data["baseline_overall_score_score"], 
                  task_data[f"{best_engine}_overall_score_score"],
                  alpha=0.6, label=task.replace("_", " ").title(), 
                  s=50, color=task_colors[task])
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("Baseline Overall Score")
    ax.set_ylabel(f"{best_engine} Overall Score")
    ax.set_title(f"Head-to-Head: {best_engine} vs Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = f"{output_prefix}_multi_model_evaluation_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to '{output_file}'")
    plt.show()

def generate_multi_model_summary_statistics(results_df: pd.DataFrame, engine_names: List[str]) -> tuple:
    """Generates comprehensive summary statistics for multiple models"""
    overall_stats = {}
    
    score_dims = ['scientific_accuracy_score', 'functional_equivalence_score', 
                  'reasoning_quality_score', 'completeness_score', 'overall_score_score']
    
    all_models = ['baseline'] + engine_names
    
    for model in all_models:
        for dim in score_dims:
            overall_stats[f'{model}_{dim}_mean'] = results_df[f'{model}_{dim}'].mean()
            overall_stats[f'{model}_{dim}_std'] = results_df[f'{model}_{dim}'].std()
    
    # Calculate improvements over baseline for each engine
    for engine_name in engine_names:
        for dim in score_dims:
            baseline_mean = overall_stats[f'baseline_{dim}_mean']
            engine_mean = overall_stats[f'{engine_name}_{dim}_mean']
            overall_stats[f'{engine_name}_{dim}_improvement'] = engine_mean - baseline_mean
        
        # Win rates
        overall_stats[f'{engine_name}_win_rate'] = (results_df[f'{engine_name}_improvement'] > 0).mean()
    
    # Task statistics
    task_stats = {}
    for task in ['forward_prediction', 'inverse_design']:
        task_data = results_df[results_df['task'] == task]
        task_stats[task] = {}
        for model in all_models:
            task_stats[task][f'{model}_mean'] = task_data[f'{model}_overall_score_score'].mean()
            task_stats[task][f'{model}_std'] = task_data[f'{model}_overall_score_score'].std()
    
    return overall_stats, task_stats

def parse_arguments():
    """Parse command line arguments with support for multiple engines"""
    parser = argparse.ArgumentParser(description="Evaluate multiple causal reasoning engines")
    parser.add_argument("--test_data", default="outputs/test_doping_data.json", 
                       help="Path to test data JSON file")
    parser.add_argument("--training_graph", default="outputs/combined_doping_data.json",
                       help="Path to training graph JSON file")
    parser.add_argument("--engines", nargs='+', required=True,
                       help="Engine specifications in format 'label:module_name'")
    parser.add_argument("--output_prefix", default=None,
                       help="Output file prefix (default: derived from test data)")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    # Parse engine specifications
    engines = {}
    engine_names = []
    
    for engine_spec in args.engines:
        if ':' not in engine_spec:
            print(f"Invalid engine specification: {engine_spec}")
            print("Use format: 'label:module_name'")
            sys.exit(1)
        
        label, module_name = engine_spec.split(':', 1)
        engine = load_engine_from_module(module_name, args.training_graph)
        if engine is None:
            print(f"Failed to load engine from module: {module_name}")
            sys.exit(1)
        
        engines[label] = engine
        engine_names.append(label)
        print(f"Loaded engine '{label}' from module '{module_name}'")
    
    # Set output prefix
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = get_file_prefix(args.test_data) + "_multi_model"
    
    print(f"Using test data file: {args.test_data}")
    print(f"Output files will use prefix: {output_prefix}")
    print(f"Evaluating {len(engines)} engines: {', '.join(engine_names)}")
    
    # Initialize baseline and evaluator
    print("Initializing baseline model and evaluator...")
    baseline_model = BaselineGeminiModel()
    llm_evaluator = LLMEvaluator()
    
    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    test_experiments = load_test_data(args.test_data)
    print(f"Loaded {len(test_experiments)} test experiments")
    
    # Run evaluation
    print(f"\nEvaluating models on {len(test_experiments)} test cases...")
    results_df = evaluate_multiple_models_on_test_set(
        engines, baseline_model, llm_evaluator, test_experiments
    )
    
    # Save detailed results
    results_csv = f"{output_prefix}_evaluation_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nDetailed results saved to '{results_csv}'")
    
    # Generate summary statistics
    overall_stats, task_stats = generate_multi_model_summary_statistics(results_df, engine_names)
    
    # Save summary statistics
    summary_json = f"{output_prefix}_summary_statistics.json"
    with open(summary_json, "w") as f:
        json.dump(overall_stats, f, indent=2) 
        print(f"Summary statistics saved to '{summary_json}'")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_multi_model_results(results_df, engine_names, output_prefix)
    
    # Print key findings
    print("\n" + "="*80)
    print("MULTI-MODEL EVALUATION RESULTS:")
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
    
    print(f"\nEvaluation complete. All output files use prefix: '{output_prefix}'")

# Example usage:
# python multi_model_evaluation.py --engines "DAG-v1:causal_engine" "DAG-v2:causal_engine_v2" "Enhanced:enhanced_causal_engine"

# # Evaluate multiple engines
# python multi_model_evaluation.py \
#     --engines "DAG-Basic:causal_engine" "DAG-Enhanced:causal_engine_v2" "DAG-Advanced:advanced_engine" \
#     --test_data "outputs/test_doping_data.json" \
#     --training_graph "outputs/combined_doping_data.json"

# # With custom output prefix
# python multi_model_evaluation.py \
#     --engines "DAG:causal_engine_stable" "DAG+CoT:causal_engine" \
#     --output_prefix "comparison_study"

# # Evaluate just two engines
# python multi_model_evaluation.py \
#     --engines "Original:causal_engine" "Improved:new_causal_engine"