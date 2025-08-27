import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
from typing import Dict, List, Tuple, Any
import pandas as pd
import seaborn as sns
import re

# Import the causal reasoning engine
from causal_engine import CausalReasoningEngine

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
        
        Provide your answer in JSON format with 'predicted_properties' and 'reasoning' fields.
        """
        
        response = self.model.generate_content(prompt)
        try:
            # Extract JSON from response
            import re
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response.text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            else:
                return {"predicted_properties": {}, "reasoning": "Failed to parse response"}
        except:
            return {"predicted_properties": {}, "reasoning": "Error in response"}
    
    def inverse_design(self, desired_properties: dict) -> dict:
        prompt = f"""
        You are an expert materials scientist. Suggest synthesis conditions to achieve 
        the following material properties.
        
        Desired Properties:
        {json.dumps(desired_properties, indent=2)}
        
        Provide your answer in JSON format with 'suggested_synthesis_conditions' and 'reasoning' fields.
        """
        
        response = self.model.generate_content(prompt)
        try:
            import re
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response.text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            else:
                return {"suggested_synthesis_conditions": {}, "reasoning": "Failed to parse response"}
        except:
            return {"suggested_synthesis_conditions": {}, "reasoning": "Error in response"}

def perturb_value(value: Any, perturbation_level: float, gemini_model: BaselineGeminiModel = None) -> Any:
    """
    Perturbs a single value based on its type and perturbation level.
    Handles nested dictionaries, lists, numbers, strings, and None values.
    """
    if value is None or value == "null" or value == "not reported":
        return value
    
    if isinstance(value, dict):
        # Recursively perturb dictionary values
        perturbed_dict = {}
        for k, v in value.items():
            perturbed_dict[k] = perturb_value(v, perturbation_level, gemini_model)
        return perturbed_dict
    
    elif isinstance(value, list):
        # Perturb each element in the list
        return [perturb_value(item, perturbation_level, gemini_model) for item in value]
    
    elif isinstance(value, (int, float)):
        # Perturb numerical values
        if perturbation_level == 0:
            return value
        perturbed = value * (1 + np.random.uniform(-perturbation_level, perturbation_level))
        return round(perturbed, 2) if isinstance(value, float) else int(round(perturbed))
    
    elif isinstance(value, str):
        # Handle strings containing numbers
        numbers = re.findall(r'-?\d+\.?\d*', value)
        if numbers and perturbation_level > 0:
            # Replace numbers in string with perturbed values
            result = value
            for num_str in numbers:
                try:
                    num = float(num_str)
                    perturbed_num = num * (1 + np.random.uniform(-perturbation_level, perturbation_level))
                    if '.' in num_str:
                        result = result.replace(num_str, f"{perturbed_num:.2f}", 1)
                    else:
                        result = result.replace(num_str, str(int(round(perturbed_num))), 1)
                except:
                    pass
            return result
        
        # For non-numeric strings, use variations based on perturbation level
        if perturbation_level < 0.3:
            return value
        elif perturbation_level < 0.6:
            # Slight variations
            variations = {
                "electrochemical intercalation": "electrochemical insertion",
                "adsorption": "surface adsorption",
                "chemisorption": "chemical adsorption",
                "vdW gap": "van der Waals gap",
                "p-type": "hole-based conductivity",
                "n-type": "electron-based conductivity",
                "increased": "enhanced",
                "decreased": "reduced"
            }
            return variations.get(value, value)
        else:
            # Major variations - use Gemini to generate semantically similar alternatives
            if gemini_model and len(value) > 5:  # Only for meaningful strings
                try:
                    prompt = f"Give a scientifically equivalent but differently worded version of: '{value}'. Return only the alternative phrase, no explanation."
                    response = gemini_model.model.generate_content(prompt)
                    return response.text.strip()
                except:
                    return value
            return value
    
    return value

def create_perturbed_queries(ground_truth_data: dict, perturbation_levels: List[float], 
                           gemini_model: BaselineGeminiModel = None) -> List[Dict]:
    """
    Creates test cases with different levels of perturbation from ground truth.
    
    Args:
        ground_truth_data: Original data from JSON
        perturbation_levels: List of perturbation strengths (0.0 = exact, 1.0 = very different)
        gemini_model: Optional Gemini model for generating semantic variations
    
    Returns:
        List of test cases with varying confidence levels
    """
    test_cases = []
    
    # Handle both old format (doping_experiments) and new format (direct list)
    if isinstance(ground_truth_data, dict) and "doping_experiments" in ground_truth_data:
        experiments = ground_truth_data["doping_experiments"]
    elif isinstance(ground_truth_data, list):
        experiments = ground_truth_data
    else:
        print("Warning: Unexpected data format. Attempting to extract experiments...")
        experiments = []
    
    # Limit to first 10 experiments for efficiency
    experiments = experiments[:10] if len(experiments) > 10 else experiments
    
    for exp_idx, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            continue
            
        # Extract synthesis conditions and property changes
        synthesis_gt = experiment.get("synthesis_conditions", {})
        properties_gt = experiment.get("property_changes", {})
        
        for pert_level in perturbation_levels:
            # Create perturbed synthesis conditions
            perturbed_synthesis = perturb_value(synthesis_gt, pert_level, gemini_model)
            
            # Create perturbed properties
            perturbed_properties = perturb_value(properties_gt, pert_level, gemini_model)
            
            test_cases.append({
                "experiment_id": f"{experiment.get('experiment_id', exp_idx)}_{pert_level}",
                "perturbation_level": pert_level,
                "synthesis_conditions": perturbed_synthesis,
                "property_changes": perturbed_properties,
                "ground_truth_synthesis": synthesis_gt,
                "ground_truth_properties": properties_gt,
                "host_material": experiment.get("host_material", "unknown"),
                "dopant": experiment.get("dopant", {})
            })
    
    return test_cases

def flatten_dict(d: dict, parent_key: str = '', sep: str = ' ') -> dict:
    """Flattens nested dictionary for comparison"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to strings
            items.append((new_key, ' '.join(str(item) for item in v)))
        elif v is not None and str(v).lower() not in ["", "not reported", "null", "none"]:
            items.append((new_key, str(v)))
    return dict(items)

def dict_to_string(d: dict) -> str:
    """Converts dictionary to string for embedding"""
    if not d or not isinstance(d, dict):
        return ""
    flattened = flatten_dict(d)
    return " and ".join([f"{key} is {value}" for key, value in sorted(flattened.items())])

def calculate_semantic_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Calculates cosine similarity between text embeddings"""
    if not text1 or not text2:
        return 0.0
    try:
        embeddings = model.encode([text1, text2])
        similarity_matrix = cosine_similarity(embeddings)
        return float(similarity_matrix[0, 1])
    except:
        return 0.0

def evaluate_models_comprehensive(
    dag_engine: CausalReasoningEngine,
    baseline_model: BaselineGeminiModel,
    test_cases: List[Dict],
    embedding_model: SentenceTransformer
) -> pd.DataFrame:
    """
    Evaluates both DAG-LLM and baseline models on test cases.
    
    Returns:
        DataFrame with evaluation results
    """
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\nEvaluating test case {i+1}/{len(test_cases)} (perturbation: {test_case['perturbation_level']:.2f})")
        
        # Forward prediction evaluation
        synthesis = test_case["synthesis_conditions"]
        gt_properties = test_case["ground_truth_properties"]
        
        try:
            # DAG-LLM prediction
            dag_result = dag_engine.forward_prediction(synthesis)
            dag_predicted = dag_result.get("predicted_properties", {})
            dag_confidence = dag_result.get("confidence", 0.0)
        except Exception as e:
            print(f"  Warning: DAG-LLM forward prediction failed: {e}")
            dag_predicted = {}
            dag_confidence = 0.0
        
        try:
            # Baseline prediction
            baseline_result = baseline_model.forward_prediction(synthesis)
            baseline_predicted = baseline_result.get("predicted_properties", {})
        except Exception as e:
            print(f"  Warning: Baseline forward prediction failed: {e}")
            baseline_predicted = {}
        
        # Calculate similarities
        gt_str = dict_to_string(gt_properties)
        dag_str = dict_to_string(dag_predicted)
        baseline_str = dict_to_string(baseline_predicted)
        
        dag_similarity = calculate_semantic_similarity(gt_str, dag_str, embedding_model)
        baseline_similarity = calculate_semantic_similarity(gt_str, baseline_str, embedding_model)
        
        results.append({
            "task": "forward_prediction",
            "experiment_id": test_case["experiment_id"],
            "host_material": test_case.get("host_material", "unknown"),
            "perturbation_level": test_case["perturbation_level"],
            "dag_confidence": dag_confidence,
            "dag_similarity": dag_similarity,
            "baseline_similarity": baseline_similarity,
            "dag_improvement": dag_similarity - baseline_similarity
        })
        
        # Inverse design evaluation
        properties = test_case["property_changes"]
        gt_synthesis = test_case["ground_truth_synthesis"]
        
        try:
            # DAG-LLM suggestion
            dag_result = dag_engine.inverse_design(properties)
            dag_suggested = dag_result.get("suggested_synthesis_conditions", {})
            dag_confidence = dag_result.get("confidence", 0.0)
        except Exception as e:
            print(f"  Warning: DAG-LLM inverse design failed: {e}")
            dag_suggested = {}
            dag_confidence = 0.0
        
        try:
            # Baseline suggestion
            baseline_result = baseline_model.inverse_design(properties)
            baseline_suggested = baseline_result.get("suggested_synthesis_conditions", {})
        except Exception as e:
            print(f"  Warning: Baseline inverse design failed: {e}")
            baseline_suggested = {}
        
        # Calculate similarities
        gt_str = dict_to_string(gt_synthesis)
        dag_str = dict_to_string(dag_suggested)
        baseline_str = dict_to_string(baseline_suggested)
        
        dag_similarity = calculate_semantic_similarity(gt_str, dag_str, embedding_model)
        baseline_similarity = calculate_semantic_similarity(gt_str, baseline_str, embedding_model)
        
        results.append({
            "task": "inverse_design",
            "experiment_id": test_case["experiment_id"],
            "host_material": test_case.get("host_material", "unknown"),
            "perturbation_level": test_case["perturbation_level"],
            "dag_confidence": dag_confidence,
            "dag_similarity": dag_similarity,
            "baseline_similarity": baseline_similarity,
            "dag_improvement": dag_similarity - baseline_similarity
        })
    
    return pd.DataFrame(results)

def plot_comprehensive_results(results_df: pd.DataFrame):
    """Creates comprehensive visualization of evaluation results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Confidence vs Similarity for DAG-LLM
    ax = axes[0, 0]
    for task in ["forward_prediction", "inverse_design"]:
        task_data = results_df[results_df["task"] == task]
        ax.scatter(task_data["dag_confidence"], task_data["dag_similarity"], 
                  alpha=0.6, label=task.replace("_", " ").title(), s=50)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel("DAG-LLM Confidence")
    ax.set_ylabel("Semantic Similarity")
    ax.set_title("DAG-LLM: Confidence vs Performance")
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
        ax.plot(grouped.index, grouped["dag_similarity"], 'o-', 
               label=f"DAG-LLM ({task})", markersize=8)
        ax.plot(grouped.index, grouped["baseline_similarity"], 's--', 
               label=f"Baseline ({task})", markersize=8)
    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("Average Semantic Similarity")
    ax.set_title("Model Robustness to Input Perturbations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Improvement over baseline
    ax = axes[0, 2]
    improvement_data = results_df.groupby(["task", "perturbation_level"])["dag_improvement"].mean().reset_index()
    for task in ["forward_prediction", "inverse_design"]:
        task_improvement = improvement_data[improvement_data["task"] == task]
        ax.bar(task_improvement["perturbation_level"] + (0.02 if task == "forward_prediction" else -0.02),
               task_improvement["dag_improvement"], width=0.04, 
               label=task.replace("_", " ").title())
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("DAG-LLM Improvement over Baseline")
    ax.set_title("Relative Performance Gain")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Confidence distribution by perturbation level
    ax = axes[1, 0]
    sns.boxplot(data=results_df, x="perturbation_level", y="dag_confidence", 
                hue="task", ax=ax)
    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("DAG-LLM Confidence")
    ax.set_title("Confidence Distribution by Perturbation Level")
    
    # 5. Performance heatmap
    ax = axes[1, 1]
    pivot_data = results_df.pivot_table(
        values="dag_similarity", 
        index="perturbation_level", 
        columns="task", 
        aggfunc="mean"
    )
    sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
    ax.set_title("DAG-LLM Performance Heatmap")
    
    # 6. Scatter: DAG vs Baseline similarity
    ax = axes[1, 2]
    scatter = ax.scatter(results_df["baseline_similarity"], results_df["dag_similarity"], 
                        alpha=0.5, c=results_df["perturbation_level"], cmap="viridis")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel("Baseline Similarity")
    ax.set_ylabel("DAG-LLM Similarity")
    ax.set_title("Head-to-Head Model Comparison")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Perturbation Level")
    
    plt.tight_layout()
    plt.savefig("comprehensive_evaluation_results.png", dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Generates summary statistics table"""
    summary = results_df.groupby(["task", "perturbation_level"]).agg({
        "dag_confidence": ["mean", "std"],
        "dag_similarity": ["mean", "std"],
        "baseline_similarity": ["mean", "std"],
        "dag_improvement": ["mean", "std", lambda x: (x > 0).sum() / len(x)]
    }).round(3)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.rename(columns={
        "dag_improvement_<lambda>": "dag_improvement_win_rate"
    }, inplace=True)
    
    return summary

if __name__ == '__main__':
    # Setup paths
    training_graph_file = 'outputs/combined_doping_data.json'
    
    # Initialize models
    print("Initializing models...")
    dag_engine = CausalReasoningEngine(training_graph_file)
    baseline_model = BaselineGeminiModel()
    embedding_model = dag_engine.embedding_model
    
    # Load training data to create test cases
    with open(training_graph_file, 'r') as f:
        training_data = json.load(f)
    
    # Create test cases with varying perturbation levels
    perturbation_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    print(f"Creating test cases with perturbation levels: {perturbation_levels}")
    
    # Pass the baseline model for generating semantic variations at high perturbation levels
    test_cases = create_perturbed_queries(training_data, perturbation_levels, baseline_model)
    
    print(f"Created {len(test_cases)} test cases")
    
    # Run comprehensive evaluation
    print(f"\nEvaluating {len(test_cases)} test cases...")
    results_df = evaluate_models_comprehensive(dag_engine, baseline_model, test_cases, embedding_model)
    
    # Save detailed results
    results_df.to_csv("detailed_evaluation_results.csv", index=False)
    print("\nDetailed results saved to 'detailed_evaluation_results.csv'")
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(results_df)
    summary_stats.to_csv("evaluation_summary_statistics.csv")
    print("Summary statistics saved to 'evaluation_summary_statistics.csv'")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_comprehensive_results(results_df)
    
    # Print key findings
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)
    
    # Average performance by model
    avg_dag = results_df["dag_similarity"].mean()
    avg_baseline = results_df["baseline_similarity"].mean()
    print(f"\nAverage Semantic Similarity:")
    print(f"  DAG-LLM: {avg_dag:.3f}")
    print(f"  Baseline: {avg_baseline:.3f}")
    print(f"  Improvement: {avg_dag - avg_baseline:.3f} ({(avg_dag - avg_baseline)/avg_baseline*100:.1f}%)")
    
    # Performance by perturbation level
    print("\nDAG-LLM Advantage by Perturbation Level:")
    for pert_level in perturbation_levels:
        level_data = results_df[results_df["perturbation_level"] == pert_level]
        if len(level_data) > 0:
            avg_improvement = level_data["dag_improvement"].mean()
            win_rate = (level_data["dag_improvement"] > 0).sum() / len(level_data)
            print(f"  Level {pert_level}: +{avg_improvement:.3f} similarity ({win_rate*100:.0f}% win rate)")
    
    # Confidence calibration
    print("\nConfidence Calibration (Pearson correlation):")
    for task in ["forward_prediction", "inverse_design"]:
        task_data = results_df[results_df["task"] == task]
        if len(task_data) > 0:
            correlation = task_data["dag_confidence"].corr(task_data["dag_similarity"])
            print(f"  {task}: {correlation:.3f}")
    
    # Save example test cases for inspection
    with open("example_test_cases.json", "w") as f:
        json.dump(test_cases[:5], f, indent=2)
    print("\nExample test cases saved to 'example_test_cases.json'")