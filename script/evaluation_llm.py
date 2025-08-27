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
import time
import jhu_colors
from jhu_colors import get_jhu_color

# Import the causal reasoning engine from your existing file
from causal_engine import CausalReasoningEngine

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

# --- NEW: LLM-BASED EVALUATOR ---

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

# --- DATA PERTURBATION (Unchanged) ---
def perturb_value(value: Any, perturbation_level: float, gemini_model: BaselineGeminiModel = None) -> Any:
    # ... (perturbation logic remains the same as your provided script)
    if value is None or value == "null" or value == "not reported":
        return value
    if isinstance(value, dict):
        return {k: perturb_value(v, perturbation_level, gemini_model) for k, v in value.items()}
    if isinstance(value, list):
        return [perturb_value(item, perturbation_level, gemini_model) for item in value]
    if isinstance(value, (int, float)):
        if perturbation_level == 0: return value
        perturbed = value * (1 + np.random.uniform(-perturbation_level, perturbation_level))
        return round(perturbed, 2) if isinstance(value, float) else int(round(perturbed))
    if isinstance(value, str):
        numbers = re.findall(r'-?\d+\.?\d*', value)
        if numbers and perturbation_level > 0:
            result = value
            for num_str in numbers:
                try:
                    num = float(num_str)
                    perturbed_num = num * (1 + np.random.uniform(-perturbation_level, perturbation_level))
                    result = result.replace(num_str, f"{perturbed_num:.2f}" if '.' in num_str else str(int(round(perturbed_num))), 1)
                except: pass
            return result
        if perturbation_level < 0.3: return value
        elif perturbation_level < 0.6:
            variations = {"p-type": "hole-based conductivity", "n-type": "electron-based conductivity", "increased": "enhanced"}
            return variations.get(value, value)
        else:
            if gemini_model and len(value) > 5:
                try:
                    prompt = f"Give a scientifically equivalent but differently worded version of: '{value}'. Return only the alternative phrase."
                    response = gemini_model.model.generate_content(prompt)
                    return response.text.strip()
                except: return value
            return value
    return value

def create_perturbed_queries(ground_truth_data: dict, perturbation_levels: List[float], gemini_model: BaselineGeminiModel = None) -> List[Dict]:
    # ... (logic remains the same)
    test_cases = []
    experiments = ground_truth_data.get("doping_experiments", [])[:10] # Limit for efficiency
    for exp_idx, experiment in enumerate(experiments):
        synthesis_gt = experiment.get("synthesis_conditions", {})
        properties_gt = experiment.get("property_changes", {})
        for pert_level in perturbation_levels:
            test_cases.append({
                "experiment_id": f"{experiment.get('experiment_id', exp_idx)}_{pert_level}",
                "perturbation_level": pert_level,
                "synthesis_conditions": perturb_value(synthesis_gt, pert_level, gemini_model),
                "property_changes": perturb_value(properties_gt, pert_level, gemini_model),
                "ground_truth_synthesis": synthesis_gt,
                "ground_truth_properties": properties_gt,
                "host_material": experiment.get("host_material", "unknown"),
            })
    return test_cases

# --- UPDATED EVALUATION WORKFLOW ---

def evaluate_models_comprehensive(
    dag_engine: CausalReasoningEngine,
    baseline_model: BaselineGeminiModel,
    llm_evaluator: LLMEvaluator,
    test_cases: List[Dict]
) -> pd.DataFrame:
    """Evaluates both models using the LLM-based evaluator."""
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\nEvaluating test case {i+1}/{len(test_cases)} (Perturbation: {test_case['perturbation_level']:.2f})")
        
        # --- Forward Prediction Evaluation ---
        print("  - Task: Forward Prediction")
        synthesis_q = test_case["synthesis_conditions"]
        properties_gt = test_case["ground_truth_properties"]
        
        dag_pred_result = dag_engine.forward_prediction(synthesis_q)
        baseline_pred_result = baseline_model.forward_prediction(synthesis_q)
        
        print("    - Evaluating DAG-LLM output...")
        dag_scores = llm_evaluator.evaluate_output("Forward Prediction", synthesis_q, properties_gt, dag_pred_result)
        print("    - Evaluating Baseline output...")
        baseline_scores = llm_evaluator.evaluate_output("Forward Prediction", synthesis_q, properties_gt, baseline_pred_result)

        results.append({
            "task": "forward_prediction", "perturbation_level": test_case["perturbation_level"],
            "host_material": test_case["host_material"],
            "dag_confidence": dag_pred_result.get("confidence", 0.0),
            **{f"dag_{k}": v for k, v in dag_scores.items()},
            **{f"baseline_{k}": v for k, v in baseline_scores.items()},
            "dag_improvement": dag_scores.get('overall_score_score', 0.0) - baseline_scores.get('overall_score_score', 0.0)
        })

        # --- Inverse Design Evaluation ---
        print("  - Task: Inverse Design")
        properties_q = test_case["property_changes"]
        synthesis_gt = test_case["ground_truth_synthesis"]

        dag_inv_result = dag_engine.inverse_design(properties_q)
        baseline_inv_result = baseline_model.inverse_design(properties_q)

        print("    - Evaluating DAG-LLM output...")
        dag_scores = llm_evaluator.evaluate_output("Inverse Design", properties_q, synthesis_gt, dag_inv_result)
        print("    - Evaluating Baseline output...")
        baseline_scores = llm_evaluator.evaluate_output("Inverse Design", properties_q, synthesis_gt, baseline_inv_result)

        results.append({
            "task": "inverse_design", "perturbation_level": test_case["perturbation_level"],
            "host_material": test_case["host_material"],
            "dag_confidence": dag_inv_result.get("confidence", 0.0),
            **{f"dag_{k}": v for k, v in dag_scores.items()},
            **{f"baseline_{k}": v for k, v in baseline_scores.items()},
            "dag_improvement": dag_scores.get('overall_score_score', 0.0) - baseline_scores.get('overall_score_score', 0.0)
        })

    return pd.DataFrame(results)

# --- UPDATED VISUALIZATION ---

def plot_comprehensive_results_llm(results_df: pd.DataFrame):
    """Creates comprehensive visualization based on LLM evaluation scores."""
    jhu_colors.use_style()
    task_colors = {"forward_prediction": get_jhu_color('Heritage Blue'), "inverse_design": get_jhu_color('Spirit Blue')}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    
    # 1. NEW: Radar Chart for Multi-Dimensional Score Comparison
    ax = axes[0, 0]
    ax.remove()
    ax = fig.add_subplot(2, 3, 1, polar=True)
    score_dims = ['Scientific Accuracy', 'Functional Equivalence', 'Reasoning Quality', 'Completeness']
    score_cols = [f"{dim.lower().replace(' ', '_')}_score" for dim in score_dims]
    labels = np.array(score_dims)
    
    dag_means = results_df[[f"dag_{col}" for col in score_cols]].mean().values
    baseline_means = results_df[[f"baseline_{col}" for col in score_cols]].mean().values
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats_dag = np.concatenate((dag_means, [dag_means[0]]))
    stats_baseline = np.concatenate((baseline_means, [baseline_means[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    ax.plot(angles, stats_dag, 'o-', linewidth=2, label='DAG-LLM', color=get_jhu_color('Heritage Blue'))
    ax.fill(angles, stats_dag, alpha=0.25, color=get_jhu_color('Heritage Blue'))
    ax.plot(angles, stats_baseline, 'o-', linewidth=2, label='Baseline', color=get_jhu_color('Red'))
    ax.fill(angles, stats_baseline, alpha=0.25, color=get_jhu_color('Red'))
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
    ax.set_title("Multi-Dimensional Performance Profile")
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 2. Perturbation Level vs Overall Score
    ax = axes[0, 1]
    sns.lineplot(data=results_df, x="perturbation_level", y="dag_overall_score_score", 
                 hue="task", style="task", markers=True, dashes=False, ax=ax, palette=task_colors,
                 label="DAG-LLM")
    sns.lineplot(data=results_df, x="perturbation_level", y="baseline_overall_score_score", 
                 hue="task", style="task", markers=True, dashes=True, ax=ax, palette={"forward_prediction": get_jhu_color('Orange'), "inverse_design": get_jhu_color('Red')})
    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("Average Overall Score")
    ax.set_title("Model Robustness to Perturbations")
    ax.legend(title="Model & Task")

    # 3. Improvement over baseline (based on Overall Score)
    ax = axes[0, 2]
    sns.barplot(data=results_df, x="perturbation_level", y="dag_improvement", hue="task", ax=ax, palette=task_colors)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("Improvement in Overall Score")
    ax.set_title("Relative Performance Gain")

    # 4. Confidence distribution by perturbation level
    ax = axes[1, 0]
    sns.boxplot(data=results_df, x="perturbation_level", y="dag_confidence", hue="task", ax=ax, palette=task_colors)
    ax.set_xlabel("Perturbation Level")
    ax.set_ylabel("DAG-LLM Confidence")
    ax.set_title("Confidence Distribution")

    # 5. Performance heatmap (Overall Score)
    ax = axes[1, 1]
    pivot_data = results_df.pivot_table(values="dag_overall_score_score", index="perturbation_level", columns="task", aggfunc="mean")
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="jhu", ax=ax, linewidths=.5)
    ax.set_title("DAG-LLM Performance Heatmap")
    
    # 6. Head-to-Head Scatter (Overall Score)
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
    plt.savefig("comprehensive_llm_evaluation_results.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # --- Setup ---
    training_graph_file = 'outputs/combined_doping_data.json'
    
    # --- Initialize Models ---
    print("Initializing models...")
    dag_engine = CausalReasoningEngine(training_graph_file)
    baseline_model = BaselineGeminiModel()
    llm_evaluator = LLMEvaluator()
    
    # --- Create Test Cases ---
    with open(training_graph_file, 'r') as f:
        training_data = json.load(f)
    
    perturbation_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    print(f"Creating perturbed test cases for evaluation...")
    test_cases = create_perturbed_queries(training_data, perturbation_levels, baseline_model)
    print(f"Created {len(test_cases)} test cases.")
    
    # --- Run Evaluation ---
    results_df = evaluate_models_comprehensive(dag_engine, baseline_model, llm_evaluator, test_cases)
    
    # --- Save and Visualize Results ---
    results_df.to_csv("llm_evaluation_detailed_results.csv", index=False)
    print("\nDetailed LLM-based evaluation results saved to 'llm_evaluation_detailed_results.csv'")
    
    print("\nGenerating new visualizations based on LLM scores...")
    plot_comprehensive_results_llm(results_df)
    
    print("\nEvaluation complete.")

