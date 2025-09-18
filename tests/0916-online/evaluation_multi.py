import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import dashscope
from dashscope import Generation
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
# from project.causal_reasoning_engine import CausalReasoningEngine
import random

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

class BaselineQwenModel:
    """Baseline Qwen model without DAG constraints for comparison"""
    def __init__(self, model_id: str = "qwen-plus"):
        # Configure DashScope API for Qwen
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY environment variable")
        
        dashscope.api_key = api_key
        dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
        self.model_id = model_id
        print(f"Initialized baseline Qwen model: {model_id}")

    def _generate_content(self, prompt: str) -> str:
        """Generate content using Qwen model via DashScope API"""
        try:
            response = Generation.call(
                model=self.model_id,
                prompt=prompt,
                temperature=0.7
            )
            
            if response.status_code == 200:
                return response.output.text
            else:
                print(f"Qwen API Error: {response.code} - {response.message}")
                return f"Error: {response.code} - {response.message}"
                
        except Exception as e:
            print(f"Exception calling Qwen API: {str(e)}")
            return f"Exception: {str(e)}"
    
    def forward_prediction(self, synthesis_inputs: dict) -> dict:
        prompt = f"""
        You are an expert materials scientist. Based on the following synthesis conditions, 
        predict the resulting material properties.
        
        Synthesis Conditions:
        {json.dumps(synthesis_inputs, indent=2)}
        
        Provide your answer in a structured JSON format within a ```json block with 'predicted_properties' and 'reasoning' fields.
        
        Example format:
        ```json
        {{
            "predicted_properties": {{
                "carrier_type": "p-type",
                "band_gap_ev": 1.5,
                "conductivity": "enhanced"
            }},
            "reasoning": "Detailed explanation of the prediction based on materials science principles."
        }}
        ```
        """
        response_text = self._generate_content(prompt)
        return extract_json_from_response(response_text)
    
    def inverse_design(self, desired_properties: dict) -> dict:
        prompt = f"""
        You are an expert materials scientist. Suggest synthesis conditions to achieve 
        the following material properties.
        
        Desired Properties:
        {json.dumps(desired_properties, indent=2)}
        
        Provide your answer in a structured JSON format within a ```json block with 'suggested_synthesis_conditions' and 'reasoning' fields.
        
        Example format:
        ```json
        {{
            "suggested_synthesis_conditions": {{
                "method": "Chemical Vapor Deposition",
                "temperature_c": 800,
                "pressure_torr": 100,
                "precursor": "MoCl5"
            }},
            "reasoning": "Detailed explanation of why these conditions should achieve the desired properties."
        }}
        ```
        """
        response_text = self._generate_content(prompt)
        return extract_json_from_response(response_text)

# --- ENHANCED LLM-BASED EVALUATOR ---

class EnhancedLLMEvaluator:
    """Enhanced evaluator with A/B testing and detailed analysis capabilities."""
    def __init__(self, model_id: str = "gemini-2.0-flash-exp"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY for evaluator.")
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(
            model_id,
            generation_config={"response_mime_type": "application/json"}
        )
        print(f"Initialized Enhanced LLM evaluator with {model_id}")
        
        self.evaluation_prompt_template = """
        You are an expert materials scientist serving as an impartial judge with exceptional analytical capabilities. Your task is to evaluate a language model's generated output against a ground truth answer for a materials science problem using rigorous scientific standards.

        **Problem Context:**
        - **Task Type:** {task_type}
        - **Input Query:** {input_query}

        **Ground Truth Answer:**
        {ground_truth}

        **Model's Generated Answer:**
        {generated_answer}

        **Enhanced Evaluation Criteria:**
        Please provide a score from 0 to 10 (integer) for each dimension. Apply rigorous scientific standards and consider both explicit content and implicit scientific understanding.

        1. **Scientific Accuracy (0-10):** 
           - Evaluate thermodynamic feasibility, kinetic plausibility, and adherence to known materials science principles
           - Consider electronic structure, defect chemistry, and phase stability
           - Score 0: Violates fundamental laws; 10: Perfectly consistent with established science

        2. **Functional Equivalence (0-10):** 
           - Assess whether the generated answer would achieve the same practical outcome as the ground truth
           - Consider alternative but equally valid approaches
           - Score 0: Completely different functional outcome; 10: Functionally identical or superior alternative

        3. **Reasoning Quality (0-10):** 
           - Evaluate logical consistency, mechanistic understanding, and depth of analysis
           - Consider chain-of-thought clarity, consideration of alternatives, and scientific rigor
           - Score 0: No reasoning or fundamentally flawed logic; 10: Exceptional scientific reasoning with deep insights

        4. **Completeness (0-10):** 
           - Assess coverage of essential parameters, conditions, and considerations
           - Consider practical implementability and missing critical details
           - Score 0: Missing most essential information; 10: Comprehensive coverage of all necessary aspects

        5. **Overall Scientific Merit (0-10):** 
           - Holistic assessment considering innovation, practicality, and scientific contribution
           - Weight accuracy and reasoning quality most heavily
           - Score 0: Scientifically unsound or impractical; 10: Exceptional scientific contribution

        **JSON Schema:**
        {{
            "scientific_accuracy": {{ "score": integer, "justification": "detailed_explanation" }},
            "functional_equivalence": {{ "score": integer, "justification": "detailed_explanation" }},
            "reasoning_quality": {{ "score": integer, "justification": "detailed_explanation" }},
            "completeness": {{ "score": integer, "justification": "detailed_explanation" }},
            "overall_score": {{ "score": integer, "justification": "comprehensive_assessment" }}
        }}
        """

        self.ab_comparison_prompt = """
        You are an expert materials scientist conducting a blind peer review. You will be presented with two responses (Response A and Response B) to the same materials science question. You must choose which response is better and explain why.

        **IMPORTANT: You do not know which model generated which response. Evaluate purely on scientific merit.**

        **Problem Context:**
        - **Task Type:** {task_type}
        - **Input Query:** {input_query}
        - **Ground Truth:** {ground_truth}

        **Response A:**
        {response_a}

        **Response B:**
        {response_b}

        **Evaluation Instructions:**
        1. Compare both responses on scientific accuracy, reasoning quality, completeness, and practical utility
        2. Consider which response would be more helpful to a materials scientist
        3. Evaluate mechanistic understanding and depth of analysis
        4. Choose the better response and provide detailed justification

        **JSON Schema:**
        {{
            "better_response": "A" or "B",
            "confidence": float between 0.0 and 1.0,
            "justification": "detailed explanation of why the chosen response is better",
            "key_differences": ["list of main differences between responses"],
            "scientific_strengths_winner": ["strengths of the winning response"],
            "scientific_weaknesses_loser": ["weaknesses of the losing response"]
        }}
        """

        self.analysis_prompt = """
        You are an expert in knowledge graph systems and materials science. Analyze why one model performed better than another and identify specific contributions from different reasoning approaches.

        **Context:**
        - **Task:** {task_type}
        - **Input:** {input_query}
        - **Ground Truth:** {ground_truth}

        **Baseline Model Response:**
        {baseline_response}

        **DAG-Enhanced Model Response:**
        {dag_response}

        **Performance Comparison:**
        - Baseline Score: {baseline_score:.3f}
        - DAG Score: {dag_score:.3f}
        - Performance Difference: {score_diff:+.3f}

        **Analysis Instructions:**
        {analysis_type}

        **JSON Schema:**
        {{
            "performance_analysis": {{
                "primary_factors": ["main reasons for performance difference"],
                "dag_contributions": ["specific ways DAG knowledge helped or hindered"],
                "baseline_limitations": ["limitations of baseline approach"],
                "knowledge_graph_insights": ["insights about DAG effectiveness"],
                "recommendations": ["suggestions for improvement"]
            }},
            "mechanistic_analysis": {{
                "scientific_depth_comparison": "comparison of scientific understanding",
                "reasoning_approach_differences": "how reasoning approaches differed",
                "knowledge_integration": "how well each model integrated different knowledge sources"
            }},
            "summary": "concise explanation of the performance difference"
        }}
        """

    def evaluate_output(self, task_type: str, input_query: dict, ground_truth: dict, generated_output: dict) -> dict:
        """Performs enhanced evaluation using Gemini Pro's superior reasoning capabilities."""
        prompt = self.evaluation_prompt_template.format(
            task_type=task_type,
            input_query=json.dumps(input_query, indent=2),
            ground_truth=json.dumps(ground_truth, indent=2),
            generated_answer=json.dumps(generated_output, indent=2)
        )
        
        for attempt in range(3):
            try:
                response = self.model.generate_content(prompt)
                evaluation = json.loads(response.text)
                
                # Normalize scores to be out of 1.0 for easier plotting
                scores = {f"{key}_score": val.get("score", 0) / 10.0 for key, val in evaluation.items()}
                scores['justifications'] = {key: val.get("justification", "") for key, val in evaluation.items()}
                
                if attempt > 0:
                    print(f"    - Evaluator succeeded on attempt {attempt+1}")
                
                return scores
                
            except Exception as e:
                print(f"    - Evaluator attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)
                    continue
        
        # Return default scores on complete failure
        print("    - All evaluator attempts failed, using default scores")
        return {
            "scientific_accuracy_score": 0.0, "functional_equivalence_score": 0.0,
            "reasoning_quality_score": 0.0, "completeness_score": 0.0,
            "overall_score_score": 0.0, 'justifications': {"error": "Evaluation failed"}
        }

    def ab_comparison(self, task_type: str, input_query: dict, ground_truth: dict, 
                     response_a: dict, response_b: dict) -> dict:
        """Conducts blind A/B comparison between two responses."""
        # Randomly shuffle responses to ensure true blind evaluation
        if random.random() < 0.5:
            actual_a, actual_b = response_a, response_b
            shuffle_map = {"A": "response_a", "B": "response_b"}
        else:
            actual_a, actual_b = response_b, response_a
            shuffle_map = {"A": "response_b", "B": "response_a"}
        
        prompt = self.ab_comparison_prompt.format(
            task_type=task_type,
            input_query=json.dumps(input_query, indent=2),
            ground_truth=json.dumps(ground_truth, indent=2),
            response_a=json.dumps(actual_a, indent=2),
            response_b=json.dumps(actual_b, indent=2)
        )
        
        for attempt in range(3):
            try:
                response = self.model.generate_content(prompt)
                comparison = json.loads(response.text)
                
                # Map back to original responses
                chosen_letter = comparison.get("better_response", "A")
                comparison["better_response_original"] = shuffle_map[chosen_letter]
                comparison["shuffle_applied"] = actual_a != response_a
                
                return comparison
                
            except Exception as e:
                print(f"    - A/B comparison attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)
                    continue
        
        return {
            "better_response": "A", "better_response_original": "response_a",
            "confidence": 0.0, "justification": "Comparison failed",
            "key_differences": [], "scientific_strengths_winner": [],
            "scientific_weaknesses_loser": [], "shuffle_applied": False
        }

    def analyze_performance_difference(self, task_type: str, input_query: dict, ground_truth: dict,
                                     baseline_response: dict, dag_response: dict,
                                     baseline_score: float, dag_score: float) -> dict:
        """Analyzes why one model performed better and identifies specific contributing factors."""
        score_diff = dag_score - baseline_score
        
        if score_diff > 0.05:  # DAG significantly outperforms
            analysis_type = """
            **Focus: DAG Enhancement Analysis**
            The DAG-enhanced model significantly outperformed the baseline. Analyze:
            1. What specific knowledge from the DAG contributed to better performance?
            2. How did the DAG help with mechanistic understanding?
            3. What causal relationships or pathways were most valuable?
            4. How did DAG knowledge enhance or correct baseline reasoning?
            5. What aspects of the response show clear DAG influence?
            """
        elif score_diff < -0.05:  # Baseline significantly outperforms
            analysis_type = """
            **Focus: DAG Limitation Analysis**
            The baseline model outperformed the DAG-enhanced version. Analyze:
            1. What limitations in the DAG knowledge led to worse performance?
            2. How did DAG integration interfere with sound baseline reasoning?
            3. What conflicts arose between DAG knowledge and fundamental principles?
            4. Were there coverage gaps or quality issues in the DAG?
            5. How could the DAG integration approach be improved?
            """
        else:  # Similar performance
            analysis_type = """
            **Focus: Comparative Analysis**
            Both models performed similarly. Analyze:
            1. What are the trade-offs between DAG and baseline approaches?
            2. In what aspects did each model excel or fall short?
            3. How complementary are the two approaches?
            4. What scenarios favor each approach?
            """
        
        prompt = self.analysis_prompt.format(
            task_type=task_type,
            input_query=json.dumps(input_query, indent=2),
            ground_truth=json.dumps(ground_truth, indent=2),
            baseline_response=json.dumps(baseline_response, indent=2),
            dag_response=json.dumps(dag_response, indent=2),
            baseline_score=baseline_score,
            dag_score=dag_score,
            score_diff=score_diff,
            analysis_type=analysis_type
        )
        
        for attempt in range(3):
            try:
                response = self.model.generate_content(prompt)
                analysis = json.loads(response.text)
                return analysis
                
            except Exception as e:
                print(f"    - Analysis attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)
                    continue
        
        return {
            "performance_analysis": {
                "primary_factors": ["Analysis failed"],
                "dag_contributions": [], "baseline_limitations": [],
                "knowledge_graph_insights": [], "recommendations": []
            },
            "mechanistic_analysis": {
                "scientific_depth_comparison": "Analysis failed",
                "reasoning_approach_differences": "Analysis failed",
                "knowledge_integration": "Analysis failed"
            },
            "summary": "Performance analysis failed"
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

# --- ENHANCED EVALUATION WORKFLOW ---

def evaluate_multiple_models_comprehensive(
    engines: Dict[str, Any],
    baseline_model: BaselineQwenModel,
    llm_evaluator: EnhancedLLMEvaluator,
    test_experiments: List[Dict]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Comprehensive evaluation including detailed analysis and A/B testing.
    Returns: (summary_df, detailed_df)
    """
    summary_results = []
    detailed_results = []
    
    for i, experiment in enumerate(test_experiments):
        print(f"\nEvaluating test case {i+1}/{len(test_experiments)}")
        
        # Extract synthesis conditions and property changes
        synthesis_conditions = experiment.get("synthesis_conditions", {})
        property_changes = experiment.get("property_changes", {})
        
        # --- Forward Prediction Evaluation ---
        print("  - Task: Forward Prediction")
        
        # Get baseline response
        try:
            baseline_response = baseline_model.forward_prediction(synthesis_conditions)
            print("    - Evaluating Baseline with Gemini...")
            baseline_scores = llm_evaluator.evaluate_output(
                "Forward Prediction", synthesis_conditions, property_changes, baseline_response
            )
        except Exception as e:
            print(f"    Warning: Baseline forward prediction failed: {e}")
            baseline_response = {"predicted_properties": {}, "reasoning": "Error"}
            baseline_scores = {
                "scientific_accuracy_score": 0.0, "functional_equivalence_score": 0.0,
                "reasoning_quality_score": 0.0, "completeness_score": 0.0,
                "overall_score_score": 0.0, 'justifications': {}
            }

        # Evaluate each engine
        for engine_name, engine in engines.items():
            try:
                engine_response = engine.forward_prediction(synthesis_conditions)
                engine_confidence = engine_response.get("confidence", 0.0)
                
                print(f"    - Evaluating {engine_name} with Gemini...")
                engine_scores = llm_evaluator.evaluate_output(
                    "Forward Prediction", synthesis_conditions, property_changes, engine_response
                )
                
                # A/B Comparison
                print(f"    - Conducting A/B comparison...")
                ab_result = llm_evaluator.ab_comparison(
                    "Forward Prediction", synthesis_conditions, property_changes,
                    baseline_response, engine_response
                )
                
                # Performance Analysis
                print(f"    - Analyzing performance difference...")
                analysis = llm_evaluator.analyze_performance_difference(
                    "Forward Prediction", synthesis_conditions, property_changes,
                    baseline_response, engine_response,
                    baseline_scores['overall_score_score'], engine_scores['overall_score_score']
                )
                
                # Create detailed record
                detailed_record = {
                    "experiment_id": experiment.get("experiment_id", i),
                    "task": "forward_prediction",
                    "host_material": experiment.get("host_material", "unknown"),
                    "dopant": experiment.get("dopant", {}).get("element", "unknown"),
                    
                    # Input and ground truth
                    "input_query": json.dumps(synthesis_conditions),
                    "ground_truth": json.dumps(property_changes),
                    
                    # Model responses
                    "baseline_response": json.dumps(baseline_response),
                    "engine_response": json.dumps(engine_response),
                    "engine_confidence": engine_confidence,
                    
                    # Individual scores
                    "baseline_scientific_accuracy": baseline_scores['scientific_accuracy_score'],
                    "baseline_functional_equivalence": baseline_scores['functional_equivalence_score'],
                    "baseline_reasoning_quality": baseline_scores['reasoning_quality_score'],
                    "baseline_completeness": baseline_scores['completeness_score'],
                    "baseline_overall_score": baseline_scores['overall_score_score'],
                    
                    "engine_scientific_accuracy": engine_scores['scientific_accuracy_score'],
                    "engine_functional_equivalence": engine_scores['functional_equivalence_score'],
                    "engine_reasoning_quality": engine_scores['reasoning_quality_score'],
                    "engine_completeness": engine_scores['completeness_score'],
                    "engine_overall_score": engine_scores['overall_score_score'],
                    
                    # A/B Comparison
                    "ab_winner": ab_result['better_response_original'],
                    "ab_confidence": ab_result['confidence'],
                    "ab_justification": ab_result['justification'],
                    "ab_key_differences": '; '.join(ab_result['key_differences']),
                    
                    # Performance Analysis
                    "analysis_summary": analysis['summary'],
                    "primary_factors": '; '.join(analysis['performance_analysis']['primary_factors']),
                    "dag_contributions": '; '.join(analysis['performance_analysis']['dag_contributions']),
                    "baseline_limitations": '; '.join(analysis['performance_analysis']['baseline_limitations']),
                    "recommendations": '; '.join(analysis['performance_analysis']['recommendations']),
                    
                    # Score justifications
                    "baseline_justifications": json.dumps(baseline_scores['justifications']),
                    "engine_justifications": json.dumps(engine_scores['justifications']),
                }
                
                detailed_results.append(detailed_record)
                
                # Create summary record
                summary_record = {
                    "task": "forward_prediction",
                    "experiment_id": experiment.get("experiment_id", i),
                    "host_material": experiment.get("host_material", "unknown"),
                    "dopant": experiment.get("dopant", {}).get("element", "unknown"),
                    "engine_name": engine_name,
                    
                    # Core metrics
                    f"baseline_overall_score_score": baseline_scores['overall_score_score'],
                    f"{engine_name}_overall_score_score": engine_scores['overall_score_score'],
                    f"{engine_name}_confidence": engine_confidence,
                    f"{engine_name}_improvement": engine_scores['overall_score_score'] - baseline_scores['overall_score_score'],
                    
                    # All score dimensions for both models
                    **{f"baseline_{k}": v for k, v in baseline_scores.items() if k != 'justifications'},
                    **{f"{engine_name}_{k}": v for k, v in engine_scores.items() if k != 'justifications'},
                }
                
                summary_results.append(summary_record)
                
            except Exception as e:
                print(f"    Warning: {engine_name} forward prediction failed: {e}")
                continue

        # --- Inverse Design Evaluation ---
        print("  - Task: Inverse Design")
        
        # Get baseline response
        try:
            baseline_response = baseline_model.inverse_design(property_changes)
            print("    - Evaluating Baseline with Gemini...")
            baseline_scores = llm_evaluator.evaluate_output(
                "Inverse Design", property_changes, synthesis_conditions, baseline_response
            )
        except Exception as e:
            print(f"    Warning: Baseline inverse design failed: {e}")
            baseline_response = {"suggested_synthesis_conditions": {}, "reasoning": "Error"}
            baseline_scores = {
                "scientific_accuracy_score": 0.0, "functional_equivalence_score": 0.0,
                "reasoning_quality_score": 0.0, "completeness_score": 0.0,
                "overall_score_score": 0.0, 'justifications': {}
            }

        # Evaluate each engine for inverse design
        for engine_name, engine in engines.items():
            try:
                engine_response = engine.inverse_design(property_changes)
                engine_confidence = engine_response.get("confidence", 0.0)
                
                print(f"    - Evaluating {engine_name} with Gemini...")
                engine_scores = llm_evaluator.evaluate_output(
                    "Inverse Design", property_changes, synthesis_conditions, engine_response
                )
                
                # A/B Comparison
                print(f"    - Conducting A/B comparison...")
                ab_result = llm_evaluator.ab_comparison(
                    "Inverse Design", property_changes, synthesis_conditions,
                    baseline_response, engine_response
                )
                
                # Performance Analysis
                print(f"    - Analyzing performance difference...")
                analysis = llm_evaluator.analyze_performance_difference(
                    "Inverse Design", property_changes, synthesis_conditions,
                    baseline_response, engine_response,
                    baseline_scores['overall_score_score'], engine_scores['overall_score_score']
                )
                
                # Create detailed record for inverse design
                detailed_record = {
                    "experiment_id": experiment.get("experiment_id", i),
                    "task": "inverse_design",
                    "host_material": experiment.get("host_material", "unknown"),
                    "dopant": experiment.get("dopant", {}).get("element", "unknown"),
                    
                    # Input and ground truth
                    "input_query": json.dumps(property_changes),
                    "ground_truth": json.dumps(synthesis_conditions),
                    
                    # Model responses
                    "baseline_response": json.dumps(baseline_response),
                    "engine_response": json.dumps(engine_response),
                    "engine_confidence": engine_confidence,
                    
                    # Individual scores
                    "baseline_scientific_accuracy": baseline_scores['scientific_accuracy_score'],
                    "baseline_functional_equivalence": baseline_scores['functional_equivalence_score'],
                    "baseline_reasoning_quality": baseline_scores['reasoning_quality_score'],
                    "baseline_completeness": baseline_scores['completeness_score'],
                    "baseline_overall_score": baseline_scores['overall_score_score'],
                    
                    "engine_scientific_accuracy": engine_scores['scientific_accuracy_score'],
                    "engine_functional_equivalence": engine_scores['functional_equivalence_score'],
                    "engine_reasoning_quality": engine_scores['reasoning_quality_score'],
                    "engine_completeness": engine_scores['completeness_score'],
                    "engine_overall_score": engine_scores['overall_score_score'],
                    
                    # A/B Comparison
                    "ab_winner": ab_result['better_response_original'],
                    "ab_confidence": ab_result['confidence'],
                    "ab_justification": ab_result['justification'],
                    "ab_key_differences": '; '.join(ab_result['key_differences']),
                    
                    # Performance Analysis
                    "analysis_summary": analysis['summary'],
                    "primary_factors": '; '.join(analysis['performance_analysis']['primary_factors']),
                    "dag_contributions": '; '.join(analysis['performance_analysis']['dag_contributions']),
                    "baseline_limitations": '; '.join(analysis['performance_analysis']['baseline_limitations']),
                    "recommendations": '; '.join(analysis['performance_analysis']['recommendations']),
                    
                    # Score justifications
                    "baseline_justifications": json.dumps(baseline_scores['justifications']),
                    "engine_justifications": json.dumps(engine_scores['justifications']),
                }
                
                detailed_results.append(detailed_record)
                
                # Create summary record for inverse design
                summary_record = {
                    "task": "inverse_design",
                    "experiment_id": experiment.get("experiment_id", i),
                    "host_material": experiment.get("host_material", "unknown"),
                    "dopant": experiment.get("dopant", {}).get("element", "unknown"),
                    "engine_name": engine_name,
                    
                    # Core metrics
                    f"baseline_overall_score_score": baseline_scores['overall_score_score'],
                    f"{engine_name}_overall_score_score": engine_scores['overall_score_score'],
                    f"{engine_name}_confidence": engine_confidence,
                    f"{engine_name}_improvement": engine_scores['overall_score_score'] - baseline_scores['overall_score_score'],
                    
                    # All score dimensions for both models
                    **{f" baseline_{k}": v for k, v in baseline_scores.items() if k != 'justifications'},
                    **{f"{engine_name}_{k}": v for k, v in engine_scores.items() if k != 'justifications'},
                }
                
                summary_results.append(summary_record)
                
            except Exception as e:
                print(f"    Warning: {engine_name} inverse design failed: {e}")
                continue

    return pd.DataFrame(summary_results), pd.DataFrame(detailed_results)

# --- VISUALIZATION (Keep existing functions) ---
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
    """Creates comprehensive visualization for multi-model evaluation with Qwen baseline."""
    
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
    fig.suptitle('Multi-Model Evaluation: DAG-Enhanced Engines vs Qwen Baseline\n(Evaluated by Gemini 2.0 Flash)', 
                 fontsize=14, fontweight='bold')
    
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

    # Plot Qwen baseline
    baseline_means = results_df[[f"baseline_{col}" for col in score_cols]].mean().values
    stats_baseline = np.concatenate((baseline_means, [baseline_means[0]]))
    ax.plot(angles, stats_baseline, 'o-', linewidth=2, label='Qwen Baseline', 
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

    # Remove default theta labels and add custom ones
    ax.set_thetagrids(angles[:-1] * 180/np.pi, [])

    # Manually place labels with custom distance
    label_distance = 1.35  # Adjust this value to control distance from circle
    for angle, label in zip(angles[:-1], wrapped_labels):
        ax.text(angle, label_distance, label,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transData)
                
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
        if model == 'baseline':
            model_names.append('Qwen Baseline')
        else:
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
    ax.set_title("Overall Model Performance\n(Gemini 2.0 Flash Evaluation)")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(model_scores) * 1.2)

    # 3. Improvement over Qwen baseline distribution
    ax = axes[0, 2]
    for engine_name in engine_names:
        improvements = results_df[f"{engine_name}_improvement"]
        ax.hist(improvements, bins=15, alpha=0.6, label=engine_name, 
                color=model_colors[engine_name])
    
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel("Improvement over Qwen Baseline")
    ax.set_ylabel("Count")
    ax.set_title("Performance Improvement Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Box plot comparison across all models
    ax = axes[1, 0]
    score_data = []
    for model in all_models:
        for _, row in results_df.iterrows():
            display_name = 'Qwen Baseline' if model == 'baseline' else model.replace('_', ' ').title()
            score_data.append({
                'Model': display_name,
                'Score': row[f"{model}_overall_score_score"]
            })
    
    score_df = pd.DataFrame(score_data)
    model_name_mapping = {'Qwen Baseline': 'baseline'}
    model_name_mapping.update({name.replace('_', ' ').title(): name for name in engine_names})
    
    palette = {}
    for display_name in score_df['Model'].unique():
        original_name = model_name_mapping.get(display_name, display_name.lower().replace(' ', '_'))
        palette[display_name] = model_colors.get(original_name, get_jhu_color('Heritage Blue'))
    
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
    display_names = ['Qwen Baseline'] + [name.replace('_', ' ').title() for name in engine_names]
    ax.set_xticklabels(display_names, rotation=45, ha='right')
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
    ax.set_xlabel("Qwen Baseline Overall Score")
    ax.set_ylabel(f"{best_engine} Overall Score")
    ax.set_title(f"Head-to-Head: {best_engine} vs Qwen")
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
    
    # Calculate improvements over Qwen baseline for each engine
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
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of causal reasoning engines against Qwen baseline")
    parser.add_argument("--test_data", default="outputs/filtered_test_doping_data.json", 
                       help="Path to test data JSON file")
    parser.add_argument("--training_graph", default="outputs/filtered_combined_doping_data.json",
                       help="Path to training graph JSON file")
    parser.add_argument("--engines", nargs='+', required=True,
                       help="Engine specifications in format 'label:module_name'")
    parser.add_argument("--output_prefix", default=None,
                       help="Output file prefix (default: derived from test data)")
    parser.add_argument("--qwen_model", default="qwen-plus",
                       help="Qwen model to use for baseline (default: qwen-plus)")
    parser.add_argument("--evaluator_model", default="gemini-2.0-flash-exp",
                       help="Gemini model to use for evaluation (default: gemini-2.0-flash-exp)")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    print("="*80)
    print("COMPREHENSIVE MULTI-MODEL EVALUATION WITH DETAILED ANALYSIS")
    print("="*80)
    print(f"Qwen Baseline Model: {args.qwen_model}")
    print(f"Gemini Evaluator Model: {args.evaluator_model}")
    print("Features: Detailed CSV output, A/B testing, Performance analysis")
    print("="*80)
    
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
        print(f"✅ Loaded engine '{label}' from module '{module_name}'")
    
    # Set output prefix
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = get_file_prefix(args.test_data) + "_comprehensive_eval"
    
    print(f"\nUsing test data file: {args.test_data}")
    print(f"Output files will use prefix: {output_prefix}")
    print(f"Evaluating {len(engines)} engines: {', '.join(engine_names)}")
    
    # Initialize models
    print("\n" + "-"*50)
    print("INITIALIZING MODELS...")
    print("-"*50)
    
    try:
        baseline_model = BaselineQwenModel(model_id=args.qwen_model)
        print("✅ Qwen baseline model initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Qwen baseline: {e}")
        sys.exit(1)
    
    try:
        llm_evaluator = EnhancedLLMEvaluator(model_id=args.evaluator_model)
        print("✅ Enhanced Gemini evaluator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini evaluator: {e}")
        sys.exit(1)
    
    # Load test data
    print(f"\n📊 Loading test data from {args.test_data}...")
    test_experiments = load_test_data(args.test_data)
    print(f"✅ Loaded {len(test_experiments)} test experiments")
    
    # Run comprehensive evaluation
    print(f"\n🔬 Running comprehensive evaluation on {len(test_experiments)} test cases...")
    print("This includes: Individual scoring, A/B testing, and performance analysis...")
    
    summary_df, detailed_df = evaluate_multiple_models_comprehensive(
        engines, baseline_model, llm_evaluator, test_experiments
    )
    
    # Save results
    summary_csv = f"{output_prefix}_summary_results.csv"
    detailed_csv = f"{output_prefix}_detailed_results.csv"
    
    summary_df.to_csv(summary_csv, index=False)
    detailed_df.to_csv(detailed_csv, index=False)
    
    print(f"\n💾 Summary results saved to '{summary_csv}'")
    print(f"💾 Detailed results with analysis saved to '{detailed_csv}'")
    
    # Generate summary statistics
    overall_stats, task_stats = generate_multi_model_summary_statistics(summary_df, engine_names)
    
    # Save summary statistics
    summary_json = f"{output_prefix}_summary_statistics.json"
    with open(summary_json, "w") as f:
        json.dump(overall_stats, f, indent=2) 
    print(f"💾 Summary statistics saved to '{summary_json}'")
    
    # Create visualizations
    print("\n📈 Generating visualizations...")
    plot_multi_model_results(summary_df, engine_names, output_prefix)
    
    # Print key findings with A/B results
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80)
    
    print(f"\n🏆 Overall Performance Ranking (by overall score):")
    all_models = ['baseline'] + engine_names
    model_scores = [(model, overall_stats[f'{model}_overall_score_score_mean']) 
                   for model in all_models]
    model_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, score) in enumerate(model_scores, 1):
        display_name = 'Qwen Baseline' if model == 'baseline' else model
        print(f"  {i}. {display_name}: {score:.3f}")
    
    print(f"\n📈 Improvement over Qwen Baseline:")
    for engine_name in engine_names:
        improvement = overall_stats[f'{engine_name}_overall_score_score_improvement']
        win_rate = overall_stats[f'{engine_name}_win_rate'] * 100
        
        # Calculate A/B win rate from detailed results
        engine_detailed = detailed_df[detailed_df['experiment_id'].isin(detailed_df['experiment_id'])]
        ab_wins = (engine_detailed['ab_winner'] == 'engine_response').sum()
        total_ab = len(engine_detailed)
        ab_win_rate = (ab_wins / total_ab * 100) if total_ab > 0 else 0
        
        status = "✅" if improvement > 0 else "❌"
        print(f"  {status} {engine_name}: {improvement:+.3f} (Score Win Rate: {win_rate:.1f}%, A/B Win Rate: {ab_win_rate:.1f}%)")
    
    print(f"\n📊 Performance by Task:")
    for task in ['forward_prediction', 'inverse_design']:
        print(f"\n  {task.replace('_', ' ').title()}:")
        task_scores = [(model, task_stats[task][f'{model}_mean']) for model in all_models]
        task_scores.sort(key=lambda x: x[1], reverse=True)
        for model, score in task_scores:
            display_name = 'Qwen Baseline' if model == 'baseline' else model
            print(f"    {display_name}: {score:.3f}")
    
    print(f"\n🎯 Key Insights:")
    best_engine = max(engine_names, key=lambda x: overall_stats[f'{x}_overall_score_score_mean'])
    best_improvement = overall_stats[f'{best_engine}_overall_score_score_improvement']
    
    if best_improvement > 0:
        print(f"  • Best performing engine: {best_engine} (+{best_improvement:.3f} over Qwen)")
        print(f"  • DAG-enhanced reasoning shows measurable improvements")
    else:
        print(f"  • Qwen baseline outperforms all DAG-enhanced engines")
        print(f"  • Consider refining DAG construction or reasoning strategies")
    
    # Analysis insights
    print(f"\n🔍 Analysis Insights:")
    print(f"  • Detailed CSV includes ground truth, model responses, and performance analysis")
    print(f"  • A/B comparisons provide blind evaluation results")
    print(f"  • Performance analysis explains mechanisms behind score differences")
    
    print(f"\n📁 Output Files:")
    print(f"  • Summary: {summary_csv}")
    print(f"  • Detailed: {detailed_csv}")
    print(f"  • Statistics: {summary_json}")
    print(f"  • Visualization: {output_prefix}_multi_model_evaluation_results.png")
    print("="*80)

# Usage examples:
# python enhanced_evaluation.py --engines "DAG-Enhanced:causal_engine_qwen"
# python enhanced_evaluation.py --engines "DAG-Basic:causal_engine_stable" "DAG-CoT:causal_engine"