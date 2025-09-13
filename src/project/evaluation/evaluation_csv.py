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

def format_model_response_for_human_evaluation(response: dict, task_type: str) -> str:
    """
    Converts model responses (JSON or other formats) into human-readable format
    for fair comparison in expert evaluation.
    """
    if not response or not isinstance(response, dict):
        return "No valid response provided."
    
    # Handle error cases
    if "error" in response:
        return f"Error in model response: {response.get('error', 'Unknown error')}"
    
    formatted_parts = []
    
    if task_type == "forward_prediction":
        # Extract predicted properties
        predicted_props = response.get("predicted_properties", {})
        if predicted_props:
            formatted_parts.append("PREDICTED PROPERTIES:")
            for key, value in predicted_props.items():
                # Clean up key names for readability
                clean_key = key.replace('_', ' ').title()
                formatted_parts.append(f"• {clean_key}: {value}")
        
        # Extract mechanistic explanation if available
        mech_exp = response.get("mechanistic_explanation", {})
        if mech_exp:
            formatted_parts.append("\nMECHANISTIC EXPLANATION:")
            for key, value in mech_exp.items():
                clean_key = key.replace('_', ' ').title()
                formatted_parts.append(f"• {clean_key}: {value}")
        
        # Extract chain of thought
        chain_of_thought = response.get("chain_of_thought", [])
        if chain_of_thought:
            formatted_parts.append("\nREASONING STEPS:")
            for i, step in enumerate(chain_of_thought, 1):
                formatted_parts.append(f"{i}. {step}")
        
        # Extract quantitative estimates
        quant_est = response.get("quantitative_estimates", {})
        if quant_est:
            formatted_parts.append("\nQUANTITATIVE ESTIMATES:")
            for key, value in quant_est.items():
                clean_key = key.replace('_', ' ').title()
                formatted_parts.append(f"• {clean_key}: {value}")
        
        # Extract alternative mechanisms
        alt_mech = response.get("alternative_mechanisms", [])
        if alt_mech:
            formatted_parts.append("\nALTERNATIVE APPROACHES:")
            for i, mech in enumerate(alt_mech, 1):
                formatted_parts.append(f"{i}. {mech}")
    
    elif task_type == "inverse_design":
        # Extract suggested synthesis conditions
        synth_cond = response.get("suggested_synthesis_conditions", {})
        if synth_cond:
            formatted_parts.append("SUGGESTED SYNTHESIS CONDITIONS:")
            for key, value in synth_cond.items():
                clean_key = key.replace('_', ' ').title()
                formatted_parts.append(f"• {clean_key}: {value}")
        
        # Extract mechanistic reasoning
        mech_reasoning = response.get("mechanistic_reasoning", {})
        if mech_reasoning:
            formatted_parts.append("\nMECHANISTIC REASONING:")
            for key, value in mech_reasoning.items():
                clean_key = key.replace('_', ' ').title()
                formatted_parts.append(f"• {clean_key}: {value}")
        
        # Extract chain of thought
        chain_of_thought = response.get("chain_of_thought", [])
        if chain_of_thought:
            formatted_parts.append("\nREASONING STEPS:")
            for i, step in enumerate(chain_of_thought, 1):
                formatted_parts.append(f"{i}. {step}")
        
        # Extract uncertainty analysis
        uncertainty = response.get("uncertainty_analysis", {})
        if uncertainty:
            formatted_parts.append("\nUNCERTAINTY ANALYSIS:")
            for key, value in uncertainty.items():
                clean_key = key.replace('_', ' ').title()
                formatted_parts.append(f"• {clean_key}: {value}")
    
    # Always include general reasoning if available
    reasoning = response.get("reasoning", "")
    if reasoning and not any("reasoning" in part.lower() for part in formatted_parts):
        formatted_parts.append(f"\nREASONING:\n{reasoning}")
    
    # Include confidence if available
    confidence = response.get("confidence", None)
    if confidence is not None:
        formatted_parts.append(f"\nCONFIDENCE: {confidence}")
    
    # If no structured content found, try to format the entire response
    if not formatted_parts:
        formatted_parts.append("MODEL RESPONSE:")
        for key, value in response.items():
            if key not in ["error", "raw_text"]:
                clean_key = key.replace('_', ' ').title()
                if isinstance(value, (dict, list)):
                    formatted_parts.append(f"• {clean_key}: {json.dumps(value, indent=2)}")
                else:
                    formatted_parts.append(f"• {clean_key}: {value}")
    
    return "\n".join(formatted_parts)

def format_ground_truth_for_human_evaluation(ground_truth: dict, task_type: str) -> str:
    """
    Formats ground truth data in human-readable format.
    """
    if not ground_truth or not isinstance(ground_truth, dict):
        return "No ground truth data available."
    
    formatted_parts = []
    
    if task_type == "forward_prediction":
        formatted_parts.append("EXPECTED MATERIAL PROPERTIES:")
    elif task_type == "inverse_design":
        formatted_parts.append("KNOWN SYNTHESIS CONDITIONS:")
    
    for key, value in ground_truth.items():
        clean_key = key.replace('_', ' ').title()
        if isinstance(value, dict):
            formatted_parts.append(f"• {clean_key}:")
            for sub_key, sub_value in value.items():
                clean_sub_key = sub_key.replace('_', ' ').title()
                formatted_parts.append(f"  - {clean_sub_key}: {sub_value}")
        else:
            formatted_parts.append(f"• {clean_key}: {value}")
    
    return "\n".join(formatted_parts)

def format_input_query_for_human_evaluation(input_query: dict, task_type: str) -> str:
    """
    Formats input query in human-readable format.
    """
    if not input_query or not isinstance(input_query, dict):
        return "No input query data available."
    
    formatted_parts = []
    
    if task_type == "forward_prediction":
        formatted_parts.append("GIVEN SYNTHESIS CONDITIONS:")
    elif task_type == "inverse_design":
        formatted_parts.append("DESIRED MATERIAL PROPERTIES:")
    
    for key, value in input_query.items():
        clean_key = key.replace('_', ' ').title()
        if isinstance(value, dict):
            formatted_parts.append(f"• {clean_key}:")
            for sub_key, sub_value in value.items():
                clean_sub_key = sub_key.replace('_', ' ').title()
                formatted_parts.append(f"  - {clean_sub_key}: {sub_value}")
        else:
            formatted_parts.append(f"• {clean_key}: {value}")
    
    return "\n".join(formatted_parts)

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

# --- LLM-BASED EVALUATOR ---

class LLMEvaluator:
    """Uses Gemini 2.5 Pro for enhanced evaluation with better reasoning capabilities."""
    def __init__(self, model_id: str = "gemini-2.5-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY for evaluator.")
        genai.configure(api_key=api_key)
        
        # Use Gemini 2.5 Pro for better reasoning
        self.model = genai.GenerativeModel(
            model_id,
            generation_config={"response_mime_type": "application/json"}
        )
        print(f"Initialized LLM evaluator with {model_id}")
        
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

        **Advanced Reasoning Instructions:**
        - Consider implicit scientific knowledge and unstated but reasonable assumptions
        - Evaluate the sophistication of the underlying scientific model
        - Assess whether the answer demonstrates understanding of structure-property relationships
        - Consider the appropriateness of the approach for the specific materials system
        - Evaluate awareness of experimental challenges and practical constraints

        **JSON Schema:**
        {{
            "scientific_accuracy": {{ "score": integer, "justification": "detailed_explanation" }},
            "functional_equivalence": {{ "score": integer, "justification": "detailed_explanation" }},
            "reasoning_quality": {{ "score": integer, "justification": "detailed_explanation" }},
            "completeness": {{ "score": integer, "justification": "detailed_explanation" }},
            "overall_score": {{ "score": integer, "justification": "comprehensive_assessment" }}
        }}
        """

    def evaluate_output(self, task_type: str, input_query: dict, ground_truth: dict, generated_output: dict) -> dict:
        """Performs enhanced evaluation using Gemini 2.5 Pro's superior reasoning capabilities."""
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
                
                # Add some debug information
                if attempt > 0:
                    print(f"    - Evaluator succeeded on attempt {attempt+1}")
                
                return scores
                
            except json.JSONDecodeError as e:
                print(f"    - JSON decode error on attempt {attempt+1}: {e}")
                if attempt < 2:
                    time.sleep(2)
                    continue
                else:
                    # Fallback: try to extract scores from text response
                    try:
                        # Simple regex fallback for score extraction
                        scores = {}
                        score_pattern = r'"score":\s*(\d+)'
                        matches = re.findall(score_pattern, response.text)
                        if len(matches) >= 5:
                            score_keys = ["scientific_accuracy_score", "functional_equivalence_score", 
                                        "reasoning_quality_score", "completeness_score", "overall_score_score"]
                            for i, key in enumerate(score_keys):
                                scores[key] = int(matches[i]) / 10.0
                            scores['justifications'] = {"fallback": "Extracted from malformed JSON"}
                            return scores
                    except:
                        pass
                        
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

    def format_ground_truth_narrative(self, ground_truth: dict, task_type: str) -> str:
        """Format ground truth data using the same LLM for consistency."""
        return format_ground_truth_with_llm(ground_truth, task_type, self)

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

def evaluate_multiple_models_on_test_set_with_answers(
    engines: Dict[str, Any],
    baseline_model: BaselineQwenModel,
    llm_evaluator: LLMEvaluator,
    test_experiments: List[Dict],
    output_prefix: str
) -> pd.DataFrame:
    """
    Evaluates multiple engines and Qwen baseline, saving all model answers 
    in human-readable format for expert evaluation.
    """
    results = []
    human_evaluation_data = []
    
    for i, experiment in enumerate(test_experiments):
        print(f"\nEvaluating test case {i+1}/{len(test_experiments)}")
        
        # Extract synthesis conditions and property changes
        synthesis_conditions = experiment.get("synthesis_conditions", {})
        property_changes = experiment.get("property_changes", {})
        
        # --- Forward Prediction Evaluation ---
        print("  - Task: Forward Prediction")
        
        # Collect all model responses for this test case
        forward_responses = {}
        
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
                forward_responses[engine_name] = pred_result
                
                print(f"    - Evaluating {engine_name} output with Gemini 2.5 Pro...")
                scores = llm_evaluator.evaluate_output("Forward Prediction", synthesis_conditions, 
                                                      property_changes, pred_result)
                engine_results[engine_name]['scores'] = scores
            except Exception as e:
                print(f"    Warning: {engine_name} forward prediction failed: {e}")
                error_result = {"predicted_properties": {}, "reasoning": "Error", "error": str(e)}
                engine_results[engine_name] = {
                    'result': error_result,
                    'confidence': 0.0,
                    'scores': {
                        "scientific_accuracy_score": 0.0, "functional_equivalence_score": 0.0,
                        "reasoning_quality_score": 0.0, "completeness_score": 0.0,
                        "overall_score_score": 0.0, 'justifications': {}
                    }
                }
                forward_responses[engine_name] = error_result
        
        # Evaluate Qwen baseline
        try:
            baseline_pred_result = baseline_model.forward_prediction(synthesis_conditions)
            forward_responses["qwen_baseline"] = baseline_pred_result
            
            print("    - Evaluating Qwen Baseline output with Gemini 2.5 Pro...")
            baseline_scores = llm_evaluator.evaluate_output("Forward Prediction", synthesis_conditions, 
                                                           property_changes, baseline_pred_result)
        except Exception as e:
            print(f"    Warning: Qwen Baseline forward prediction failed: {e}")
            baseline_pred_result = {"predicted_properties": {}, "reasoning": "Error", "error": str(e)}
            forward_responses["qwen_baseline"] = baseline_pred_result
            baseline_scores = {
                "scientific_accuracy_score": 0.0, "functional_equivalence_score": 0.0,
                "reasoning_quality_score": 0.0, "completeness_score": 0.0,
                "overall_score_score": 0.0, 'justifications': {}
            }

        # Save human evaluation data for forward prediction
        human_eval_entry = {
            "experiment_id": experiment.get("experiment_id", i),
            "task_type": "forward_prediction", 
            "host_material": experiment.get("host_material", "unknown"),
            "dopant_element": experiment.get("dopant", {}).get("element", "unknown"),
            "input_query": format_input_query_for_human_evaluation(synthesis_conditions, "forward_prediction"),
            "ground_truth_raw": format_ground_truth_for_human_evaluation(property_changes, "forward_prediction"),
            # Add LLM-formatted ground truth
            "ground_truth_narrative": llm_evaluator.format_ground_truth_narrative(property_changes, "forward_prediction")
        }
        # Right after creating human_eval_entry for forward prediction:
        print(f"    - Formatting ground truth with {llm_evaluator.model._model_name}...")
        human_eval_entry["ground_truth_narrative"] = llm_evaluator.format_ground_truth_narrative(
            property_changes, "forward_prediction")


        # Add formatted model responses
        human_eval_entry["qwen_baseline_response"] = format_model_response_for_human_evaluation(
            forward_responses["qwen_baseline"], "forward_prediction")
        
        for engine_name in engines.keys():
            human_eval_entry[f"{engine_name}_response"] = format_model_response_for_human_evaluation(
                forward_responses[engine_name], "forward_prediction")
        
        human_evaluation_data.append(human_eval_entry.copy())

        # Create result entry for forward prediction (for statistical analysis)
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
        
        # Collect all model responses for this test case
        inverse_responses = {}
        
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
                inverse_responses[engine_name] = inv_result
                
                print(f"    - Evaluating {engine_name} output with Gemini 2.5 Pro...")
                scores = llm_evaluator.evaluate_output("Inverse Design", property_changes, 
                                                      synthesis_conditions, inv_result)
                engine_results[engine_name]['scores'] = scores
            except Exception as e:
                print(f"    Warning: {engine_name} inverse design failed: {e}")
                error_result = {"suggested_synthesis_conditions": {}, "reasoning": "Error", "error": str(e)}
                engine_results[engine_name] = {
                    'result': error_result,
                    'confidence': 0.0,
                    'scores': {
                        "scientific_accuracy_score": 0.0, "functional_equivalence_score": 0.0,
                        "reasoning_quality_score": 0.0, "completeness_score": 0.0,
                        "overall_score_score": 0.0, 'justifications': {}
                    }
                }
                inverse_responses[engine_name] = error_result
        
        # Evaluate Qwen baseline
        try:
            baseline_inv_result = baseline_model.inverse_design(property_changes)
            inverse_responses["qwen_baseline"] = baseline_inv_result
            
            print("    - Evaluating Qwen Baseline output with Gemini 2.5 Pro...")
            baseline_scores = llm_evaluator.evaluate_output("Inverse Design", property_changes, 
                                                           synthesis_conditions, baseline_inv_result)
        except Exception as e:
            print(f"    Warning: Qwen Baseline inverse design failed: {e}")
            baseline_inv_result = {"suggested_synthesis_conditions": {}, "reasoning": "Error", "error": str(e)}
            inverse_responses["qwen_baseline"] = baseline_inv_result
            baseline_scores = {
                "scientific_accuracy_score": 0.0, "functional_equivalence_score": 0.0,
                "reasoning_quality_score": 0.0, "completeness_score": 0.0,
                "overall_score_score": 0.0, 'justifications': {}
            }

        # Save human evaluation data for inverse design
        human_eval_entry = {
            "experiment_id": experiment.get("experiment_id", i),
            "task_type": "inverse_design",
            "host_material": experiment.get("host_material", "unknown"), 
            "dopant_element": experiment.get("dopant", {}).get("element", "unknown"),
            "input_query": format_input_query_for_human_evaluation(property_changes, "inverse_design"),
            "ground_truth_raw": format_ground_truth_for_human_evaluation(synthesis_conditions, "inverse_design"),
            # Add LLM-formatted ground truth  
            "ground_truth_narrative": llm_evaluator.format_ground_truth_narrative(synthesis_conditions, "inverse_design")
        }
        # And similarly for inverse design:
        print(f"    - Formatting ground truth with {llm_evaluator.model._model_name}...")  
        human_eval_entry["ground_truth_narrative"] = llm_evaluator.format_ground_truth_narrative(
            synthesis_conditions, "inverse_design")
                    
        # Add formatted model responses
        human_eval_entry["qwen_baseline_response"] = format_model_response_for_human_evaluation(
            inverse_responses["qwen_baseline"], "inverse_design")
        
        for engine_name in engines.keys():
            human_eval_entry[f"{engine_name}_response"] = format_model_response_for_human_evaluation(
                inverse_responses[engine_name], "inverse_design")
        
        human_evaluation_data.append(human_eval_entry.copy())

        # Create result entry for inverse design (for statistical analysis)
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

    # Save human evaluation data to CSV
    human_eval_df = pd.DataFrame(human_evaluation_data)
    human_eval_csv = f"{output_prefix}_human_evaluation_responses.csv"
    human_eval_df.to_csv(human_eval_csv, index=False)
    print(f"\n💾 Human evaluation responses saved to '{human_eval_csv}'")
    
    # Create a summary CSV with key information for questionnaire
    summary_data = []
    for _, row in human_eval_df.iterrows():
        summary_entry = {
            "Test_Case_ID": f"{row['experiment_id']}_{row['task_type']}",
            "Task": row['task_type'].replace('_', ' ').title(),
            "Material_System": f"{row['host_material']} + {row['dopant_element']}",
            "Problem_Statement": row['input_query'],
            "Expected_Answer_Raw": row['ground_truth_raw'],
            "Expected_Answer_Narrative": row['ground_truth_narrative'],  # New LLM-formatted version
            "Qwen_Baseline_Answer": row['qwen_baseline_response']
        }
    
    # Add DAG engine answers
    for engine_name in engines.keys():
        summary_entry[f"{engine_name}_Answer"] = row[f"{engine_name}_response"]
    
    summary_data.append(summary_entry)
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = f"{output_prefix}_expert_questionnaire.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"💾 Expert questionnaire CSV saved to '{summary_csv}'")

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
    fig.suptitle('Multi-Model Evaluation: DAG-Enhanced Engines vs Qwen Baseline\n(Evaluated by Gemini 2.5 Pro)', 
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
    ax.set_title("Overall Model Performance\n(Gemini 2.5 Pro Evaluation)")
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
    ax.set_title(f"Head-to-Head: {best_engine } vs Qwen")
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
    parser = argparse.ArgumentParser(description="Evaluate multiple causal reasoning engines against Qwen baseline")
    parser.add_argument("--test_data", default="outputs/test_doping_data.json", 
                       help="Path to test data JSON file")
    parser.add_argument("--training_graph", default="outputs/combined_doping_data.json",
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

def format_ground_truth_with_llm(ground_truth: dict, task_type: str, llm_evaluator) -> str:
    """
    Uses the same LLM (Gemini) to format ground truth data in narrative style
    to match the formatting of model responses for fair comparison.
    """
    if not ground_truth or not isinstance(ground_truth, dict):
        return "No ground truth data available."
    
    if task_type == "forward_prediction":
        prompt = f"""
        You are formatting experimental results for expert evaluation. Convert the following 
        material property changes (ground truth data) into a clear, narrative format that 
        matches how a materials scientist would describe predicted properties.
        
        Property Changes Data:
        {json.dumps(ground_truth, indent=2)}
        
        Format this as a materials scientist would describe PREDICTED PROPERTIES, using the same 
        style and structure as model predictions. Include mechanistic explanations where the 
        data suggests underlying mechanisms.
        
        Use this structure:
        - Start with "PREDICTED PROPERTIES:" section
        - Add "MECHANISTIC EXPLANATION:" if mechanisms can be inferred
        - Use bullet points with clear property names and values
        - Write in professional materials science language
        """
    
    elif task_type == "inverse_design":
        prompt = f"""
        You are formatting experimental data for expert evaluation. Convert the following 
        synthesis conditions (ground truth data) into a clear, narrative format that 
        matches how a materials scientist would describe suggested synthesis approaches.
        
        Synthesis Conditions Data:
        {json.dumps(ground_truth, indent=2)}
        
        Format this as a materials scientist would describe SUGGESTED SYNTHESIS CONDITIONS, 
        using the same style and structure as model predictions. Include reasoning for 
        why these conditions would work.
        
        Use this structure:
        - Start with "SUGGESTED SYNTHESIS CONDITIONS:" section  
        - Add "MECHANISTIC REASONING:" section explaining why these conditions work
        - Use bullet points with clear parameter names and values
        - Write in professional materials science language
        """
    
    try:
        # Use the same model as the evaluator for consistency
        response = llm_evaluator.model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"    Warning: Failed to format ground truth with LLM: {e}")
        # Fallback to the original formatting function
        return format_ground_truth_for_human_evaluation(ground_truth, task_type)

if __name__ == '__main__':
    args = parse_arguments()
    
    print("="*80)
    print("MULTI-MODEL EVALUATION WITH HUMAN-READABLE ANSWER EXPORT")
    print("="*80)
    print(f"Qwen Baseline Model: {args.qwen_model}")
    print(f"Gemini Evaluator Model: {args.evaluator_model}")
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
        output_prefix = get_file_prefix(args.test_data) + "_qwen_baseline_gemini_eval"
    
    print(f"\nUsing test data file: {args.test_data}")
    print(f"Output files will use prefix: {output_prefix}")
    print(f"Evaluating {len(engines)} engines: {', '.join(engine_names)}")
    
    # Initialize Qwen baseline and Gemini evaluator
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
        llm_evaluator = LLMEvaluator(model_id=args.evaluator_model)
        print("✅ Gemini 2.5 Pro evaluator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini evaluator: {e}")
        sys.exit(1)
    
    # Load test data
    print(f"\n📊 Loading test data from {args.test_data}...")
    test_experiments = load_test_data(args.test_data)
    print(f"✅ Loaded {len(test_experiments)} test experiments")
    
    # Run evaluation with answer saving
    print(f"\n🔬 Evaluating models on {len(test_experiments)} test cases...")
    print("This may take several minutes due to enhanced evaluation and answer formatting...")
    
    results_df = evaluate_multiple_models_on_test_set_with_answers(
        engines, baseline_model, llm_evaluator, test_experiments, output_prefix
    )
    
    # Save detailed results
    results_csv = f"{output_prefix}_evaluation_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n💾 Detailed results saved to '{results_csv}'")
    
    # Generate summary statistics
    overall_stats, task_stats = generate_multi_model_summary_statistics(results_df, engine_names)
    
    # Save summary statistics
    summary_json = f"{output_prefix}_summary_statistics.json"
    with open(summary_json, "w") as f:
        json.dump(overall_stats, f, indent=2) 
    print(f"💾 Summary statistics saved to '{summary_json}'")
    
    # Create visualizations
    print("\n📈 Generating visualizations...")
    plot_multi_model_results(results_df, engine_names, output_prefix)
    
    # Print key findings
    print("\n" + "="*80)
    print("EVALUATION RESULTS: DAG-ENHANCED ENGINES vs QWEN BASELINE")
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
        status = "✅" if improvement > 0 else "❌"
        print(f"  {status} {engine_name}: {improvement:+.3f} (Win Rate: {win_rate:.1f}%)")
    
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
    
    print(f"\n📁 Files Generated:")
    print(f"  • Statistical results: {output_prefix}_evaluation_results.csv")
    print(f"  • Human evaluation data: {output_prefix}_human_evaluation_responses.csv")
    print(f"  • Expert questionnaire: {output_prefix}_expert_questionnaire.csv")
    print(f"  • Summary statistics: {output_prefix}_summary_statistics.json")
    print(f"  • Visualization: {output_prefix}_multi_model_evaluation_results.png")
    
    print("\n🔍 The expert questionnaire CSV contains human-readable model responses")
    print("    formatted for direct expert evaluation and comparison.")
    print("="*80)