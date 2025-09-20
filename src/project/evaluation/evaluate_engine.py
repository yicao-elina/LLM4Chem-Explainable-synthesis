import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import importlib
import argparse
import time
import sys
import os
from omegaconf import OmegaConf

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

def load_engine_from_module(module_path: str, training_graph_file: str):
    """Dynamically load CausalReasoningEngine from a module file path."""
    try:
        # Convert file path to module name
        module_path = Path(module_path)
        if not module_path.exists():
            raise FileNotFoundError(f"Engine module not found: {module_path}")
        
        # Add the module's directory to sys.path if not already there
        module_dir = str(module_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        
        # Import the module by name
        module_name = module_path.stem
        module = importlib.import_module(module_name)
        
        # Get the engine class
        engine_class = getattr(module, 'CausalReasoningEngine')
        return engine_class(training_graph_file)
    except Exception as e:
        print(f"Error loading engine from {module_path}: {e}")
        return None

# --- DATA LOADING ---

def load_data(test_file: str) -> List[Dict]:
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

# --- ENGINE EVALUATION WORKFLOW ---

def run_engines_on_dataset(
    engines: Dict[str, Any],
    test_experiments: List[Dict],
    result_paths: Dict[str, str]
) -> Dict[str, pd.DataFrame]:
    """Run multiple engines on test set and save results separately."""
    
    all_results = {}
    
    for engine_name, engine in engines.items():
        print(f"\n{'='*50}")
        print(f"Running Engine: {engine_name}")
        print(f"{'='*50}")
        
        engine_results = []
        
        for i, experiment in enumerate(test_experiments):
            print(f"\nProcessing test case {i+1}/{len(test_experiments)} for {engine_name}")
            
            # Extract synthesis conditions and property changes
            synthesis_conditions = experiment.get("synthesis_conditions", {})
            property_changes = experiment.get("property_changes", {})
            
            # --- Forward Prediction ---
            print("  - Task: Forward Prediction")
            try:
                start_time = time.time()
                forward_result = engine.forward_prediction(synthesis_conditions)
                forward_time = time.time() - start_time
                
                forward_confidence = forward_result.get("confidence", 0.0)
                
                # Create result entry for forward prediction
                result_entry = {
                    "task": "forward_prediction",
                    "experiment_id": experiment.get("experiment_id", i),
                    "host_material": experiment.get("host_material", "unknown"),
                    "dopant": experiment.get("dopant", {}).get("element", "unknown"),
                    "synthesis_conditions": json.dumps(synthesis_conditions),
                    "ground_truth_properties": json.dumps(property_changes),
                    "predicted_properties": json.dumps(forward_result.get("predicted_properties", {})),
                    "reasoning": forward_result.get("reasoning", ""),
                    "confidence": forward_confidence,
                    "execution_time": forward_time,
                    "raw_response": json.dumps(forward_result)
                }
                
                engine_results.append(result_entry)
                print(f"    - Completed in {forward_time:.2f}s, confidence: {forward_confidence:.3f}")
                
            except Exception as e:
                print(f"    - Error in forward prediction: {e}")
                result_entry = {
                    "task": "forward_prediction",
                    "experiment_id": experiment.get("experiment_id", i),
                    "host_material": experiment.get("host_material", "unknown"),
                    "dopant": experiment.get("dopant", {}).get("element", "unknown"),
                    "synthesis_conditions": json.dumps(synthesis_conditions),
                    "ground_truth_properties": json.dumps(property_changes),
                    "predicted_properties": json.dumps({}),
                    "reasoning": f"Error: {str(e)}",
                    "confidence": 0.0,
                    "execution_time": 0.0,
                    "raw_response": json.dumps({"error": str(e)})
                }
                engine_results.append(result_entry)
            
            # --- Inverse Design ---
            print("  - Task: Inverse Design")
            try:
                start_time = time.time()
                inverse_result = engine.inverse_design(property_changes)
                inverse_time = time.time() - start_time
                
                inverse_confidence = inverse_result.get("confidence", 0.0)
                
                # Create result entry for inverse design
                result_entry = {
                    "task": "inverse_design",
                    "experiment_id": experiment.get("experiment_id", i),
                    "host_material": experiment.get("host_material", "unknown"),
                    "dopant": experiment.get("dopant", {}).get("element", "unknown"),
                    "desired_properties": json.dumps(property_changes),
                    "ground_truth_conditions": json.dumps(synthesis_conditions),
                    "suggested_conditions": json.dumps(inverse_result.get("suggested_synthesis_conditions", {})),
                    "reasoning": inverse_result.get("reasoning", ""),
                    "confidence": inverse_confidence,
                    "execution_time": inverse_time,
                    "raw_response": json.dumps(inverse_result)
                }
                
                engine_results.append(result_entry)
                print(f"    - Completed in {inverse_time:.2f}s, confidence: {inverse_confidence:.3f}")
                
            except Exception as e:
                print(f"    - Error in inverse design: {e}")
                result_entry = {
                    "task": "inverse_design",
                    "experiment_id": experiment.get("experiment_id", i),
                    "host_material": experiment.get("host_material", "unknown"),
                    "dopant": experiment.get("dopant", {}).get("element", "unknown"),
                    "desired_properties": json.dumps(property_changes),
                    "ground_truth_conditions": json.dumps(synthesis_conditions),
                    "suggested_conditions": json.dumps({}),
                    "reasoning": f"Error: {str(e)}",
                    "confidence": 0.0,
                    "execution_time": 0.0,
                    "raw_response": json.dumps({"error": str(e)})
                }
                engine_results.append(result_entry)
        
        # Convert to DataFrame and save
        engine_df = pd.DataFrame(engine_results)
        all_results[engine_name] = engine_df
        
        # Save results for this engine
        output_path = result_paths.get(engine_name, f"outputs/engine_results_{engine_name}.csv")
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        engine_df.to_csv(output_path, index=False)
        print(f"\nResults for {engine_name} saved to: {output_path}")
        
        # Print summary statistics
        forward_results = engine_df[engine_df['task'] == 'forward_prediction']
        inverse_results = engine_df[engine_df['task'] == 'inverse_design']
        
        print(f"\nSummary for {engine_name}:")
        print(f"  Forward Prediction - Avg Confidence: {forward_results['confidence'].mean():.3f}")
        print(f"  Inverse Design - Avg Confidence: {inverse_results['confidence'].mean():.3f}")
        print(f"  Total Execution Time: {engine_df['execution_time'].sum():.2f}s")
    
    return all_results

# --- MAIN EXECUTION ---

def main():
    parser = argparse.ArgumentParser(description="Run engines on test data and save results")
    parser.add_argument("--config", default="config/eval_engine_filtered.yaml",
                       help="Path to configuration YAML file")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of test cases to process")
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = OmegaConf.load(args.config)
    
    # Get data path using OmegaConf
    for data_type in config.data_path:
        data_path = config.data_path.get(data_type)
        print(f"\nLoading {data_type} data from: {data_path}")
        experiments = load_data(data_path)
        if args.limit:
            experiments = experiments[:args.limit]
        print(f"Loaded {len(experiments)} {data_type} experiments")

        engine_paths = config.engine_path
        result_paths = config.result_path
        
        print(f"\nLoading engines...")
        engines = {}
        
        for method_name, engine_path in engine_paths.items():
            print(f"  Loading {method_name} from {engine_path}...")
            engine = load_engine_from_module(engine_path, data_path)
            if engine is None:
                print(f"  Failed to load {method_name} from {engine_path}")
                continue
            engines[method_name] = engine     
            print(f"\nSuccessfully loaded {len(engines)} engines: {list(engines.keys())}")
            print(f"\nRunning engines on {len(experiments)} {data_type} cases...")
            all_results = run_engines_on_dataset(engines, experiments, result_paths[method_name])
        
        # Final summary
        print(f"\n{'='*80}")
        print("ENGINE EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Processed {len(experiments)} {data_type} cases")
        print(f"Evaluated {len(engines)} engines")
        print("\nOutput files:")
        for method_name in engines.keys():
            output_path = result_paths.get(method_name, f"outputs/engine_results_{method_name}.csv")
            print(f"  {method_name}: {output_path}")
        
        print(f"\nResults are ready for LLM evaluation!")

if __name__ == "__main__":
    main()
