#!/usr/bin/env python3
"""
Script to evaluate filtered_evaluation_responses.csv using LLMEvaluator.
This script extracts causal engine results (DAG-Basic and DAG-CoT) and evaluates them
across 5 dimensions using the LLMEvaluator class.
output: a csv file with evaluation results
"""

import pandas as pd
import json
import os
import sys
from typing import Dict, List, Any
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.project.evaluation.evaluation_multi import LLMEvaluator


def parse_response_field(response_text: str) -> Dict[str, Any]:
    """
    Parse response field from CSV. Handles both JSON strings and plain text.
    
    Args:
        response_text: The response text from CSV
        
    Returns:
        Parsed response as dictionary
    """
    if pd.isna(response_text) or response_text.strip() == "":
        return {}
    
    # Try to parse as JSON first
    try:
        return json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        # If not JSON, return as plain text
        return {"response": response_text.strip()}


def parse_ground_truth(ground_truth_raw: str, ground_truth_narrative: str) -> Dict[str, Any]:
    """
    Parse ground truth from CSV fields.
    
    Args:
        ground_truth_raw: Raw ground truth data
        ground_truth_narrative: Narrative description
        
    Returns:
        Combined ground truth dictionary
    """
    result = {}
    
    # Parse raw ground truth
    if not pd.isna(ground_truth_raw) and ground_truth_raw.strip():
        try:
            result.update(json.loads(ground_truth_raw))
        except (json.JSONDecodeError, TypeError):
            result["raw"] = ground_truth_raw.strip()
    
    # Add narrative
    if not pd.isna(ground_truth_narrative) and ground_truth_narrative.strip():
        result["narrative"] = ground_truth_narrative.strip()
    
    return result


def parse_input_query(input_query: str) -> Dict[str, Any]:
    """
    Parse input query from CSV.
    
    Args:
        input_query: Input query string
        
    Returns:
        Parsed input query dictionary
    """
    if pd.isna(input_query) or input_query.strip() == "":
        return {}
    
    try:
        return json.loads(input_query)
    except (json.JSONDecodeError, TypeError):
        return {"query": input_query.strip()}


def evaluate_causal_engine_responses(csv_file: str, output_file: str = None) -> pd.DataFrame:
    """
    Evaluate causal engine responses from the CSV file.
    
    Args:
        csv_file: Path to the filtered_evaluation_responses.csv file
        output_file: Optional output file path for results
        
    Returns:
        DataFrame with evaluation results compatible with plot_multi_model_results
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} experiments")
    
    # Initialize evaluator
    print("Initializing LLMEvaluator...")
    evaluator = LLMEvaluator()
    
    # Prepare results storage - format compatible with evaluation_multi.py
    results = []
    
    # Models to evaluate: baseline + causal engines
    models_to_evaluate = {
        'qwen_baseline': 'qwen_baseline_response',
        'DAG-Basic': 'DAG-Basic_response', 
        'DAG-CoT': 'DAG-CoT_response'
    }
    
    for idx, row in df.iterrows():
        print(f"\nProcessing experiment {idx + 1}/{len(df)}: {row['experiment_id']}")
        
        # Parse common fields
        experiment_id = row['experiment_id']
        task_type = row['task_type']
        
        # Parse input query
        input_query = parse_input_query(row['input_query'])
        
        # Parse ground truth
        ground_truth = parse_ground_truth(row['ground_truth_raw'], row['ground_truth_narrative'])
        
        # Store evaluations for this experiment
        experiment_scores = {
            'task': task_type,
            'experiment_id': experiment_id,
            'host_material': row['host_material'],
            'dopant': row['dopant_element']
        }
        
        # Evaluate each model
        for model_name, response_column in models_to_evaluate.items():
            if response_column not in row or pd.isna(row[response_column]):
                print(f"  - Skipping {model_name}: No response found")
                # Add zero scores for missing models
                score_dims = ['scientific_accuracy', 'functional_equivalence', 'reasoning_quality', 
                             'completeness', 'overall_score']
                for dim in score_dims:
                    experiment_scores[f'{model_name}_{dim}_score'] = 0.0
                continue
                
            print(f"  - Evaluating {model_name}...")
            
            # Parse model response
            model_response = parse_response_field(row[response_column])
            
            # Skip empty responses
            if not model_response:
                print(f"    - Empty response, using zero scores")
                score_dims = ['scientific_accuracy', 'functional_equivalence', 'reasoning_quality', 
                             'completeness', 'overall_score']
                for dim in score_dims:
                    experiment_scores[f'{model_name}_{dim}_score'] = 0.0
                continue
            
            # Evaluate using LLMEvaluator
            evaluation_scores = evaluator.evaluate_output(
                task_type=task_type,
                input_query=input_query,
                ground_truth=ground_truth,
                generated_output=model_response
            )
            
            # Store scores in compatible format
            for score_key, score_value in evaluation_scores.items():
                if score_key != 'justifications':
                    experiment_scores[f'{model_name}_{score_key}'] = score_value
            
            # Print scores
            print(f"    - Scientific Accuracy: {evaluation_scores['scientific_accuracy_score']:.3f}")
            print(f"    - Functional Equivalence: {evaluation_scores['functional_equivalence_score']:.3f}")
            print(f"    - Reasoning Quality: {evaluation_scores['reasoning_quality_score']:.3f}")
            print(f"    - Completeness: {evaluation_scores['completeness_score']:.3f}")
            print(f"    - Overall Score: {evaluation_scores['overall_score_score']:.3f}")
        
        # Calculate improvements over baseline for causal engines
        if 'qwen_baseline_overall_score_score' in experiment_scores:
            baseline_score = experiment_scores['qwen_baseline_overall_score_score']
            for engine_name in ['DAG-Basic', 'DAG-CoT']:
                if f'{engine_name}_overall_score_score' in experiment_scores:
                    engine_score = experiment_scores[f'{engine_name}_overall_score_score']
                    experiment_scores[f'{engine_name}_improvement'] = engine_score - baseline_score
                    # Add confidence score (placeholder - you can modify this logic)
                    experiment_scores[f'{engine_name}_confidence'] = min(1.0, engine_score + 0.1)
        
        results.append(experiment_scores)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    if output_file is None:
        output_file = "filtered_responses_evaluation_results.csv"
    
    results_df.to_csv(output_file, index=False)
    print(f"\nEvaluation results saved to {output_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    all_models = ['qwen_baseline', 'DAG-Basic', 'DAG-CoT']
    
    for model_name in all_models:
        score_cols = [f'{model_name}_{dim}_score' for dim in ['scientific_accuracy', 'functional_equivalence', 
                                                             'reasoning_quality', 'completeness', 'overall_score']]
        
        if all(col in results_df.columns for col in score_cols):
            print(f"\n{model_name.replace('_', ' ').title()} Performance:")
            print(f"  - Experiments evaluated: {len(results_df)}")
            print(f"  - Scientific Accuracy: {results_df[f'{model_name}_scientific_accuracy_score'].mean():.3f} ± {results_df[f'{model_name}_scientific_accuracy_score'].std():.3f}")
            print(f"  - Functional Equivalence: {results_df[f'{model_name}_functional_equivalence_score'].mean():.3f} ± {results_df[f'{model_name}_functional_equivalence_score'].std():.3f}")
            print(f"  - Reasoning Quality: {results_df[f'{model_name}_reasoning_quality_score'].mean():.3f} ± {results_df[f'{model_name}_reasoning_quality_score'].std():.3f}")
            print(f"  - Completeness: {results_df[f'{model_name}_completeness_score'].mean():.3f} ± {results_df[f'{model_name}_completeness_score'].std():.3f}")
            print(f"  - Overall Score: {results_df[f'{model_name}_overall_score_score'].mean():.3f} ± {results_df[f'{model_name}_overall_score_score'].std():.3f}")
    
    # Improvement over baseline
    print(f"\nImprovement over Baseline:")
    for engine_name in ['DAG-Basic', 'DAG-CoT']:
        if f'{engine_name}_improvement' in results_df.columns:
            avg_improvement = results_df[f'{engine_name}_improvement'].mean()
            print(f"  {engine_name}: {avg_improvement:+.3f}")
    
    # Task-specific performance
    print(f"\nPerformance by Task Type:")
    for task_type in results_df['task'].unique():
        task_results = results_df[results_df['task'] == task_type]
        print(f"\n  {task_type.replace('_', ' ').title()}:")
        for model_name in all_models:
            if f'{model_name}_overall_score_score' in task_results.columns:
                avg_score = task_results[f'{model_name}_overall_score_score'].mean()
                print(f"    {model_name.replace('_', ' ').title()}: {avg_score:.3f} ({len(task_results)} experiments)")
    
    return results_df


def generate_visualizations(results_df: pd.DataFrame, output_prefix: str = "filtered_responses"):
    """
    Generate visualizations using the same plot_multi_model_results function from evaluation_multi.py
    
    Args:
        results_df: DataFrame with evaluation results
        output_prefix: Prefix for output files
    """
    try:
        # Import the plotting function
        sys.path.append('/home/lwang240/Projects/LLM4Chem-Explainable-synthesis/script')
        from evaluation_multi import plot_multi_model_results
        
        # Engine names for plotting (excluding baseline)
        engine_names = ['DAG-Basic', 'DAG-CoT']   
        
        print(f"\nGenerating visualizations with prefix: {output_prefix}")
        plot_multi_model_results(results_df, engine_names, output_prefix)
        print(f"Visualizations saved with prefix: {output_prefix}")  
        
    except ImportError as e:
        print(f"Warning: Could not import plotting functions: {e}")
        print("Skipping visualization generation")
    except Exception as e:
        print(f"Warning: Error generating visualizations: {e}")
        print("Skipping visualization generation")


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate filtered causal engine responses")
    parser.add_argument("--input", "-i", 
                       default="/home/lwang240/Projects/LLM4Chem-Explainable-synthesis/filtered_evaluation_responses.csv",
                       help="Path to input CSV file")
    parser.add_argument("--output", "-o", 
                       default=None,
                       help="Path to output CSV file")
    parser.add_argument("--limit", "-l", 
                       type=int, 
                       default=None,
                       help="Limit number of experiments to process (for testing)")
    parser.add_argument("--no-plots", 
                       action="store_true",
                       help="Skip generating visualizations")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        sys.exit(1)
    
    # Load and optionally limit data
    if args.limit:
        print(f"Loading first {args.limit} experiments for testing...")
        df = pd.read_csv(args.input).head(args.limit)
        df.to_csv("temp_limited_data.csv", index=False)
        input_file = "temp_limited_data.csv"
    else:
        input_file = args.input
    
    # # Run evaluation
    # results_df = evaluate_causal_engine_responses(input_file, args.output)
    
    # # Generate visualizations
    with open("filtered_responses_evaluation_results.csv", "r") as f:
        results_df = pd.read_csv(f)
    if not args.no_plots:
        output_prefix = args.output.replace('.csv', '') if args.output else 'filtered_responses_evaluation'
        generate_visualizations(results_df, output_prefix)
    
    # # Cleanup temporary file
    # if args.limit and os.path.exists("temp_limited_data.csv"):
    #     os.remove("temp_limited_data.csv")
    
    # print(f"\nEvaluation completed successfully!")
    # print(f"Results saved to: {args.output or 'filtered_responses_evaluation_results.csv'}")



if __name__ == "__main__":
    main()
