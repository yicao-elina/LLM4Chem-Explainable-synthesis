"""
ARIA Phase I Evaluation Runner

Reproduces main_table.tex results using 6 variants on in-domain and out-of-domain data.

Systems evaluated:
1. BASELINE (Baseline LLM)
2. NAIVE_KG (Naive KG+LLM)
3. ARIA_SEARCH (Online KG+LLM)
4. ARIA_FULL (ARIA)

Tasks:
- Forward Prediction
- Inverse Design

Domains:
- In-Domain (materials in KG)
- Out-of-Domain (novel materials not in KG)

Author: ARIA Team
Date: 2026-02-02
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer
import sys
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import variants
from src.variants.baseline_ollama import BaselineOllama
from src.variants.naive_kg_ollama import NaiveKGOllama
from src.variants.aria_search_ollama import ARIASearchOllama
from src.variants.aria_full_ollama import ARIAFullOllama

# Import metrics
from src.evaluation.metrics import evaluate_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase1Evaluator:
    """
    Comprehensive evaluator for Phase I metric reproduction.
    """

    def __init__(self, kg_file_id: str, kg_file_ood: str):
        """
        Initialize evaluator with data paths.

        Args:
            kg_file_id: Path to in-domain KG
            kg_file_ood: Path to out-of-domain KG
        """
        self.kg_file_id = kg_file_id
        self.kg_file_ood = kg_file_ood

        # Load embedding model
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize systems
        logger.info("Initializing systems...")
        self.systems = self._initialize_systems()

        # Load test data
        logger.info("Loading test data...")
        self.test_data_id = self._load_test_data(kg_file_id, 'in-domain')
        self.test_data_ood = self._load_test_data(kg_file_ood, 'out-of-domain')

        logger.info(f"Loaded {len(self.test_data_id)} in-domain test cases")
        logger.info(f"Loaded {len(self.test_data_ood)} out-of-domain test cases")

    def _initialize_systems(self) -> Dict:
        """Initialize all 4 systems for evaluation."""
        systems = {}

        try:
            systems['Baseline LLM'] = BaselineOllama(model="qwen2:7b")
            logger.info("  ✅ Baseline LLM initialized")
        except Exception as e:
            logger.error(f"  ❌ Baseline LLM failed: {e}")
            systems['Baseline LLM'] = None

        try:
            systems['Naive KG+LLM'] = NaiveKGOllama(
                kg_file=self.kg_file_id,
                model="qwen2:7b"
            )
            logger.info("  ✅ Naive KG+LLM initialized")
        except Exception as e:
            logger.error(f"  ❌ Naive KG+LLM failed: {e}")
            systems['Naive KG+LLM'] = None

        try:
            systems['Online KG+LLM'] = ARIASearchOllama(
                kg_file=self.kg_file_id,
                model="qwen2:7b"
            )
            logger.info("  ✅ Online KG+LLM initialized")
        except Exception as e:
            logger.error(f"  ❌ Online KG+LLM failed: {e}")
            systems['Online KG+LLM'] = None

        try:
            systems['ARIA'] = ARIAFullOllama(
                kg_file=self.kg_file_id,
                model="qwen2:7b"
            )
            logger.info("  ✅ ARIA initialized")
        except Exception as e:
            logger.error(f"  ❌ ARIA failed: {e}")
            systems['ARIA'] = None

        return systems

    def _load_test_data(self, kg_file: str, domain: str) -> List[Dict]:
        """
        Load and prepare test cases from KG file.

        Args:
            kg_file: Path to KG JSON file
            domain: 'in-domain' or 'out-of-domain'

        Returns:
            List of test cases
        """
        with open(kg_file, 'r') as f:
            data = json.load(f)

        # Handle different KG formats
        if isinstance(data, dict) and 'causal_relationships' in data:
            relationships = data['causal_relationships']
        elif isinstance(data, list):
            relationships = data
        else:
            logger.warning(f"Unexpected data format in {kg_file}")
            return []

        test_cases = []

        # Limit to first 10 for efficiency
        for i, rel in enumerate(relationships[:10]):
            if not isinstance(rel, dict):
                continue

            # Extract synthesis and property information
            synthesis = {
                'cause_parameter': rel.get('cause_parameter', ''),
                'method': rel.get('method', ''),
                'temperature': rel.get('temperature', '')
            }

            properties = {
                'effect_on_doping': rel.get('effect_on_doping', ''),
                'affected_property': rel.get('affected_property', ''),
                'property_changes': rel.get('property_changes', {})
            }

            test_cases.append({
                'id': f"{domain}_{i}",
                'domain': domain,
                'synthesis_conditions': synthesis,
                'properties': properties,
                'ground_truth_synthesis': synthesis,
                'ground_truth_properties': properties
            })

        return test_cases

    def evaluate_system(self, system_name: str, system: Any, test_cases: List[Dict],
                       task: str) -> List[Dict]:
        """
        Evaluate a single system on test cases.

        Args:
            system_name: Name of the system
            system: System instance
            test_cases: List of test cases
            task: 'forward' or 'inverse'

        Returns:
            List of result dictionaries
        """
        if system is None:
            logger.warning(f"Skipping {system_name} - not initialized")
            return []

        results = []

        for i, test_case in enumerate(test_cases):
            logger.info(f"  [{i+1}/{len(test_cases)}] {system_name} - {task} - {test_case['domain']}")

            try:
                if task == 'forward':
                    # Forward prediction: synthesis → properties
                    prediction = system.forward_prediction(test_case['synthesis_conditions'])
                    ground_truth = test_case['ground_truth_properties']
                else:
                    # Inverse design: properties → synthesis
                    prediction = system.inverse_design(test_case['properties'])
                    ground_truth = test_case['ground_truth_synthesis']

                # Evaluate prediction
                scores = evaluate_prediction(prediction, ground_truth, self.embedding_model)

                # Add metadata
                result = {
                    'system': system_name,
                    'domain': test_case['domain'],
                    'task': task,
                    'test_id': test_case['id'],
                    **scores
                }

                results.append(result)

            except Exception as e:
                logger.error(f"    ❌ Error: {e}")
                # Add failed result with zeros
                results.append({
                    'system': system_name,
                    'domain': test_case['domain'],
                    'task': task,
                    'test_id': test_case['id'],
                    'scientific_accuracy': 0.0,
                    'functional_equivalence': 0.0,
                    'reasoning_quality': 0.0,
                    'completeness': 0.0,
                    'interpretability': 0.0,
                    'overall': 0.0,
                    'error': str(e)
                })

        return results

    def run_full_evaluation(self) -> pd.DataFrame:
        """
        Run complete evaluation on all systems, tasks, and domains.

        Returns:
            DataFrame with all results
        """
        all_results = []

        # Evaluate on both domains
        for domain_name, test_cases in [
            ('in-domain', self.test_data_id),
            ('out-of-domain', self.test_data_ood)
        ]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating on {domain_name.upper()} data")
            logger.info(f"{'='*60}")

            # Evaluate both tasks
            for task in ['forward', 'inverse']:
                logger.info(f"\nTask: {task.upper()}")

                # Evaluate all systems
                for system_name, system in self.systems.items():
                    results = self.evaluate_system(
                        system_name, system, test_cases, task
                    )
                    all_results.extend(results)

        return pd.DataFrame(all_results)

    def generate_summary_table(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary table matching main_table.tex format.

        Args:
            results_df: Raw evaluation results

        Returns:
            Summary DataFrame
        """
        # Group by system, domain, task and average metrics
        summary = results_df.groupby(['system', 'domain', 'task']).agg({
            'scientific_accuracy': 'mean',
            'functional_equivalence': 'mean',
            'reasoning_quality': 'mean',
            'completeness': 'mean',
            'interpretability': 'mean',
            'overall': 'mean'
        }).reset_index()

        # Round to 2 decimal places
        for col in ['scientific_accuracy', 'functional_equivalence', 'reasoning_quality',
                    'completeness', 'interpretability', 'overall']:
            summary[col] = summary[col].round(2)

        return summary

    def calculate_domain_gaps(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate domain gap (ID - OOD) for each system and task.

        Args:
            summary_df: Summary table

        Returns:
            DataFrame with domain gaps
        """
        gaps = []

        for system in summary_df['system'].unique():
            for task in summary_df['task'].unique():
                id_data = summary_df[
                    (summary_df['system'] == system) &
                    (summary_df['task'] == task) &
                    (summary_df['domain'] == 'in-domain')
                ]

                ood_data = summary_df[
                    (summary_df['system'] == system) &
                    (summary_df['task'] == task) &
                    (summary_df['domain'] == 'out-of-domain')
                ]

                if not id_data.empty and not ood_data.empty:
                    gap = {
                        'system': system,
                        'task': task,
                        'scientific_accuracy_gap': (id_data['scientific_accuracy'].values[0] -
                                                   ood_data['scientific_accuracy'].values[0]) * 100,
                        'functional_equivalence_gap': (id_data['functional_equivalence'].values[0] -
                                                      ood_data['functional_equivalence'].values[0]) * 100,
                        'reasoning_quality_gap': (id_data['reasoning_quality'].values[0] -
                                                 ood_data['reasoning_quality'].values[0]) * 100,
                        'completeness_gap': (id_data['completeness'].values[0] -
                                            ood_data['completeness'].values[0]) * 100,
                        'interpretability_gap': (id_data['interpretability'].values[0] -
                                                ood_data['interpretability'].values[0]) * 100,
                        'overall_gap': (id_data['overall'].values[0] -
                                       ood_data['overall'].values[0]) * 100
                    }
                    gaps.append(gap)

        return pd.DataFrame(gaps)


def generate_latex_table(summary_df: pd.DataFrame, output_file: str):
    """
    Generate LaTeX table matching main_table.tex format.

    Args:
        summary_df: Summary table with results
        output_file: Output file path
    """
    # This is a simplified version - full implementation would match exact format
    latex = []
    latex.append("\\begin{table*}[ht!]")
    latex.append("\\centering")
    latex.append("\\caption{\\textbf{Reproduced In-domain vs. out-of-domain performance analysis.}}")
    latex.append("\\label{tab:reproduced_results}")
    latex.append("\\begin{tabular}{lccccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{System} & \\textbf{Domain} & \\textbf{Scientific} & \\textbf{Functional} & \\textbf{Reasoning} & \\textbf{Completeness} & \\textbf{Interpretability} & \\textbf{Overall} \\\\")
    latex.append("\\midrule")

    # Add data rows (simplified)
    for task in ['forward', 'inverse']:
        latex.append(f"\\multicolumn{{8}}{{c}}{{\\textbf{{{task.title()} Prediction}}}} \\\\")
        latex.append("\\midrule")

        for system in summary_df['system'].unique():
            task_data = summary_df[
                (summary_df['system'] == system) &
                (summary_df['task'] == task)
            ].sort_values('domain')

            for _, row in task_data.iterrows():
                latex.append(f"{system} & {row['domain']} & "
                           f"{row['scientific_accuracy']:.2f} & "
                           f"{row['functional_equivalence']:.2f} & "
                           f"{row['reasoning_quality']:.2f} & "
                           f"{row['completeness']:.2f} & "
                           f"{row['interpretability']:.2f} & "
                           f"{row['overall']:.2f} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table*}")

    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))

    logger.info(f"LaTeX table saved to: {output_file}")


if __name__ == '__main__':
    # Paths
    kg_file_id = "data/KG/outputs/combined_doping_data.json"
    kg_file_ood = "data/KG/outputs/test_doping_data.json"

    # Check if files exist
    if not Path(kg_file_id).exists():
        print(f"❌ Error: In-domain KG file not found: {kg_file_id}")
        sys.exit(1)

    if not Path(kg_file_ood).exists():
        print(f"⚠️  Warning: Out-of-domain KG file not found: {kg_file_ood}")
        print(f"    Using in-domain data for both domains as fallback")
        kg_file_ood = kg_file_id

    # Create output directory
    output_dir = Path("results/phase2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    print("\n" + "="*80)
    print("ARIA PHASE I EVALUATION")
    print("="*80)

    evaluator = Phase1Evaluator(kg_file_id, kg_file_ood)

    print("\nRunning comprehensive evaluation...")
    results_df = evaluator.run_full_evaluation()

    # Save raw results
    results_file = output_dir / "phase1_raw_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ Raw results saved to: {results_file}")

    # Generate summary
    print("\nGenerating summary table...")
    summary_df = evaluator.generate_summary_table(results_df)

    summary_file = output_dir / "phase1_summary_table.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Summary table saved to: {summary_file}")

    # Calculate domain gaps
    print("\nCalculating domain gaps...")
    gaps_df = evaluator.calculate_domain_gaps(summary_df)

    gaps_file = output_dir / "phase1_domain_gaps.csv"
    gaps_df.to_csv(gaps_file, index=False)
    print(f"✅ Domain gaps saved to: {gaps_file}")

    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    latex_file = output_dir / "reproduced_main_table.tex"
    generate_latex_table(summary_df, str(latex_file))

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults:")
    print(f"  - Raw results: {results_file}")
    print(f"  - Summary: {summary_file}")
    print(f"  - Domain gaps: {gaps_file}")
    print(f"  - LaTeX table: {latex_file}")

    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))

    print("\nDomain Gaps (%):")
    print(gaps_df.to_string(index=False))
