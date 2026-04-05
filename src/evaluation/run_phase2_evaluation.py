"""
ARIA Phase II-C: Full 6-Variant Evaluation with LLM Judge

Evaluates all 6 ARIA variants using domain-specific 2D materials metrics:
1. BASELINE (Pure LLM)
2. KG_ONLY (Pure graph traversal)
3. NAIVE_KG (KG+LLM, no retrieval)
4. ARIA_CORE (3-tier reasoning)
5. ARIA_SEARCH (+ literature search)
6. ARIA_FULL (+ chain-of-thought)

Uses OllamaJudge with 4 metrics:
- Processing Feasibility (40 pts)
- Structure Emergence (30 pts)
- Property Consistency (20 pts)
- Causal PSP Reasoning (10 pts)

Author: ARIA Team
Date: 2026-02-02
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import all 6 variants
from variants.baseline_ollama import BaselineOllama
from variants.kg_only import KGOnly
from variants.naive_kg_ollama import NaiveKGOllama
from variants.aria_core_ollama import ARIACoreOllama
from variants.aria_search_ollama import ARIASearchOllama
from variants.aria_full_ollama import ARIAFullOllama

# Import LLM judge
from evaluation.ollama_judge import OllamaJudge, create_judge_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase2Evaluator:
    """
    Complete evaluator for Phase II: 6 variants with LLM-as-a-Judge.
    """

    def __init__(self, kg_file_id: str, kg_file_ood: str, judge_model: str = "qwen2:7b"):
        """
        Initialize evaluator with data paths and judge.

        Args:
            kg_file_id: Path to in-domain KG
            kg_file_ood: Path to out-of-domain KG
            judge_model: Ollama model for LLM judge
        """
        self.kg_file_id = kg_file_id
        self.kg_file_ood = kg_file_ood

        # Initialize LLM judge
        logger.info(f"Initializing LLM judge with {judge_model}...")
        self.judge = OllamaJudge(model=judge_model, temperature=0.0)

        # Initialize all 6 systems
        logger.info("Initializing 6 ARIA variants...")
        self.systems = self._initialize_systems()

        # Load test data
        logger.info("Loading test data...")
        self.test_data_id = self._load_test_data(kg_file_id, 'in-domain')
        self.test_data_ood = self._load_test_data(kg_file_ood, 'out-of-domain')

        logger.info(f"Loaded {len(self.test_data_id)} in-domain test cases")
        logger.info(f"Loaded {len(self.test_data_ood)} out-of-domain test cases")

    def _initialize_systems(self) -> Dict[str, Any]:
        """Initialize all 6 systems for evaluation."""
        systems = {}

        # System 1: BASELINE (Pure LLM)
        try:
            systems['BASELINE'] = BaselineOllama(model="qwen2:7b")
            logger.info("  ✅ BASELINE initialized")
        except Exception as e:
            logger.error(f"  ❌ BASELINE failed: {e}")
            systems['BASELINE'] = None

        # System 2: KG_ONLY (Pure graph)
        try:
            systems['KG_ONLY'] = KGOnly(kg_file=self.kg_file_id)
            logger.info("  ✅ KG_ONLY initialized")
        except Exception as e:
            logger.error(f"  ❌ KG_ONLY failed: {e}")
            systems['KG_ONLY'] = None

        # System 3: NAIVE_KG (KG+LLM, no retrieval)
        try:
            systems['NAIVE_KG'] = NaiveKGOllama(
                kg_file=self.kg_file_id,
                model="qwen2:7b"
            )
            logger.info("  ✅ NAIVE_KG initialized")
        except Exception as e:
            logger.error(f"  ❌ NAIVE_KG failed: {e}")
            systems['NAIVE_KG'] = None

        # System 4: ARIA_CORE (3-tier reasoning)
        try:
            systems['ARIA_CORE'] = ARIACoreOllama(
                kg_file=self.kg_file_id,
                model="qwen2:7b"
            )
            logger.info("  ✅ ARIA_CORE initialized")
        except Exception as e:
            logger.error(f"  ❌ ARIA_CORE failed: {e}")
            systems['ARIA_CORE'] = None

        # System 5: ARIA_SEARCH (+ literature search)
        try:
            systems['ARIA_SEARCH'] = ARIASearchOllama(
                kg_file=self.kg_file_id,
                model="qwen2:7b"
            )
            logger.info("  ✅ ARIA_SEARCH initialized")
        except Exception as e:
            logger.error(f"  ❌ ARIA_SEARCH failed: {e}")
            systems['ARIA_SEARCH'] = None

        # System 6: ARIA_FULL (+ chain-of-thought)
        try:
            systems['ARIA_FULL'] = ARIAFullOllama(
                kg_file=self.kg_file_id,
                model="qwen2:7b"
            )
            logger.info("  ✅ ARIA_FULL initialized")
        except Exception as e:
            logger.error(f"  ❌ ARIA_FULL failed: {e}")
            systems['ARIA_FULL'] = None

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

        # Limit to first 10 for efficiency (adjust as needed)
        for i, rel in enumerate(relationships[:10]):
            if not isinstance(rel, dict):
                continue

            # Extract synthesis and property information
            synthesis = {
                'cause_parameter': rel.get('cause_parameter', ''),
                'method': rel.get('method', ''),
                'temperature': rel.get('temperature', ''),
                'atmosphere': rel.get('atmosphere', ''),
                'time': rel.get('time', '')
            }

            properties = {
                'effect_on_doping': rel.get('effect_on_doping', ''),
                'affected_property': rel.get('affected_property', ''),
                'property_changes': rel.get('property_changes', {}),
                'band_gap': rel.get('band_gap', ''),
                'carrier_type': rel.get('carrier_type', '')
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

    def evaluate_system(
        self,
        system_name: str,
        system: Any,
        test_cases: List[Dict],
        task: str
    ) -> List[Dict]:
        """
        Evaluate a single system using LLM judge.

        Args:
            system_name: Name of the system
            system: System instance
            test_cases: List of test cases
            task: 'forward' or 'inverse'

        Returns:
            List of result dictionaries with LLM judge scores
        """
        if system is None:
            logger.warning(f"Skipping {system_name} - not initialized")
            return []

        results = []

        for i, test_case in enumerate(test_cases):
            logger.info(f"  [{i+1}/{len(test_cases)}] {system_name} - {task} - {test_case['domain']}")

            try:
                # Get prediction from system
                if task == 'forward':
                    # Forward prediction: synthesis → properties
                    prediction = system.forward_prediction(test_case['synthesis_conditions'])
                    ground_truth = test_case['ground_truth_properties']
                    query = test_case['synthesis_conditions']
                else:
                    # Inverse design: properties → synthesis
                    prediction = system.inverse_design(test_case['properties'])
                    ground_truth = test_case['ground_truth_synthesis']
                    query = test_case['properties']

                # Evaluate with LLM judge
                logger.info(f"    Evaluating with LLM judge...")
                judge_result = self.judge.evaluate_all_metrics(
                    query=query,
                    prediction=prediction,
                    ground_truth=ground_truth
                )

                # Extract scores
                result = {
                    'system': system_name,
                    'domain': test_case['domain'],
                    'task': task,
                    'test_id': test_case['id'],
                    'processing_feasibility': judge_result['metric_scores']['processing_feasibility']['score'],
                    'structure_emergence': judge_result['metric_scores']['structure_emergence']['score'],
                    'property_consistency': judge_result['metric_scores']['property_consistency']['score'],
                    'causal_psp_reasoning': judge_result['metric_scores']['causal_psp_reasoning']['score'],
                    'overall_score': judge_result['overall_score'],
                    'overall_normalized': judge_result['overall_score_normalized'],
                    'prediction': prediction,
                    'ground_truth': ground_truth
                }

                results.append(result)
                logger.info(f"    Overall: {result['overall_score']:.1f}/100")

            except Exception as e:
                logger.error(f"    ❌ Error: {e}")
                # Add failed result with zeros
                results.append({
                    'system': system_name,
                    'domain': test_case['domain'],
                    'task': task,
                    'test_id': test_case['id'],
                    'processing_feasibility': 0.0,
                    'structure_emergence': 0.0,
                    'property_consistency': 0.0,
                    'causal_psp_reasoning': 0.0,
                    'overall_score': 0.0,
                    'overall_normalized': 0.0,
                    'error': str(e)
                })

        return results

    def run_full_evaluation(self) -> pd.DataFrame:
        """
        Run complete evaluation on all 6 systems, tasks, and domains.

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

                # Evaluate all 6 systems
                for system_name, system in self.systems.items():
                    results = self.evaluate_system(
                        system_name, system, test_cases, task
                    )
                    all_results.extend(results)

        return pd.DataFrame(all_results)

    def generate_summary_table(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary table with averaged metrics.

        Args:
            results_df: Raw evaluation results

        Returns:
            Summary DataFrame
        """
        # Group by system, domain, task and average metrics
        summary = results_df.groupby(['system', 'domain', 'task']).agg({
            'processing_feasibility': 'mean',
            'structure_emergence': 'mean',
            'property_consistency': 'mean',
            'causal_psp_reasoning': 'mean',
            'overall_score': 'mean',
            'overall_normalized': 'mean'
        }).reset_index()

        # Round to 1 decimal place
        for col in ['processing_feasibility', 'structure_emergence', 'property_consistency',
                    'causal_psp_reasoning', 'overall_score', 'overall_normalized']:
            summary[col] = summary[col].round(1)

        return summary

    def calculate_component_contributions(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate component contributions (ablation analysis).

        Component progression:
        BASELINE → NAIVE_KG (+ KG)
        NAIVE_KG → ARIA_CORE (+ 3-tier)
        ARIA_CORE → ARIA_SEARCH (+ literature)
        ARIA_SEARCH → ARIA_FULL (+ CoT)

        Args:
            summary_df: Summary table

        Returns:
            DataFrame with component contributions
        """
        contributions = []

        # Define component pairs
        pairs = [
            ('KG Value', 'NAIVE_KG', 'BASELINE'),
            ('Tier Value', 'ARIA_CORE', 'NAIVE_KG'),
            ('Search Value', 'ARIA_SEARCH', 'ARIA_CORE'),
            ('CoT Value', 'ARIA_FULL', 'ARIA_SEARCH')
        ]

        for component_name, system_with, system_without in pairs:
            for task in summary_df['task'].unique():
                for domain in summary_df['domain'].unique():
                    # Get scores for both systems
                    with_scores = summary_df[
                        (summary_df['system'] == system_with) &
                        (summary_df['task'] == task) &
                        (summary_df['domain'] == domain)
                    ]

                    without_scores = summary_df[
                        (summary_df['system'] == system_without) &
                        (summary_df['task'] == task) &
                        (summary_df['domain'] == domain)
                    ]

                    if not with_scores.empty and not without_scores.empty:
                        contribution = {
                            'component': component_name,
                            'task': task,
                            'domain': domain,
                            'processing_feasibility_delta': (
                                with_scores['processing_feasibility'].values[0] -
                                without_scores['processing_feasibility'].values[0]
                            ),
                            'structure_emergence_delta': (
                                with_scores['structure_emergence'].values[0] -
                                without_scores['structure_emergence'].values[0]
                            ),
                            'property_consistency_delta': (
                                with_scores['property_consistency'].values[0] -
                                without_scores['property_consistency'].values[0]
                            ),
                            'causal_psp_reasoning_delta': (
                                with_scores['causal_psp_reasoning'].values[0] -
                                without_scores['causal_psp_reasoning'].values[0]
                            ),
                            'overall_delta': (
                                with_scores['overall_score'].values[0] -
                                without_scores['overall_score'].values[0]
                            )
                        }
                        contributions.append(contribution)

        return pd.DataFrame(contributions)

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
                        'processing_feasibility_gap': (
                            id_data['processing_feasibility'].values[0] -
                            ood_data['processing_feasibility'].values[0]
                        ),
                        'structure_emergence_gap': (
                            id_data['structure_emergence'].values[0] -
                            ood_data['structure_emergence'].values[0]
                        ),
                        'property_consistency_gap': (
                            id_data['property_consistency'].values[0] -
                            ood_data['property_consistency'].values[0]
                        ),
                        'causal_psp_reasoning_gap': (
                            id_data['causal_psp_reasoning'].values[0] -
                            ood_data['causal_psp_reasoning'].values[0]
                        ),
                        'overall_gap': (
                            id_data['overall_score'].values[0] -
                            ood_data['overall_score'].values[0]
                        )
                    }
                    gaps.append(gap)

        return pd.DataFrame(gaps)


def generate_latex_table_phase2(summary_df: pd.DataFrame, output_file: str):
    """
    Generate LaTeX table matching NEW_TABLE_2 format.

    Args:
        summary_df: Summary table with results
        output_file: Output file path
    """
    latex = []
    latex.append("\\begin{table*}[ht!]")
    latex.append("\\centering")
    latex.append("\\caption{\\textbf{Six-variant ablation study on 2D materials processing reasoning.}}")
    latex.append("\\label{tab:phase2_results}")
    latex.append("\\small")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{System} & \\textbf{Domain} & \\textbf{Processing} & \\textbf{Structure} & \\textbf{Property} & \\textbf{PSP} & \\textbf{Overall} \\\\")
    latex.append("& & \\textbf{Feasibility} & \\textbf{Emergence} & \\textbf{Consistency} & \\textbf{Reasoning} & \\\\")
    latex.append("& & (0-40) & (0-30) & (0-20) & (0-10) & (0-100) \\\\")
    latex.append("\\midrule")

    # Define system order
    systems = ['BASELINE', 'KG_ONLY', 'NAIVE_KG', 'ARIA_CORE', 'ARIA_SEARCH', 'ARIA_FULL']

    # Add data rows for each task
    for task in ['forward', 'inverse']:
        latex.append(f"\\multicolumn{{7}}{{c}}{{\\textit{{\\textbf{{{task.title()} Prediction}}}}}} \\\\")
        latex.append("\\midrule")

        for system in systems:
            task_data = summary_df[
                (summary_df['system'] == system) &
                (summary_df['task'] == task)
            ].sort_values('domain')

            for _, row in task_data.iterrows():
                domain_label = 'ID' if row['domain'] == 'in-domain' else 'OOD'
                latex.append(
                    f"{system} & {domain_label} & "
                    f"{row['processing_feasibility']:.1f} & "
                    f"{row['structure_emergence']:.1f} & "
                    f"{row['property_consistency']:.1f} & "
                    f"{row['causal_psp_reasoning']:.1f} & "
                    f"{row['overall_score']:.1f} \\\\"
                )

        if task == 'forward':
            latex.append("\\midrule")

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
    print("ARIA PHASE II EVALUATION (6 Variants + LLM Judge)")
    print("="*80)

    evaluator = Phase2Evaluator(kg_file_id, kg_file_ood, judge_model="qwen2:7b")

    print("\nRunning comprehensive evaluation with LLM judge...")
    print("⚠️  Note: This will take significantly longer than Phase I")
    print("   (LLM judge evaluates 4 metrics per prediction)")
    results_df = evaluator.run_full_evaluation()

    # Save raw results
    results_file = output_dir / "phase2_raw_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n✅ Raw results saved to: {results_file}")

    # Generate summary
    print("\nGenerating summary table...")
    summary_df = evaluator.generate_summary_table(results_df)

    summary_file = output_dir / "phase2_summary_table.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Summary table saved to: {summary_file}")

    # Calculate component contributions
    print("\nCalculating component contributions...")
    contributions_df = evaluator.calculate_component_contributions(summary_df)

    contributions_file = output_dir / "phase2_component_contributions.csv"
    contributions_df.to_csv(contributions_file, index=False)
    print(f"✅ Component contributions saved to: {contributions_file}")

    # Calculate domain gaps
    print("\nCalculating domain gaps...")
    gaps_df = evaluator.calculate_domain_gaps(summary_df)

    gaps_file = output_dir / "phase2_domain_gaps.csv"
    gaps_df.to_csv(gaps_file, index=False)
    print(f"✅ Domain gaps saved to: {gaps_file}")

    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    latex_file = output_dir / "phase2_results_table.tex"
    generate_latex_table_phase2(summary_df, str(latex_file))

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults:")
    print(f"  - Raw results: {results_file}")
    print(f"  - Summary: {summary_file}")
    print(f"  - Component contributions: {contributions_file}")
    print(f"  - Domain gaps: {gaps_file}")
    print(f"  - LaTeX table: {latex_file}")

    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))

    print("\nComponent Contributions:")
    print(contributions_df.to_string(index=False))

    print("\nDomain Gaps:")
    print(gaps_df.to_string(index=False))
