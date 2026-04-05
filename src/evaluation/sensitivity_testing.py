"""
ARIA Phase II-D: Sensitivity & Robustness Testing

Tests ARIA variants' robustness to various perturbations:
1. Physical Inconsistency - Invalid processing parameters
2. Broken PSP Chains - Missing causal links
3. Fluent-but-Invalid - Plausible-sounding nonsense
4. Paraphrase Robustness - Synonym/rephrasing variations

Expected behavior:
- Good systems should detect physical inconsistencies (low scores)
- Should maintain reasoning even with broken chains (fill gaps)
- Should score lower on fluent-but-invalid predictions
- Should be robust to paraphrasing (similar scores)

Author: ARIA Team
Date: 2026-02-02
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys
import logging
import random

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.ollama_judge import OllamaJudge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensitivityTester:
    """
    Sensitivity and robustness testing framework.
    """

    def __init__(self, judge_model: str = "qwen2:7b"):
        """
        Initialize sensitivity tester.

        Args:
            judge_model: Ollama model for LLM judge
        """
        self.judge = OllamaJudge(model=judge_model, temperature=0.0)
        logger.info(f"SensitivityTester initialized with {judge_model}")

    # ===== Perturbation Type 1: Physical Inconsistency =====

    def create_physical_inconsistency(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create physically impossible/inconsistent predictions.

        Examples:
        - Temperature exceeds material decomposition point
        - Pressure incompatible with synthesis method
        - Contradictory atmosphere (oxidizing + reducing)
        - Time scale too short for defect formation

        Args:
            test_case: Original test case

        Returns:
            Perturbed test case with physical inconsistency
        """
        perturbations = [
            {
                'type': 'impossible_temperature',
                'description': 'Temperature exceeds material stability limit',
                'perturbation': {
                    'processing': {
                        'method': 'CVD',
                        'temperature': '3000°C',  # Way too high for any 2D material
                        'atmosphere': 'H2/Ar'
                    },
                    'structure': {
                        'defects': 'sulfur vacancies',
                        'material': 'MoS2'
                    },
                    'properties': {
                        'band_gap': '1.8 eV',
                        'carrier_type': 'n-type'
                    }
                }
            },
            {
                'type': 'contradictory_atmosphere',
                'description': 'Oxidizing and reducing conditions simultaneously',
                'perturbation': {
                    'processing': {
                        'method': 'CVD',
                        'temperature': '800°C',
                        'atmosphere': 'O2 + H2'  # Contradictory
                    },
                    'structure': {
                        'defects': 'oxygen substitution',
                        'material': 'MoS2'
                    },
                    'properties': {
                        'band_gap': '2.0 eV',
                        'carrier_type': 'p-type'
                    }
                }
            },
            {
                'type': 'impossible_time_scale',
                'description': 'Time too short for defect annealing',
                'perturbation': {
                    'processing': {
                        'method': 'Thermal annealing',
                        'temperature': '400°C',
                        'time': '0.01 seconds',  # Way too short
                        'atmosphere': 'vacuum'
                    },
                    'structure': {
                        'defects': 'fully annealed defects',
                        'material': 'MoS2'
                    },
                    'properties': {
                        'band_gap': '1.9 eV',
                        'carrier_type': 'intrinsic'
                    }
                }
            }
        ]

        # Select random perturbation
        perturbation = random.choice(perturbations)

        return {
            'id': f"{test_case.get('id', 'unknown')}_phys_inconsist",
            'perturbation_type': 'physical_inconsistency',
            'perturbation_subtype': perturbation['type'],
            'description': perturbation['description'],
            'query': test_case.get('synthesis_conditions', {}),
            'prediction': perturbation['perturbation'],
            'ground_truth': test_case.get('ground_truth_properties', {}),
            'expected_behavior': 'Low scores, especially Processing Feasibility'
        }

    # ===== Perturbation Type 2: Broken PSP Chains =====

    def create_broken_psp_chain(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create predictions with missing causal links.

        Examples:
        - Processing → Properties (skip structure)
        - Structure described but no processing explanation
        - Properties without structural justification

        Args:
            test_case: Original test case

        Returns:
            Perturbed test case with broken PSP chain
        """
        perturbations = [
            {
                'type': 'missing_structure_link',
                'description': 'Processing → Properties (no structure explanation)',
                'perturbation': {
                    'processing': {
                        'method': 'CVD',
                        'temperature': '800°C',
                        'atmosphere': 'H2/Ar'
                    },
                    'structure': {},  # Missing!
                    'properties': {
                        'band_gap': '1.8 eV',
                        'carrier_type': 'n-type'
                    },
                    'reasoning': 'CVD at 800°C leads to n-type conductivity.'
                    # No mention of HOW (via what structural changes)
                }
            },
            {
                'type': 'missing_processing_link',
                'description': 'Structure described without processing explanation',
                'perturbation': {
                    'processing': {},  # Missing!
                    'structure': {
                        'defects': 'sulfur vacancies',
                        'material': 'MoS2',
                        'defect_density': '10^12 cm^-2'
                    },
                    'properties': {
                        'band_gap': '1.8 eV',
                        'carrier_type': 'n-type'
                    },
                    'reasoning': 'Sulfur vacancies cause n-type doping.'
                    # No explanation of how vacancies were created
                }
            },
            {
                'type': 'missing_property_link',
                'description': 'Processing and structure without property explanation',
                'perturbation': {
                    'processing': {
                        'method': 'CVD',
                        'temperature': '800°C',
                        'atmosphere': 'H2/Ar'
                    },
                    'structure': {
                        'defects': 'sulfur vacancies',
                        'material': 'MoS2'
                    },
                    'properties': {},  # Missing!
                    'reasoning': 'H2 atmosphere creates sulfur vacancies in MoS2.'
                    # No mention of what properties result
                }
            }
        ]

        perturbation = random.choice(perturbations)

        return {
            'id': f"{test_case.get('id', 'unknown')}_broken_psp",
            'perturbation_type': 'broken_psp_chain',
            'perturbation_subtype': perturbation['type'],
            'description': perturbation['description'],
            'query': test_case.get('synthesis_conditions', {}),
            'prediction': perturbation['perturbation'],
            'ground_truth': test_case.get('ground_truth_properties', {}),
            'expected_behavior': 'Low Causal PSP Reasoning score, systems should fill gaps'
        }

    # ===== Perturbation Type 3: Fluent-but-Invalid =====

    def create_fluent_invalid(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create plausible-sounding but scientifically incorrect predictions.

        Examples:
        - Correct terminology but wrong relationships
        - Fluent explanations of impossible phenomena
        - Mixing incompatible concepts

        Args:
            test_case: Original test case

        Returns:
            Perturbed test case with fluent-but-invalid content
        """
        perturbations = [
            {
                'type': 'wrong_carrier_mechanism',
                'description': 'Fluent but carrier type mechanism is backwards',
                'perturbation': {
                    'processing': {
                        'method': 'CVD',
                        'temperature': '800°C',
                        'atmosphere': 'H2/Ar'
                    },
                    'structure': {
                        'defects': 'sulfur vacancies',
                        'material': 'MoS2',
                        'defect_density': '10^12 cm^-2'
                    },
                    'properties': {
                        'band_gap': '1.8 eV',
                        'carrier_type': 'p-type'  # WRONG! Should be n-type
                    },
                    'reasoning': 'The reducing H2 atmosphere creates sulfur vacancies '
                                'which act as hole acceptors, leading to p-type conductivity. '
                                'This is consistent with the observed band gap of 1.8 eV.'
                    # Fluent and detailed, but scientifically backwards
                }
            },
            {
                'type': 'impossible_property_combination',
                'description': 'Internally contradictory properties',
                'perturbation': {
                    'processing': {
                        'method': 'Exfoliation',
                        'temperature': '25°C',
                        'method_details': 'mechanical exfoliation'
                    },
                    'structure': {
                        'material': 'graphene',
                        'layers': 'monolayer',
                        'defects': 'pristine'
                    },
                    'properties': {
                        'band_gap': '2.5 eV',  # WRONG! Graphene is zero-gap
                        'carrier_type': 'insulating',  # WRONG!
                        'mobility': '200000 cm^2/Vs'  # High mobility contradicts insulating
                    },
                    'reasoning': 'Pristine monolayer graphene exhibits a large band gap '
                                'of 2.5 eV due to quantum confinement, making it an excellent '
                                'insulator while maintaining high carrier mobility.'
                    # Fluent but physically impossible combination
                }
            },
            {
                'type': 'mixing_incompatible_concepts',
                'description': 'Combines concepts from different material systems incorrectly',
                'perturbation': {
                    'processing': {
                        'method': 'MBE',
                        'temperature': '600°C',
                        'substrate': 'sapphire'
                    },
                    'structure': {
                        'material': 'MoS2',
                        'defects': 'edge dislocations'  # Wrong type for 2D materials
                    },
                    'properties': {
                        'band_gap': '1.8 eV',
                        'mobility': '50 cm^2/Vs',
                        'ferromagnetism': 'strong'  # MoS2 is not magnetic!
                    },
                    'reasoning': 'Edge dislocations in MoS2 create magnetic domains '
                                'that exhibit ferromagnetic ordering at room temperature, '
                                'similar to what is observed in bulk ferromagnets.'
                    # Mixing 3D defect concepts with 2D materials + wrong magnetism
                }
            }
        ]

        perturbation = random.choice(perturbations)

        return {
            'id': f"{test_case.get('id', 'unknown')}_fluent_invalid",
            'perturbation_type': 'fluent_invalid',
            'perturbation_subtype': perturbation['type'],
            'description': perturbation['description'],
            'query': test_case.get('synthesis_conditions', {}),
            'prediction': perturbation['perturbation'],
            'ground_truth': test_case.get('ground_truth_properties', {}),
            'expected_behavior': 'Should detect scientific errors despite fluency'
        }

    # ===== Perturbation Type 4: Paraphrase Robustness =====

    def create_paraphrase(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create paraphrased version with synonyms/rephrasing.

        Examples:
        - "CVD" → "Chemical Vapor Deposition"
        - "800°C" → "1073 K"
        - "n-type" → "electron-doped"

        Args:
            test_case: Original test case

        Returns:
            Paraphrased test case (should get similar scores)
        """
        # Synonym mappings
        method_synonyms = {
            'CVD': 'Chemical Vapor Deposition',
            'MBE': 'Molecular Beam Epitaxy',
            'PLD': 'Pulsed Laser Deposition'
        }

        carrier_synonyms = {
            'n-type': 'electron-doped',
            'p-type': 'hole-doped',
            'intrinsic': 'undoped'
        }

        defect_synonyms = {
            'sulfur vacancies': 'missing sulfur atoms',
            'oxygen substitution': 'oxygen replacing sulfur',
            'interstitial': 'extra atoms between layers'
        }

        # Get original prediction
        original = test_case.get('ground_truth_properties', {})

        # Create paraphrased version
        paraphrased = {
            'processing': {},
            'structure': {},
            'properties': {},
            'reasoning': ''
        }

        # Paraphrase processing
        if 'method' in test_case.get('synthesis_conditions', {}):
            method = test_case['synthesis_conditions']['method']
            paraphrased['processing']['method'] = method_synonyms.get(method, method)

        if 'temperature' in test_case.get('synthesis_conditions', {}):
            temp = test_case['synthesis_conditions']['temperature']
            # Convert °C to K if possible
            if '°C' in temp:
                try:
                    celsius = int(temp.replace('°C', ''))
                    kelvin = celsius + 273
                    paraphrased['processing']['temperature'] = f'{kelvin} K'
                except:
                    paraphrased['processing']['temperature'] = temp
            else:
                paraphrased['processing']['temperature'] = temp

        # Paraphrase structure
        paraphrased['structure'] = test_case.get('ground_truth_synthesis', {}).copy()

        # Paraphrase properties
        paraphrased['properties'] = test_case.get('ground_truth_properties', {}).copy()
        if 'carrier_type' in paraphrased['properties']:
            carrier = paraphrased['properties']['carrier_type']
            paraphrased['properties']['carrier_type'] = carrier_synonyms.get(carrier, carrier)

        # Paraphrase reasoning (simple rephrasing)
        paraphrased['reasoning'] = (
            'The synthesis process leads to structural modifications that '
            'result in the observed electronic properties.'
        )

        return {
            'id': f"{test_case.get('id', 'unknown')}_paraphrase",
            'perturbation_type': 'paraphrase',
            'perturbation_subtype': 'synonym_replacement',
            'description': 'Paraphrased with synonyms and unit conversions',
            'query': test_case.get('synthesis_conditions', {}),
            'prediction': paraphrased,
            'ground_truth': test_case.get('ground_truth_properties', {}),
            'expected_behavior': 'Should get similar scores to original'
        }

    # ===== Batch Perturbation =====

    def generate_perturbation_suite(
        self,
        test_cases: List[Dict[str, Any]],
        perturbations_per_case: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Generate complete perturbation suite for test cases.

        Args:
            test_cases: Original test cases
            perturbations_per_case: Number of perturbations per case (default: 4, one of each type)

        Returns:
            List of perturbed test cases
        """
        perturbed_cases = []

        for test_case in test_cases:
            # Generate one perturbation of each type
            perturbed_cases.append(self.create_physical_inconsistency(test_case))
            perturbed_cases.append(self.create_broken_psp_chain(test_case))
            perturbed_cases.append(self.create_fluent_invalid(test_case))
            perturbed_cases.append(self.create_paraphrase(test_case))

        logger.info(f"Generated {len(perturbed_cases)} perturbed test cases "
                   f"from {len(test_cases)} originals")

        return perturbed_cases

    # ===== Evaluation =====

    def evaluate_perturbation_suite(
        self,
        perturbed_cases: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Evaluate all perturbed cases with LLM judge.

        Args:
            perturbed_cases: List of perturbed test cases

        Returns:
            DataFrame with sensitivity analysis results
        """
        results = []

        for i, case in enumerate(perturbed_cases):
            logger.info(f"Evaluating perturbation {i+1}/{len(perturbed_cases)}: "
                       f"{case['perturbation_type']} - {case['perturbation_subtype']}")

            try:
                # Evaluate with judge
                judge_result = self.judge.evaluate_all_metrics(
                    query=case['query'],
                    prediction=case['prediction'],
                    ground_truth=case['ground_truth']
                )

                # Extract scores
                result = {
                    'test_id': case['id'],
                    'perturbation_type': case['perturbation_type'],
                    'perturbation_subtype': case['perturbation_subtype'],
                    'description': case['description'],
                    'expected_behavior': case['expected_behavior'],
                    'processing_feasibility': judge_result['metric_scores']['processing_feasibility']['score'],
                    'structure_emergence': judge_result['metric_scores']['structure_emergence']['score'],
                    'property_consistency': judge_result['metric_scores']['property_consistency']['score'],
                    'causal_psp_reasoning': judge_result['metric_scores']['causal_psp_reasoning']['score'],
                    'overall_score': judge_result['overall_score']
                }

                results.append(result)
                logger.info(f"  Overall: {result['overall_score']:.1f}/100")

            except Exception as e:
                logger.error(f"  ❌ Error: {e}")
                results.append({
                    'test_id': case['id'],
                    'perturbation_type': case['perturbation_type'],
                    'perturbation_subtype': case['perturbation_subtype'],
                    'description': case['description'],
                    'expected_behavior': case['expected_behavior'],
                    'processing_feasibility': 0.0,
                    'structure_emergence': 0.0,
                    'property_consistency': 0.0,
                    'causal_psp_reasoning': 0.0,
                    'overall_score': 0.0,
                    'error': str(e)
                })

        return pd.DataFrame(results)

    # ===== Analysis =====

    def analyze_sensitivity(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze sensitivity testing results.

        Args:
            results_df: Results from evaluate_perturbation_suite

        Returns:
            Dictionary with analysis summary
        """
        analysis = {}

        # Group by perturbation type
        by_type = results_df.groupby('perturbation_type').agg({
            'processing_feasibility': 'mean',
            'structure_emergence': 'mean',
            'property_consistency': 'mean',
            'causal_psp_reasoning': 'mean',
            'overall_score': 'mean'
        }).reset_index()

        analysis['by_perturbation_type'] = by_type

        # Expected score patterns
        analysis['expected_vs_actual'] = {
            'physical_inconsistency': {
                'expected': 'Low Processing Feasibility scores',
                'actual_mean': by_type[by_type['perturbation_type'] == 'physical_inconsistency']['processing_feasibility'].values[0] if len(by_type[by_type['perturbation_type'] == 'physical_inconsistency']) > 0 else None
            },
            'broken_psp_chain': {
                'expected': 'Low Causal PSP Reasoning scores',
                'actual_mean': by_type[by_type['perturbation_type'] == 'broken_psp_chain']['causal_psp_reasoning'].values[0] if len(by_type[by_type['perturbation_type'] == 'broken_psp_chain']) > 0 else None
            },
            'fluent_invalid': {
                'expected': 'Low overall scores despite fluency',
                'actual_mean': by_type[by_type['perturbation_type'] == 'fluent_invalid']['overall_score'].values[0] if len(by_type[by_type['perturbation_type'] == 'fluent_invalid']) > 0 else None
            },
            'paraphrase': {
                'expected': 'Similar scores to original (high consistency)',
                'actual_mean': by_type[by_type['perturbation_type'] == 'paraphrase']['overall_score'].values[0] if len(by_type[by_type['perturbation_type'] == 'paraphrase']) > 0 else None
            }
        }

        return analysis


def generate_sensitivity_report(
    results_df: pd.DataFrame,
    analysis: Dict[str, Any],
    output_file: str
):
    """
    Generate sensitivity testing report.

    Args:
        results_df: Results dataframe
        analysis: Analysis dictionary
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ARIA Phase II-D: Sensitivity & Robustness Testing Report\n")
        f.write("="*80 + "\n\n")

        # Summary by perturbation type
        f.write("SUMMARY BY PERTURBATION TYPE\n")
        f.write("-"*80 + "\n\n")
        f.write(analysis['by_perturbation_type'].to_string(index=False))
        f.write("\n\n")

        # Expected vs Actual
        f.write("EXPECTED VS ACTUAL BEHAVIOR\n")
        f.write("-"*80 + "\n\n")
        for pert_type, data in analysis['expected_vs_actual'].items():
            f.write(f"{pert_type.upper()}:\n")
            f.write(f"  Expected: {data['expected']}\n")
            f.write(f"  Actual: {data['actual_mean']:.1f}\n\n")

        # Detailed results
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(results_df.to_string(index=False))

    logger.info(f"Sensitivity report saved to: {output_file}")


if __name__ == '__main__':
    # Example usage
    print("\n" + "="*80)
    print("ARIA Phase II-D: Sensitivity Testing Example")
    print("="*80 + "\n")

    # Initialize tester
    tester = SensitivityTester(judge_model="qwen2:7b")

    # Example test case
    test_case = {
        'id': 'example_1',
        'synthesis_conditions': {
            'method': 'CVD',
            'temperature': '800°C',
            'atmosphere': 'H2/Ar'
        },
        'ground_truth_properties': {
            'band_gap': '1.8 eV',
            'carrier_type': 'n-type'
        },
        'ground_truth_synthesis': {
            'method': 'CVD',
            'temperature': '800°C'
        }
    }

    # Generate perturbations
    print("Generating perturbation suite...")
    perturbed = tester.generate_perturbation_suite([test_case])
    print(f"Generated {len(perturbed)} perturbed cases\n")

    # Evaluate (commented out - uncomment to run full evaluation)
    # print("Evaluating perturbations with LLM judge...")
    # results_df = tester.evaluate_perturbation_suite(perturbed)
    # analysis = tester.analyze_sensitivity(results_df)
    # generate_sensitivity_report(results_df, analysis, 'results/phase2/sensitivity_report.txt')

    print("Sensitivity testing framework ready!")
    print("Run full evaluation by uncommenting the evaluation section above.")
