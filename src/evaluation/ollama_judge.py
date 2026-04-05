"""
ARIA Phase II-B: LLM-as-a-Judge Implementation

LLM-based evaluator for 2D materials processing reasoning using Ollama.
Implements 4 domain-specific metrics with detailed rubrics.

Author: ARIA Team
Date: 2026-02-02
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from ollama_client import get_ollama_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaJudge:
    """
    LLM-based evaluator for 2D materials processing predictions.

    Uses Ollama to score predictions on 4 domain-specific metrics:
    1. Processing Feasibility (0-40)
    2. Structure Emergence (0-30)
    3. Property Consistency (0-20)
    4. Causal PSP Reasoning (0-10)
    """

    # Metric definitions with scoring rubrics
    METRICS = {
        "processing_feasibility": {
            "name": "Processing Feasibility",
            "max_score": 40,
            "description": "Thermodynamic & kinetic viability of predicted processing conditions",
            "rubric": """
Score 35-40: All conditions thermodynamically/kinetically viable, realistic equipment
  - Temperature ranges appropriate for material phase stability
  - Pressure conditions compatible with synthesis method
  - Time scales realistic for defect formation/annealing
  - Atmosphere correct for doping/oxidation control
  - Equipment feasible in typical labs
  - No safety hazards or unrealistic conditions

Score 25-34: Minor feasibility issues
  - Slightly high/low temperature (but within possible range)
  - Extended time scales (but not impossible)
  - Common equipment with minor modifications

Score 15-24: Significant issues
  - Wrong atmosphere for desired outcome
  - Unrealistic pressure for method
  - Temperature incompatible with substrate/precursor

Score 0-14: Fundamentally impossible
  - Violates phase diagram or thermodynamic constraints
  - Conditions destroy material before defects form
  - Physically impossible parameter combinations
"""
        },

        "structure_emergence": {
            "name": "Structure Emergence",
            "max_score": 30,
            "description": "Accuracy of predicted structural outcomes from processing",
            "rubric": """
Score 25-30: Excellent structural prediction
  - Correct defect type (vacancy, substitution, interstitial)
  - Realistic defect density/concentration
  - Accurate lattice strain effects
  - Correct stacking order (for heterostructures)
  - Phase purity considerations addressed
  - Crystal structure/symmetry preserved or correctly modified

Score 18-24: Good with minor inaccuracies
  - Correct defect type but density slightly off
  - Strain magnitude imprecise but directionally correct
  - Missing minor structural details

Score 10-17: Partially correct
  - Right defect family but wrong specific type
  - Major uncertainty in defect density
  - Incomplete structural description

Score 0-9: Incorrect or major errors
  - Wrong defect type for processing conditions
  - Structurally impossible outcomes
  - Ignores symmetry constraints
"""
        },

        "property_consistency": {
            "name": "Property Consistency",
            "max_score": 20,
            "description": "Coherence between predicted properties and structure/processing",
            "rubric": """
Score 17-20: Fully consistent
  - Electronic properties (band gap, carrier type, conductivity) match structure
  - Mechanical properties (strength, modulus) consistent with defect density
  - Optical properties aligned with electronic structure
  - Magnetic properties match dopant configuration
  - No violations of conservation laws or symmetry

Score 12-16: Minor inconsistencies
  - Band gap magnitude off by <0.5 eV
  - Carrier concentration estimates imprecise
  - Qualitative trends correct but quantitative values approximate

Score 6-11: Significant inconsistencies
  - Wrong carrier type (n vs p)
  - Mechanical properties inconsistent with defects
  - Optical absorption doesn't match band structure

Score 0-5: Major violations
  - Metallic predicted for insulating structure
  - Physically impossible property values
  - Contradictions between different properties
"""
        },

        "causal_psp_reasoning": {
            "name": "Causal PSP Reasoning",
            "max_score": 10,
            "description": "Quality of Processing→Structure→Property causal chain",
            "rubric": """
Score 8-10: Excellent causal reasoning
  - Clear P→S→P chain with explicit connections
  - Mechanistic explanations (how processing creates structure)
  - Physical justifications grounded in theory
  - Uncertainty acknowledged and quantified
  - References to principles or literature

Score 5-7: Good partial reasoning
  - P→S→P chain present but incomplete
  - Some mechanistic explanations
  - Basic physical reasoning
  - Limited uncertainty discussion

Score 2-4: Minimal reasoning
  - Mentions connections but lacks detail
  - Empirical correlations without mechanisms
  - Weak causal links

Score 0-1: No causal reasoning
  - No P→S→P chain
  - Pure correlation or guessing
  - Missing scientific justification
"""
        }
    }

    def __init__(self, model: str = "qwen2:7b", temperature: float = 0.0):
        """
        Initialize the LLM judge.

        Args:
            model: Ollama model name
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model = model
        self.temperature = temperature
        self.ollama = get_ollama_client(model=model)
        logger.info(f"OllamaJudge initialized with model={model}")

    def _create_judge_prompt(
        self,
        query: Dict[str, Any],
        prediction: Dict[str, Any],
        ground_truth: Dict[str, Any],
        metric_key: str
    ) -> str:
        """
        Create judge prompt for specific metric evaluation.

        Args:
            query: Input query/conditions
            prediction: Model's prediction
            ground_truth: Reference answer
            metric_key: Which metric to evaluate

        Returns:
            Formatted prompt string
        """
        metric = self.METRICS[metric_key]

        prompt = f"""You are an expert materials scientist evaluating AI-generated predictions for 2D materials processing.

**TASK:** Score the prediction on **{metric['name']}** ({metric['description']}).

**METRIC DEFINITION:**
{metric['name']} (0-{metric['max_score']} points)
{metric['description']}

**SCORING RUBRIC:**
{metric['rubric']}

---

**INPUT QUERY:**
```json
{json.dumps(query, indent=2)}
```

**GROUND TRUTH (Reference):**
```json
{json.dumps(ground_truth, indent=2)}
```

**MODEL PREDICTION (To Evaluate):**
```json
{json.dumps(prediction, indent=2)}
```

---

**EVALUATION INSTRUCTIONS:**

1. **Compare** the prediction to ground truth on {metric['name']}
2. **Score** the prediction (0-{metric['max_score']}) using the rubric above
3. **Justify** your score with specific examples from the prediction
4. **Identify** failure modes (what went wrong, if anything)
5. **Identify** strengths (what the model did well)

**OUTPUT FORMAT (strict JSON):**
```json
{{
  "metric": "{metric_key}",
  "score": <float 0-{metric['max_score']}>,
  "justification": "<detailed explanation of score>",
  "failure_modes": ["<specific issue 1>", "<specific issue 2>", ...],
  "strengths": ["<what worked well 1>", "<what worked well 2>", ...],
  "key_evidence": "<most important evidence for this score>"
}}
```

**IMPORTANT:**
- Be strict but fair - only high scores for truly excellent predictions
- Focus on {metric['name']} specifically, not overall quality
- Provide concrete evidence from the prediction text
- Output ONLY valid JSON, no other text
"""
        return prompt

    def score_prediction(
        self,
        query: Dict[str, Any],
        prediction: Dict[str, Any],
        ground_truth: Dict[str, Any],
        metric_key: str
    ) -> Dict[str, Any]:
        """
        Score a single prediction on a specific metric.

        Args:
            query: Input query
            prediction: Model's prediction
            ground_truth: Reference answer
            metric_key: Which metric to evaluate

        Returns:
            Dictionary with score, justification, failure modes, strengths
        """
        if metric_key not in self.METRICS:
            raise ValueError(f"Unknown metric: {metric_key}. "
                           f"Available: {list(self.METRICS.keys())}")

        prompt = self._create_judge_prompt(query, prediction, ground_truth, metric_key)

        try:
            # Get LLM judgment
            response = self.ollama.generate_json(
                prompt,
                temperature=self.temperature
            )

            # Validate response structure
            required_fields = ['metric', 'score', 'justification',
                             'failure_modes', 'strengths']
            for field in required_fields:
                if field not in response:
                    logger.warning(f"Missing field '{field}' in judge response")
                    response[field] = [] if field.endswith('s') else ""

            # Validate score range
            max_score = self.METRICS[metric_key]['max_score']
            if not (0 <= response.get('score', 0) <= max_score):
                logger.warning(f"Score {response.get('score')} out of range [0, {max_score}], "
                             f"clipping")
                response['score'] = max(0, min(response.get('score', 0), max_score))

            return response

        except Exception as e:
            logger.error(f"Judge scoring failed for {metric_key}: {e}")
            return {
                'metric': metric_key,
                'score': 0.0,
                'justification': f"Evaluation failed: {str(e)}",
                'failure_modes': ['Judge evaluation error'],
                'strengths': [],
                'key_evidence': '',
                'error': str(e)
            }

    def evaluate_all_metrics(
        self,
        query: Dict[str, Any],
        prediction: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate prediction on all 4 metrics.

        Args:
            query: Input query
            prediction: Model's prediction
            ground_truth: Reference answer

        Returns:
            Dictionary with scores for all metrics plus overall score
        """
        results = {
            'query': query,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'metric_scores': {}
        }

        total_score = 0.0
        max_total = 100.0  # 40 + 30 + 20 + 10

        for metric_key in self.METRICS.keys():
            logger.info(f"  Evaluating {self.METRICS[metric_key]['name']}...")

            score_result = self.score_prediction(
                query, prediction, ground_truth, metric_key
            )

            results['metric_scores'][metric_key] = score_result
            total_score += score_result.get('score', 0)

            logger.info(f"    Score: {score_result.get('score', 0)}/{self.METRICS[metric_key]['max_score']}")

        results['overall_score'] = total_score
        results['overall_score_normalized'] = (total_score / max_total) * 100  # 0-100 scale

        return results

    def batch_evaluate(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple test cases.

        Args:
            test_cases: List of dicts with 'query', 'prediction', 'ground_truth'

        Returns:
            List of evaluation results
        """
        results = []

        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i+1}/{len(test_cases)}")

            result = self.evaluate_all_metrics(
                test_case.get('query', {}),
                test_case.get('prediction', {}),
                test_case.get('ground_truth', {})
            )

            result['test_case_id'] = test_case.get('id', i)
            results.append(result)

        return results


def create_judge_report(
    evaluation_results: List[Dict[str, Any]],
    output_file: str
) -> None:
    """
    Create human-readable evaluation report.

    Args:
        evaluation_results: Results from batch_evaluate
        output_file: Path to save report
    """
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ARIA Phase II: LLM Judge Evaluation Report\n")
        f.write("=" * 80 + "\n\n")

        for i, result in enumerate(evaluation_results):
            f.write(f"\n{'='*80}\n")
            f.write(f"Test Case {i+1}: {result.get('test_case_id', 'N/A')}\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"Overall Score: {result['overall_score']:.1f}/100 "
                   f"({result['overall_score_normalized']:.1f}%)\n\n")

            for metric_key, score_data in result.get('metric_scores', {}).items():
                metric_name = OllamaJudge.METRICS[metric_key]['name']
                max_score = OllamaJudge.METRICS[metric_key]['max_score']

                f.write(f"--- {metric_name} ---\n")
                f.write(f"Score: {score_data.get('score', 0)}/{max_score}\n")
                f.write(f"Justification: {score_data.get('justification', 'N/A')}\n")

                if score_data.get('strengths'):
                    f.write(f"Strengths:\n")
                    for strength in score_data['strengths']:
                        f.write(f"  + {strength}\n")

                if score_data.get('failure_modes'):
                    f.write(f"Failure Modes:\n")
                    for failure in score_data['failure_modes']:
                        f.write(f"  - {failure}\n")

                f.write("\n")

        # Summary statistics
        f.write(f"\n{'='*80}\n")
        f.write("SUMMARY STATISTICS\n")
        f.write(f"{'='*80}\n\n")

        if evaluation_results:
            avg_overall = sum(r['overall_score'] for r in evaluation_results) / len(evaluation_results)
            f.write(f"Average Overall Score: {avg_overall:.1f}/100\n\n")

            for metric_key in OllamaJudge.METRICS.keys():
                scores = [r['metric_scores'][metric_key]['score']
                         for r in evaluation_results]
                avg_score = sum(scores) / len(scores)
                max_score = OllamaJudge.METRICS[metric_key]['max_score']
                metric_name = OllamaJudge.METRICS[metric_key]['name']

                f.write(f"{metric_name}: {avg_score:.1f}/{max_score} "
                       f"({(avg_score/max_score)*100:.1f}%)\n")

    logger.info(f"Report saved to: {output_file}")


if __name__ == '__main__':
    # Example usage
    print("\n" + "="*80)
    print("ARIA Phase II-B: LLM-as-a-Judge Example")
    print("="*80 + "\n")

    # Initialize judge
    judge = OllamaJudge(model="qwen2:7b", temperature=0.0)

    # Example test case
    test_case = {
        'id': 'example_1',
        'query': {
            'task': 'forward_prediction',
            'processing': {
                'method': 'CVD',
                'temperature': '800°C',
                'atmosphere': 'H2/Ar',
                'precursor': 'MoO3'
            }
        },
        'prediction': {
            'structure': {
                'material': 'MoS2',
                'defects': 'sulfur vacancies',
                'defect_density': '~10^12 cm^-2'
            },
            'properties': {
                'band_gap': '1.8 eV (direct)',
                'carrier_type': 'n-type',
                'mobility': '~50 cm^2/Vs'
            },
            'reasoning': 'CVD at 800°C with H2 atmosphere creates reducing '
                        'conditions that lead to sulfur vacancies in MoS2. '
                        'These vacancies donate electrons, making the material n-type.'
        },
        'ground_truth': {
            'structure': {
                'material': 'MoS2',
                'defects': 'sulfur vacancies',
                'defect_density': '10^12-10^13 cm^-2'
            },
            'properties': {
                'band_gap': '1.8 eV',
                'carrier_type': 'n-type',
                'conductivity': 'enhanced'
            }
        }
    }

    print("Evaluating example prediction...\n")
    result = judge.evaluate_all_metrics(
        test_case['query'],
        test_case['prediction'],
        test_case['ground_truth']
    )

    print(f"\nOverall Score: {result['overall_score']:.1f}/100\n")

    for metric_key, score_data in result['metric_scores'].items():
        metric_name = judge.METRICS[metric_key]['name']
        print(f"{metric_name}: {score_data['score']}/{judge.METRICS[metric_key]['max_score']}")

    print("\nExample evaluation complete!")
