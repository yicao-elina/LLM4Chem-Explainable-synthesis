"""
ARIA Evaluation Metrics - Phase I

Implements 5 core metrics for evaluating materials processing predictions:
1. Scientific Accuracy
2. Functional Equivalence
3. Reasoning Quality
4. Completeness
5. Interpretability

Author: ARIA Team
Date: 2026-02-02
"""

import re
import numpy as np
from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================================================================
# Helper Functions
# ==============================================================================

def flatten_dict(d: dict, parent_key: str = '', sep: str = ' ') -> dict:
    """Flattens nested dictionary for comparison."""
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
    """Converts dictionary to string for embedding."""
    if not d or not isinstance(d, dict):
        return ""
    flattened = flatten_dict(d)
    return " and ".join([f"{key} is {value}" for key, value in sorted(flattened.items())])


def calculate_semantic_similarity(text1: str, text2: str,
                                 model: SentenceTransformer) -> float:
    """Calculates cosine similarity between text embeddings."""
    if not text1 or not text2:
        return 0.0
    try:
        embeddings = model.encode([text1, text2])
        similarity_matrix = cosine_similarity(embeddings)
        return float(similarity_matrix[0, 1])
    except:
        return 0.0


def extract_scientific_terms(text: str) -> set:
    """Extract scientific terms, units, and concepts from text."""
    if not isinstance(text, str):
        text = str(text)

    # Common materials science terms
    terms = set()

    # Extract chemical formulas (e.g., MoS2, WS2, etc.)
    formulas = re.findall(r'\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\b', text)
    terms.update(formulas)

    # Extract numbers with units
    units = re.findall(r'\d+\.?\d*\s*(?:°C|K|eV|nm|μm|GPa|MPa|%|V|A|Ω)', text)
    terms.update(units)

    # Extract common materials science keywords
    keywords = [
        'n-type', 'p-type', 'conductivity', 'mobility', 'doping',
        'bandgap', 'defect', 'strain', 'CVD', 'PVD', 'annealing',
        'oxidation', 'reduction', 'substrate', 'monolayer', 'bilayer',
        'carrier', 'electron', 'hole', 'dopant', 'crystal', 'phase'
    ]
    for keyword in keywords:
        if keyword.lower() in text.lower():
            terms.add(keyword)

    return terms


def extract_key_values(data: dict) -> dict:
    """Extract key-value pairs from nested dict."""
    result = {}

    def recursive_extract(d, prefix=''):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    recursive_extract(v, f"{prefix}{k}_" if prefix else f"{k}_")
                elif v is not None and str(v).lower() not in ["", "not reported", "null", "none"]:
                    key = f"{prefix}{k}" if prefix else k
                    result[key] = v

    recursive_extract(data)
    return result


# ==============================================================================
# Metric 1: Scientific Accuracy
# ==============================================================================

def scientific_accuracy(predicted: dict, ground_truth: dict,
                       embedding_model: SentenceTransformer) -> float:
    """
    Measure scientific correctness of prediction.

    Evaluates:
    - Overlap of scientific terms (formulas, units, concepts)
    - Semantic similarity of scientific content
    - Presence of correct values and ranges

    Args:
        predicted: Model's prediction
        ground_truth: Ground truth data
        embedding_model: Sentence transformer model

    Returns:
        Score between 0 and 1 (1 = perfect scientific accuracy)
    """
    # Convert to strings
    pred_str = dict_to_string(predicted)
    gt_str = dict_to_string(ground_truth)

    if not pred_str or not gt_str:
        return 0.0

    # Component 1: Scientific term overlap (40%)
    pred_terms = extract_scientific_terms(pred_str)
    gt_terms = extract_scientific_terms(gt_str)

    if gt_terms:
        term_overlap = len(pred_terms & gt_terms) / len(gt_terms)
    else:
        term_overlap = 0.0

    # Component 2: Semantic similarity (40%)
    semantic_sim = calculate_semantic_similarity(pred_str, gt_str, embedding_model)

    # Component 3: Key-value accuracy (20%)
    pred_kv = extract_key_values(predicted)
    gt_kv = extract_key_values(ground_truth)

    if gt_kv:
        matching_keys = sum(1 for k in gt_kv if k in pred_kv)
        kv_accuracy = matching_keys / len(gt_kv)
    else:
        kv_accuracy = 0.0

    # Weighted combination
    score = 0.4 * term_overlap + 0.4 * semantic_sim + 0.2 * kv_accuracy

    return float(np.clip(score, 0, 1))


# ==============================================================================
# Metric 2: Functional Equivalence
# ==============================================================================

def functional_equivalence(predicted: dict, ground_truth: dict,
                          embedding_model: SentenceTransformer) -> float:
    """
    Measure functional similarity (does it achieve the same outcome?).

    Focuses on:
    - Outcome/effect rather than specific process
    - Property changes and their direction
    - Functional goals achieved

    Args:
        predicted: Model's prediction
        ground_truth: Ground truth data
        embedding_model: Sentence transformer model

    Returns:
        Score between 0 and 1 (1 = functionally equivalent)
    """
    # Extract functional aspects
    def extract_functional(data: dict) -> str:
        """Extract outcome-focused information."""
        functional_keys = [
            'doping_outcome', 'carrier_type', 'conductivity', 'property_changes',
            'properties', 'effect', 'result', 'outcome', 'characteristics',
            'predicted_properties', 'electrical_properties', 'optical_properties'
        ]

        functional_text = []

        def search_keys(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    # Check if key is functional
                    if any(fk in k.lower() for fk in functional_keys):
                        if isinstance(v, (str, int, float)):
                            functional_text.append(f"{k}: {v}")
                        elif isinstance(v, dict):
                            functional_text.append(dict_to_string(v))
                    # Recurse
                    if isinstance(v, dict):
                        search_keys(v)

        search_keys(data)
        return " ".join(functional_text)

    pred_functional = extract_functional(predicted)
    gt_functional = extract_functional(ground_truth)

    if not pred_functional or not gt_functional:
        # Fall back to full semantic similarity
        pred_str = dict_to_string(predicted)
        gt_str = dict_to_string(ground_truth)
        if not pred_str or not gt_str:
            return 0.0
        return calculate_semantic_similarity(pred_str, gt_str, embedding_model)

    # Semantic similarity of functional outcomes
    score = calculate_semantic_similarity(pred_functional, gt_functional, embedding_model)

    return float(np.clip(score, 0, 1))


# ==============================================================================
# Metric 3: Reasoning Quality
# ==============================================================================

def reasoning_quality(output: dict) -> float:
    """
    Measure quality of reasoning chain.

    Evaluates:
    - Presence of causal language
    - Mechanism explanations
    - Logical coherence
    - Step-by-step reasoning

    Args:
        output: Model's output (should include 'reasoning' field)

    Returns:
        Score between 0 and 1 (1 = excellent reasoning)
    """
    # Extract reasoning text
    reasoning = ""
    if isinstance(output, dict):
        reasoning = output.get('reasoning', '')
        if not reasoning and 'raw_response' in output:
            reasoning = output.get('raw_response', '')

    if not reasoning or not isinstance(reasoning, str):
        return 0.0

    reasoning = reasoning.lower()

    # Component 1: Causal language (30%)
    causal_keywords = [
        'because', 'therefore', 'thus', 'leads to', 'causes', 'results in',
        'due to', 'consequently', 'as a result', 'since', 'so that',
        'mechanism', 'process', 'pathway'
    ]
    causal_count = sum(1 for kw in causal_keywords if kw in reasoning)
    causal_score = min(1.0, causal_count / 3)  # Expect at least 3 causal markers

    # Component 2: Mechanism explanations (30%)
    mechanism_keywords = [
        'mechanism', 'explain', 'how', 'why', 'interaction', 'bond',
        'structure', 'defect', 'doping', 'substitution', 'electron',
        'carrier', 'energy', 'band'
    ]
    mechanism_count = sum(1 for kw in mechanism_keywords if kw in reasoning)
    mechanism_score = min(1.0, mechanism_count / 4)  # Expect at least 4 mechanism terms

    # Component 3: Length/depth (20%)
    # Longer reasoning tends to be more thorough (but cap at reasonable length)
    word_count = len(reasoning.split())
    depth_score = min(1.0, word_count / 100)  # Expect ~100 words for good reasoning

    # Component 4: Structure (20%)
    # Check for logical connectors and organization
    structure_keywords = [
        'first', 'second', 'third', 'finally', 'additionally', 'moreover',
        'however', 'although', 'in contrast', 'similarly'
    ]
    structure_count = sum(1 for kw in structure_keywords if kw in reasoning)
    structure_score = min(1.0, structure_count / 2)  # Expect at least 2 structure markers

    # Weighted combination
    score = (0.30 * causal_score +
             0.30 * mechanism_score +
             0.20 * depth_score +
             0.20 * structure_score)

    return float(np.clip(score, 0, 1))


# ==============================================================================
# Metric 4: Completeness
# ==============================================================================

def completeness(predicted: dict, expected_fields: Optional[List[str]] = None) -> float:
    """
    Measure completeness of answer.

    Evaluates:
    - Coverage of expected fields
    - Depth of information in each field
    - Presence of all key aspects

    Args:
        predicted: Model's prediction
        expected_fields: List of expected field names (optional)

    Returns:
        Score between 0 and 1 (1 = complete answer)
    """
    if not isinstance(predicted, dict):
        return 0.0

    # Default expected fields for materials processing
    if expected_fields is None:
        expected_fields = [
            'reasoning', 'predicted_properties', 'confidence',
            'suggested_synthesis_conditions', 'properties', 'synthesis',
            'doping_outcome', 'carrier_type', 'method', 'temperature'
        ]

    # Extract all keys from nested dict
    def get_all_keys(d, prefix=''):
        keys = []
        if isinstance(d, dict):
            for k, v in d.items():
                keys.append(f"{prefix}{k}" if prefix else k)
                if isinstance(v, dict):
                    keys.extend(get_all_keys(v, f"{prefix}{k}_" if prefix else f"{k}_"))
        return keys

    all_keys = set(get_all_keys(predicted))

    # Component 1: Field coverage (50%)
    # How many expected fields are present?
    present_fields = sum(1 for ef in expected_fields
                        if any(ef.lower() in k.lower() for k in all_keys))
    field_coverage = present_fields / len(expected_fields) if expected_fields else 0

    # Component 2: Field depth (30%)
    # Are fields populated with meaningful content?
    non_empty_count = 0
    total_count = 0

    def count_non_empty(d):
        nonlocal non_empty_count, total_count
        if isinstance(d, dict):
            for v in d.values():
                total_count += 1
                if v is not None and str(v).lower() not in ["", "not reported", "null", "none"]:
                    non_empty_count += 1
                if isinstance(v, dict):
                    count_non_empty(v)

    count_non_empty(predicted)
    field_depth = non_empty_count / total_count if total_count > 0 else 0

    # Component 3: Overall richness (20%)
    # Total amount of information
    total_keys = len(all_keys)
    richness_score = min(1.0, total_keys / 10)  # Expect at least 10 fields

    # Weighted combination
    score = 0.50 * field_coverage + 0.30 * field_depth + 0.20 * richness_score

    return float(np.clip(score, 0, 1))


# ==============================================================================
# Metric 5: Interpretability
# ==============================================================================

def interpretability(output: dict) -> float:
    """
    Measure how easy to understand the output.

    Evaluates:
    - Structure clarity
    - Presence of explanations
    - Readability
    - Transparency

    Args:
        output: Model's output

    Returns:
        Score between 0 and 1 (1 = highly interpretable)
    """
    if not isinstance(output, dict):
        return 0.0

    # Component 1: Structure clarity (30%)
    # Is the output well-structured with clear sections?
    expected_sections = ['reasoning', 'predicted_properties', 'confidence',
                        'suggested_synthesis_conditions']
    present_sections = sum(1 for sec in expected_sections if sec in output)
    structure_score = present_sections / len(expected_sections)

    # Component 2: Reasoning presence (30%)
    # Does it explain its reasoning?
    has_reasoning = 'reasoning' in output and output['reasoning']
    reasoning_score = 1.0 if has_reasoning else 0.0

    # Component 3: Confidence reporting (20%)
    # Does it report confidence/uncertainty?
    has_confidence = 'confidence' in output and output['confidence'] is not None
    confidence_score = 1.0 if has_confidence else 0.0

    # Component 4: Chain-of-thought transparency (20%)
    # For ARIA_FULL, check if CoT is present
    has_cot = 'chain_of_thought' in output and output['chain_of_thought']

    if has_cot:
        cot_score = 1.0
    elif has_reasoning and output['reasoning']:
        # Even without explicit CoT, good reasoning helps
        reasoning_text = str(output['reasoning'])
        cot_score = min(1.0, len(reasoning_text.split()) / 100)
    else:
        cot_score = 0.0

    # Weighted combination
    score = (0.30 * structure_score +
             0.30 * reasoning_score +
             0.20 * confidence_score +
             0.20 * cot_score)

    return float(np.clip(score, 0, 1))


# ==============================================================================
# Aggregate Metric
# ==============================================================================

def calculate_overall_score(scientific_acc: float, functional_eq: float,
                           reasoning_qual: float, complete: float,
                           interpret: float) -> float:
    """
    Calculate overall score as average of 5 metrics.

    Args:
        scientific_acc: Scientific accuracy score
        functional_eq: Functional equivalence score
        reasoning_qual: Reasoning quality score
        complete: Completeness score
        interpret: Interpretability score

    Returns:
        Overall score (0-1)
    """
    scores = [scientific_acc, functional_eq, reasoning_qual, complete, interpret]
    return float(np.mean(scores))


# ==============================================================================
# Comprehensive Evaluation Function
# ==============================================================================

def evaluate_prediction(predicted: dict, ground_truth: dict,
                       embedding_model: SentenceTransformer) -> dict:
    """
    Evaluate a single prediction on all 5 metrics.

    Args:
        predicted: Model's prediction
        ground_truth: Ground truth data
        embedding_model: Sentence transformer model

    Returns:
        Dictionary with all metric scores
    """
    sci_acc = scientific_accuracy(predicted, ground_truth, embedding_model)
    func_eq = functional_equivalence(predicted, ground_truth, embedding_model)
    reason_qual = reasoning_quality(predicted)
    complete_score = completeness(predicted)
    interp = interpretability(predicted)
    overall = calculate_overall_score(sci_acc, func_eq, reason_qual,
                                     complete_score, interp)

    return {
        'scientific_accuracy': sci_acc,
        'functional_equivalence': func_eq,
        'reasoning_quality': reason_qual,
        'completeness': complete_score,
        'interpretability': interp,
        'overall': overall
    }


if __name__ == '__main__':
    # Test metrics with sample data
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Sample prediction
    predicted = {
        'reasoning': 'The CVD process at high temperature causes MoS2 to form with Nb dopant substitution. This leads to n-type conductivity because Nb donates electrons.',
        'predicted_properties': {
            'doping_outcome': 'n-type conductivity',
            'carrier_type': 'electron',
            'conductivity': 'enhanced'
        },
        'confidence': 0.85
    }

    # Sample ground truth
    ground_truth = {
        'property_changes': {
            'doping_outcome': 'n-type doping',
            'carrier_concentration': 'increased',
            'conductivity_type': 'electron-based'
        }
    }

    # Evaluate
    scores = evaluate_prediction(predicted, ground_truth, model)

    print("Metric Scores:")
    print("="*50)
    for metric, score in scores.items():
        print(f"{metric:30s}: {score:.3f}")
