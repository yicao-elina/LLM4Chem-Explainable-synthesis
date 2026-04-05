"""
IMPROVED ARIA Evaluation Metrics - Physics-Guided Assessment

This module provides enhanced metrics that:
1. Properly handle ARIA's structured reasoning output
2. Assess physics-guided reasoning, not just narrative structure
3. Evaluate KG grounding and literature validation
4. Check tier-confidence calibration
5. Measure mechanistic soundness

Author: ARIA Team
Date: 2026-02-07
"""

import re
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================================================================
# Enhanced Metric 3: Reasoning Quality v2 (Handles Both Formats)
# ==============================================================================

def reasoning_quality_v2(output: dict) -> float:
    """
    IMPROVED: Measure quality of reasoning chain.
    
    Now handles BOTH:
    - Narrative reasoning (Baseline style)
    - Structured reasoning (ARIA style with mechanistic_explanation)
    
    Evaluates:
    - Presence of causal logic (narrative or structural)
    - Mechanism explanations
    - Step-by-step reasoning
    - Proper tier-confidence calibration
    
    Args:
        output: Model's output
    
    Returns:
        Score between 0 and 1
    """
    
    # ==========================================================================
    # PART A: ARIA-Style Structured Reasoning
    # ==========================================================================
    
    mech_data = output.get('mechanistic_explanation', None)
    # If missing or None, treat as empty narrative
    if mech_data is None:
        return 0.0
    if isinstance(mech_data, dict):
        # Score ARIA's structured mechanistic reasoning
        components_present = 0
        components_total = 4
        # 1. Primary mechanism explanation (30%)
        primary_mech = mech_data.get('primary_mechanism', '')
        if primary_mech and isinstance(primary_mech, str) and len(primary_mech) > 20:
            primary_score = 1.0
            components_present += 1
        else:
            primary_score = 0.0
        # 2. Chain of thought steps (30%)
        cot = mech_data.get('chain_of_thought', [])
        if isinstance(cot, list) and len(cot) >= 2:
            cot_score = min(1.0, len(cot) / 5)
            components_present += 1
        else:
            cot_score = 0.0
        # 3. Quantitative estimates (20%)
        quantitative = mech_data.get('quantitative_estimates', {})
        if quantitative and isinstance(quantitative, dict) and len(quantitative) > 0:
            quant_score = 1.0
            components_present += 1
        else:
            quant_score = 0.0
        # 4. Alternative mechanisms (20%)
        alternatives = mech_data.get('alternative_mechanisms', '')
        if alternatives and isinstance(alternatives, str) and len(alternatives) > 10:
            alt_score = 1.0
            components_present += 1
        else:
            alt_score = 0.0
        # Weighted score for structured reasoning
        structured_score = (0.30 * primary_score + 
                           0.30 * cot_score + 
                           0.20 * quant_score + 
                           0.20 * alt_score)
        return float(np.clip(structured_score, 0, 1))
    elif isinstance(mech_data, str):
        # If mechanistic_explanation is a string, treat as narrative reasoning
        reasoning = mech_data
        reasoning_lower = reasoning.lower()
        causal_keywords = [
            'because', 'therefore', 'thus', 'leads to', 'causes', 'results in',
            'due to', 'consequently', 'as a result', 'since', 'so that',
            'mechanism', 'process', 'pathway'
        ]
        causal_count = sum(1 for kw in causal_keywords if kw in reasoning_lower)
        causal_score = min(1.0, causal_count / 3)
        mechanism_keywords = [
            'mechanism', 'explain', 'how', 'why', 'interaction', 'bond',
            'structure', 'defect', 'doping', 'substitution', 'electron',
            'carrier', 'energy', 'band'
        ]
        mechanism_count = sum(1 for kw in mechanism_keywords if kw in reasoning_lower)
        mechanism_score = min(1.0, mechanism_count / 4)
        word_count = len(reasoning_lower.split())
        depth_score = min(1.0, word_count / 100)
        structure_keywords = [
            'first', 'second', 'third', 'finally', 'additionally', 'moreover',
            'however', 'although', 'in contrast', 'similarly'
        ]
        structure_count = sum(1 for kw in structure_keywords if kw in reasoning_lower)
        structure_score = min(1.0, structure_count / 2)
        narrative_score = (0.30 * causal_score +
                          0.30 * mechanism_score +
                          0.20 * depth_score +
                          0.20 * structure_score)
        return float(np.clip(narrative_score, 0, 1))
    else:
        # Unexpected type, treat as empty narrative
        return 0.0
    
    # ==========================================================================
    # PART B: Narrative Reasoning (Baseline style)
    # ==========================================================================
    
    reasoning = output.get('reasoning', '')
    if not reasoning or not isinstance(reasoning, str):
        return 0.0
    
    reasoning_lower = reasoning.lower()
    
    # Component 1: Causal language (30%)
    causal_keywords = [
        'because', 'therefore', 'thus', 'leads to', 'causes', 'results in',
        'due to', 'consequently', 'as a result', 'since', 'so that',
        'mechanism', 'process', 'pathway'
    ]
    causal_count = sum(1 for kw in causal_keywords if kw in reasoning_lower)
    causal_score = min(1.0, causal_count / 3)
    
    # Component 2: Mechanism keywords (30%)
    mechanism_keywords = [
        'mechanism', 'explain', 'how', 'why', 'interaction', 'bond',
        'structure', 'defect', 'doping', 'substitution', 'electron',
        'carrier', 'energy', 'band'
    ]
    mechanism_count = sum(1 for kw in mechanism_keywords if kw in reasoning_lower)
    mechanism_score = min(1.0, mechanism_count / 4)
    
    # Component 3: Length/depth (20%)
    word_count = len(reasoning_lower.split())
    depth_score = min(1.0, word_count / 100)
    
    # Component 4: Structure (20%)
    structure_keywords = [
        'first', 'second', 'third', 'finally', 'additionally', 'moreover',
        'however', 'although', 'in contrast', 'similarly'
    ]
    structure_count = sum(1 for kw in structure_keywords if kw in reasoning_lower)
    structure_score = min(1.0, structure_count / 2)
    
    # Weighted combination for narrative
    narrative_score = (0.30 * causal_score +
                      0.30 * mechanism_score +
                      0.20 * depth_score +
                      0.20 * structure_score)
    
    return float(np.clip(narrative_score, 0, 1))


# ==============================================================================
# NEW Metric: Tier-Confidence Calibration
# ==============================================================================

def tier_confidence_calibration(output: dict) -> float:
    """
    Measure whether confidence is appropriately calibrated to reasoning tier.
    
    ARIA's tier system:
    - Tier 1 (direct KG match): confidence should be high (0.7-1.0)
    - Tier 2 (transfer learning): confidence should be moderate (0.5-0.8)
    - Tier 3 (LLM fallback): confidence should be lower (0.3-0.6)
    
    Also penalizes over/under-confidence.
    
    Args:
        output: Model's output with 'tier' and 'confidence' fields
    
    Returns:
        Score between 0 and 1
    """
    if 'tier' not in output or 'confidence' not in output:
        return 0.0
    
    tier = output.get('tier')
    confidence = output.get('confidence', 0.5)
    
    # Expected confidence ranges by tier
    tier_ranges = {
        1: (0.7, 1.0),      # Direct match: high confidence
        2: (0.5, 0.8),      # Transfer: moderate-high confidence
        3: (0.2, 0.6)       # Fallback: lower confidence
    }
    
    if tier not in tier_ranges:
        return 0.5  # Unknown tier
    
    min_conf, max_conf = tier_ranges[tier]
    
    # Check if confidence is in appropriate range
    if min_conf <= confidence <= max_conf:
        return 1.0
    
    # Penalize based on how far from range
    if confidence > max_conf:
        # Over-confident for tier
        excess = confidence - max_conf
        return max(0.0, 1.0 - excess)
    else:
        # Under-confident for tier
        deficit = min_conf - confidence
        return max(0.0, 1.0 - deficit)


# ==============================================================================
# NEW Metric: KG Grounding Quality
# ==============================================================================

def kg_grounding_quality(output: dict) -> float:
    """
    Measure how well predictions are grounded in knowledge graph.
    
    Evaluates:
    - Whether KG paths were used
    - Tier level (Tier 1 = best grounding)
    - Number of KG edges traversed
    - Presence of mechanistic explanation from KG
    
    Args:
        output: Model's output
    
    Returns:
        Score between 0 and 1 (1 = excellent KG grounding)
    """
    
    # Component 1: KG usage (40%)
    kg_paths = output.get('kg_paths_used', 0)
    kg_edges = output.get('kg_edges_used', 0)
    
    if kg_paths > 0 or kg_edges > 0:
        kg_usage_score = 1.0
    else:
        kg_usage_score = 0.0
    
    # Component 2: Tier level (40%)
    # Tier 1 (direct) > Tier 2 (transfer) > Tier 3 (fallback)
    tier = output.get('tier', 3)
    tier_scores = {
        1: 1.0,      # Direct KG match: best
        2: 0.7,      # Transfer learning: decent
        3: 0.0       # Pure LLM: no KG grounding
    }
    tier_score = tier_scores.get(tier, 0.0)
    
    # Component 3: Mechanistic explanation (20%)
    has_mechanism = False
    if 'mechanistic_explanation' in output:
        mech = output['mechanistic_explanation']
        if isinstance(mech, dict) and mech.get('primary_mechanism'):
            has_mechanism = True
    
    mech_score = 1.0 if has_mechanism else 0.0
    
    # Weighted combination
    score = (0.40 * kg_usage_score +
             0.40 * tier_score +
             0.20 * mech_score)
    
    return float(np.clip(score, 0, 1))


# ==============================================================================
# NEW Metric: Literature Grounding Quality
# ==============================================================================

def literature_grounding_quality(output: dict) -> float:
    """
    Measure how well predictions are validated by literature.
    
    Evaluates:
    - Number of literature papers cited
    - Literature search was performed
    - Citation alignment with prediction
    
    Args:
        output: Model's output
    
    Returns:
        Score between 0 and 1 (1 = excellent literature grounding)
    """
    
    # Check if literature search was performed
    literature_papers = output.get('literature_papers', 0)
    
    if literature_papers <= 0:
        # No literature search: 0.5 (neutral, could be valid for KG-only reasoning)
        return 0.5
    
    # Score based on number of papers
    # 1-3 papers = 0.6, 4-6 papers = 0.8, 7+ papers = 1.0
    if literature_papers >= 7:
        return 1.0
    elif literature_papers >= 4:
        return 0.8
    elif literature_papers >= 1:
        return 0.6
    else:
        return 0.5


# ==============================================================================
# NEW Metric: Mechanistic Soundness
# ==============================================================================

# ==============================================================================
# NEW Metric: Physics-Guided Reasoning Score (PGRS)
# ==============================================================================

def physics_guided_reasoning_score(output: dict, variant: str, weights: dict) -> dict:
    """
    Calculates the Physics-Guided Reasoning Score (PGRS).

    This score evaluates reasoning based on three pillars of scientific AI:
    - Causal Coherence (C): Adherence to the Processing-Structure-Property chain.
    - Source Grounding (G): Use of external KG and literature evidence.
    - Internal Validity (V): The soundness and detail of the explanation itself.

    Args:
        output: The model's prediction output dictionary.
        variant: The name of the ARIA variant being tested.
        weights: A dictionary containing weights for {coherence, grounding, validity}.

    Returns:
        A dictionary with the sub-scores and the final PGRS.
    """
    # 1. Causal Coherence (C) - Based on ARIA's reasoning tier
    tier = output.get('tier', 3)
    if tier == 1:
        coherence_score = 1.0  # Direct causal path
    elif tier == 2:
        coherence_score = 0.7  # Analogy-based transfer
    else:
        coherence_score = 0.3  # LLM fallback

    # 2. Source Grounding (G) - Adapts to variant capabilities
    kg_usage = 1.0 if output.get('kg_paths_used', 0) > 0 else 0.0
    lit_usage = 1.0 if output.get('literature_papers', 0) > 0 else 0.0
    variant_lower = variant.lower() if variant else ''

    if 'full' in variant_lower or 'search' in variant_lower:
        grounding_score = 0.6 * kg_usage + 0.4 * lit_usage
    elif 'core' in variant_lower:
        grounding_score = kg_usage
    else:  # Baseline or unknown
        grounding_score = 0.0

    # 3. Internal Validity (V) - Quality of the explanation
    mech_data = output.get('mechanistic_explanation', {})
    reasoning_text = output.get('reasoning', '')
    
    # Prefer structured explanation if available
    if isinstance(mech_data, dict) and mech_data:
        has_primary = 1 if mech_data.get('primary_mechanism') and len(str(mech_data.get('primary_mechanism'))) > 10 else 0
        has_cot = 1 if mech_data.get('chain_of_thought') and isinstance(mech_data.get('chain_of_thought'), list) and len(mech_data.get('chain_of_thought')) > 1 else 0
        has_quant = 1 if mech_data.get('quantitative_estimates') and isinstance(mech_data.get('quantitative_estimates'), dict) and mech_data.get('quantitative_estimates') else 0
        validity_score = (has_primary + has_cot + has_quant) / 3.0
    # Fallback to narrative reasoning text
    elif isinstance(reasoning_text, str) and reasoning_text:
        word_count = len(reasoning_text.split())
        causal_kw = ['because', 'therefore', 'leads to', 'causes', 'results in', 'due to']
        causal_count = sum(1 for kw in causal_kw if kw in reasoning_text.lower())
        validity_score = np.clip((word_count / 150.0) * (0.5 + 0.5 * min(1.0, causal_count / 2.0)), 0, 1)
    else:
        validity_score = 0.0
    
    # Final Weighted Score Calculation
    w_c = weights.get('coherence', 0.4)
    w_g = weights.get('grounding', 0.4)
    w_v = weights.get('validity', 0.2)
    
    final_score = (w_c * coherence_score + 
                   w_g * grounding_score + 
                   w_v * validity_score)

    return {
        'pgrs_coherence': coherence_score,
        'pgrs_grounding': grounding_score,
        'pgrs_validity': validity_score,
        'physics_guided_reasoning_score': np.clip(final_score, 0, 1)
    }


# ==============================================================================
# IMPROVED: Overall Score Calculation
# ==============================================================================

def calculate_overall_score_v2(
    scientific_acc: float,
    functional_eq: float,
    reasoning_qual: float,
    completeness: float,
    interpretability: float,
    kg_grounding: Optional[float] = None,
    literature_grounding: Optional[float] = None,
    tier_calibration: Optional[float] = None,
    pgrs_scores: Optional[dict] = None
) -> Dict[str, float]:
    """
    Calculate overall score with optional enhanced metrics.
    
    Args:
        Standard 5 metrics (0-1)
        Optional enhanced metrics specific to ARIA, including PGRS.
    
    Returns:
        Dict with component scores and overall score.
    """
    
    results = {
        'scientific_accuracy': scientific_acc,
        'functional_equivalence': functional_eq,
        'reasoning_quality': reasoning_qual,
        'completeness': completeness,
        'interpretability': interpretability
    }
    
    # Calculate base score (original 5 metrics)
    base_metrics = [scientific_acc, functional_eq, reasoning_qual, completeness, interpretability]
    results['base_score'] = float(np.mean(base_metrics))
    
    # Add enhanced metrics if provided
    enhanced_metrics = []
    
    if kg_grounding is not None:
        results['kg_grounding'] = kg_grounding
    
    if literature_grounding is not None:
        results['literature_grounding'] = literature_grounding
        enhanced_metrics.append(literature_grounding)
    
    if tier_calibration is not None:
        results['tier_calibration'] = tier_calibration
        enhanced_metrics.append(tier_calibration)
    
    if pgrs_scores:
        results.update(pgrs_scores)
        if 'physics_guided_reasoning_score' in pgrs_scores:
            enhanced_metrics.append(pgrs_scores['physics_guided_reasoning_score'])

    # Overall score: weighted average of base + enhanced metrics
    if enhanced_metrics:
        # Weight: base metrics 50%, enhanced metrics 50%
        base_weight = 0.5
        enhanced_weight = 0.5
        
        results['overall_score'] = (base_weight * results['base_score'] + 
                                   enhanced_weight * np.mean(enhanced_metrics))
    else:
        results['overall_score'] = results['base_score']
    
    return results


# ==============================================================================
# Updated Evaluation Function
# ==============================================================================

def evaluate_prediction_v2(
    predicted: dict,
    ground_truth: dict,
    embedding_model: SentenceTransformer,
    kg: Optional[nx.DiGraph] = None,
    use_enhanced_metrics: bool = True,
    variant: Optional[str] = None,
    metric_config: Optional[dict] = None
) -> Dict[str, float]:
    """
    Evaluate a prediction using all available metrics.
    
    Args:
        predicted: Model's prediction.
        ground_truth: Ground truth data.
        embedding_model: Sentence transformer model.
        kg: Optional knowledge graph for grounding assessment.
        use_enhanced_metrics: Whether to compute enhanced ARIA-specific metrics.
        variant: The name of the variant being tested.
        metric_config: Configuration for the metrics, e.g., PGRS weights.
    
    Returns:
        Dict with all metric scores.
    """
    # Import original metrics (assumed available)
    from evaluation.metrics import (
        scientific_accuracy, 
        functional_equivalence, 
        completeness, 
        interpretability
    )
    
    # Compute standard metrics
    sci_acc = scientific_accuracy(predicted, ground_truth, embedding_model)
    func_eq = functional_equivalence(predicted, ground_truth, embedding_model)
    reason_qual = reasoning_quality_v2(predicted)  # Use improved version
    complete_score = completeness(predicted)
    interp = interpretability(predicted)
    
    # Compute enhanced metrics if requested
    kg_ground = None
    lit_ground = None
    tier_calib = None
    pgrs_scores = None # Initialize pgrs_scores here
    
    if use_enhanced_metrics:
        kg_ground = kg_grounding_quality(predicted)
        lit_ground = literature_grounding_quality(predicted)
        tier_calib = tier_confidence_calibration(predicted)
        
        if variant and metric_config:
            pgrs_config = metric_config.get('physics_guided_reasoning_score', {})
            variation_name = pgrs_config.get('active_variation', 'default')
            weights = pgrs_config.get('variations', {}).get(variation_name, {}).get('weights', {})
            pgrs_scores = physics_guided_reasoning_score(predicted, variant, weights) # <--- CALL NEW FUNCTION
    
    # Calculate overall score
    results = calculate_overall_score_v2(
        sci_acc, func_eq, reason_qual, complete_score, interp,
        kg_ground, lit_ground, tier_calib, pgrs_scores # <--- PASS PGRS_SCORES
    )
    
    return results


if __name__ == '__main__':
    # Test the improved metrics
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test ARIA-style output
    aria_output = {
        'tier': 1,
        'confidence': 0.85,
        'kg_paths_used': 1,
        'literature_papers': 3,
        'mechanistic_explanation': {
            'primary_mechanism': 'CVD process at high temperature enables Nb substitution into MoS2, which acts as electron donor increasing n-type carrier concentration',
            'chain_of_thought': [
                'CVD provides thermal energy for Nb-MoS2 interaction',
                'Nb substitution occurs at Mo lattice sites',
                'Nb has one more valence electron than Mo',
                'Extra electron becomes free carrier in conduction band',
                'Results in n-type conductivity enhancement'
            ],
            'quantitative_estimates': {
                'carrier_concentration': '1e12 to 1e13 cm-3',
                'mobility': '10-50 cm2/Vs'
            },
            'alternative_mechanisms': 'Nb could also create acceptor states via interstitial insertion, but substitution is more favorable thermodynamically'
        },
        'predicted_properties': {
            'carrier_type': 'n-type',
            'conductivity': 'enhanced',
            'mobility': 'moderate'
        }
    }
    
    # Test Baseline-style output
    baseline_output = {
        'reasoning': 'Based on fundamental materials science principles, CVD at high temperature provides thermal energy for atomic diffusion. Nb dopant substituting for Mo in MoS2 adds donor states. Since Nb has one more valence electron than Mo, the extra electron becomes a free carrier. This mechanism therefore leads to n-type conductivity enhancement. The mechanism explains why this doping approach is effective for increasing carrier concentration and mobility.',
        'predicted_properties': {
            'carrier_type': 'n-type',
            'conductivity': 'enhanced'
        },
        'confidence': 0.75,
        'tier': 3
    }

    # Define some dummy weights for testing PGRS
    test_weights = {'coherence': 0.4, 'grounding': 0.4, 'validity': 0.2}

    print("\nARIA Output Evaluation:")
    print("="*60)
    print(f"reasoning_quality_v2: {reasoning_quality_v2(aria_output):.3f}")
    print(f"tier_confidence_calibration: {tier_confidence_calibration(aria_output):.3f}")
    print(f"kg_grounding_quality: {kg_grounding_quality(aria_output):.3f}")
    print(f"literature_grounding_quality: {literature_grounding_quality(aria_output):.3f}")
    pgrs_aria = physics_guided_reasoning_score(aria_output, "ARIA_FULL", test_weights)
    print(f"PGRS (ARIA_FULL): {pgrs_aria['physics_guided_reasoning_score']:.3f} (C:{pgrs_aria['pgrs_coherence']:.2f} G:{pgrs_aria['pgrs_grounding']:.2f} V:{pgrs_aria['pgrs_validity']:.2f})")
    
    print("\n\nBaseline Output Evaluation:")
    print("="*60)
    print(f"reasoning_quality_v2: {reasoning_quality_v2(baseline_output):.3f}")
    print(f"tier_confidence_calibration: {tier_confidence_calibration(baseline_output):.3f}")
    print(f"kg_grounding_quality: {kg_grounding_quality(baseline_output):.3f}")
    print(f"literature_grounding_quality: {literature_grounding_quality(baseline_output):.3f}")
    pgrs_baseline = physics_guided_reasoning_score(baseline_output, "BASELINE", test_weights)
    print(f"PGRS (BASELINE): {pgrs_baseline['physics_guided_reasoning_score']:.3f} (C:{pgrs_baseline['pgrs_coherence']:.2f} G:{pgrs_baseline['pgrs_grounding']:.2f} V:{pgrs_baseline['pgrs_validity']:.2f})")
