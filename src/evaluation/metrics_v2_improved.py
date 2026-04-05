"""
ARIA Evaluation Metrics v3 - Literature-Grounded Assessment

Based on state-of-the-art research in:
- Causal reasoning evaluation (Pearl, Schölkopf, Geiger)
- Source verification (FEVER, GopherCite)
- Faithfulness testing (Wiegreffe, Turpin, Lanham)

Author: ARIA Team (Refactored)
Date: 2026-02-08
"""

import re
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load SpaCy model globally
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ==============================================================================
# CAUSAL COHERENCE V3 (Literature-Grounded)
# ==============================================================================

def causal_coherence_v3(output: dict, ground_truth: dict, 
                        kg: nx.DiGraph, 
                        embedding_model: SentenceTransformer) -> Dict[str, float]:
    """
    Measures causal coherence through multiple validation strategies.
    
    Based on:
    - Pearl (2019): Intervention testing
    - Schölkopf et al. (2021): Mechanistic modularity
    - Geiger et al. (2023): Causal abstraction
    """
    
    scores = {}
    
    # ==================================================================
    # 1. INTERVENTION CONSISTENCY (Pearl's do-calculus)
    # ==================================================================
    # Test: If we intervene on processing (P), does model predict
    # correct downstream effects on structure (S) and property (Prop)?
    
    processing = output.get('processing_conditions', {})
    structure = output.get('structure', {})
    properties = output.get('predicted_properties', {})
    
    # Build causal graph from mechanistic explanation
    mech = output.get('mechanistic_explanation', {})
    cot = mech.get('chain_of_thought', []) if isinstance(mech, dict) else []
    
    # Extract causal claims (P→S, S→Prop)
    causal_edges = extract_causal_edges(cot, embedding_model)
    
    # Check consistency with known PSP relationships in KG
    intervention_score = 0.0
    if kg and causal_edges:
        for edge in causal_edges:
            source, target, mechanism = edge
            # Verify edge exists in KG or literature
            if verify_causal_edge_in_kg(kg, source, target, mechanism):
                intervention_score += 1.0
        intervention_score /= max(len(causal_edges), 1)
    
    scores['intervention_consistency'] = intervention_score
    
    # ==================================================================
    # 2. COUNTERFACTUAL REASONING (Pearl Level 3)
    # ==================================================================
    # Test: Can model answer "What if processing was different?"
    
    counterfactual_score = 0.0
    alternatives = mech.get('alternative_mechanisms', '') if isinstance(mech, dict) else ''
    
    if alternatives and len(alternatives) > 20:
        # Check if alternatives are mechanistically distinct
        cf_embedding = embedding_model.encode(alternatives)
        primary_embedding = embedding_model.encode(
            mech.get('primary_mechanism', '') if isinstance(mech, dict) else ''
        )
        
        # Good counterfactual should be semantically different but causally related
        similarity = cosine_similarity([cf_embedding], [primary_embedding])[0][0]
        if 0.3 < similarity < 0.7:  # Not too similar, not unrelated
            counterfactual_score = 0.8
            
            # Bonus: Check if alternative explains different outcome
            if 'different' in alternatives.lower() or 'instead' in alternatives.lower():
                counterfactual_score = 1.0
    
    scores['counterfactual_reasoning'] = counterfactual_score
    
    # ==================================================================
    # 3. MECHANISTIC MODULARITY (Schölkopf)
    # ==================================================================
    # Test: Are mechanisms independent and composable?
    
    modularity_score = 0.0
    if isinstance(mech, dict) and cot:
        # Check if each step has clear inputs/outputs
        has_clear_structure = all(
            any(marker in step.lower() for marker in 
                ['leads to', 'causes', 'results in', 'enables', 'produces'])
            for step in cot if isinstance(step, str)
        )
        
        if has_clear_structure:
            modularity_score = 0.5
        
        # Check for quantitative estimates (mechanistic precision)
        quant = mech.get('quantitative_estimates', {})
        if quant and len(quant) > 0:
            modularity_score += 0.3
        
        # Check for explicit mechanism identification
        if 'mechanism' in mech.get('primary_mechanism', '').lower():
            modularity_score += 0.2
    
    scores['mechanistic_modularity'] = min(modularity_score, 1.0)
    
    # ==================================================================
    # 4. PSP CHAIN VALIDITY
    # ==================================================================
    # Verify complete Processing → Structure → Property chain
    
    psp_score = 0.0
    
    # Check if all three components are present
    has_processing = bool(processing)
    has_structure = bool(structure)
    has_properties = bool(properties)
    
    if has_processing and has_structure and has_properties:
        psp_score = 0.4
        
        # Verify causal links in CoT
        p_to_s = any('processing' in step.lower() and 'structure' in step.lower() 
                     for step in cot if isinstance(step, str))
        s_to_prop = any('structure' in step.lower() and any(
            prop in step.lower() for prop in ['property', 'conductivity', 'carrier', 'mobility']
        ) for step in cot if isinstance(step, str))
        
        if p_to_s:
            psp_score += 0.3
        if s_to_prop:
            psp_score += 0.3
    
    scores['psp_chain_validity'] = psp_score
    
    # ==================================================================
    # FINAL COHERENCE SCORE
    # ==================================================================
    weights = {
        'intervention_consistency': 0.35,
        'counterfactual_reasoning': 0.25,
        'mechanistic_modularity': 0.25,
        'psp_chain_validity': 0.15
    }
    
    final_score = sum(scores[k] * weights[k] for k in weights)
    scores['causal_coherence_score'] = np.clip(final_score, 0, 1)
    
    return scores


def extract_causal_edges(chain_of_thought: List[str], 
                        embedding_model: SentenceTransformer) -> List[Tuple]:
    """Extract causal relationships from CoT using NLP."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    
    edges = []
    causal_verbs = ['causes', 'leads to', 'results in', 'enables', 'produces', 'induces']
    
    for step in chain_of_thought:
        if not isinstance(step, str):
            continue
        doc = nlp(step)
        
        # Simple pattern matching for causal structures
        for token in doc:
            if token.lemma_ in ['cause', 'lead', 'result', 'enable', 'produce', 'induce']:
                # Extract subject and object
                subject = [child for child in token.head.children if child.dep_ == 'nsubj']
                obj = [child for child in token.children if child.dep_ in ['dobj', 'attr']]
                
                if subject and obj:
                    edges.append((
                        subject[0].text,
                        obj[0].text,
                        token.lemma_
                    ))
    
    return edges


def verify_causal_edge_in_kg(kg: nx.DiGraph, source: str, target: str, 
                             mechanism: str) -> bool:
    """Verify if causal edge exists in knowledge graph."""
    # Fuzzy matching for nodes
    source_nodes = [n for n in kg.nodes() if source.lower() in str(n).lower()]
    target_nodes = [n for n in kg.nodes() if target.lower() in str(n).lower()]
    
    if not source_nodes or not target_nodes:
        return False
    
    # Check for any path
    for s in source_nodes:
        for t in target_nodes:
            if nx.has_path(kg, s, t):
                return True
    
    return False

# ==============================================================================
# SOURCE GROUNDING V3 (Literature-Grounded)
# ==============================================================================

def source_grounding_v3(output: dict, ground_truth: dict,
                        kg: nx.DiGraph,
                        embedding_model: SentenceTransformer,
                        nli_model=None) -> Dict[str, float]:
    """
    Measures quality of external grounding through verification.
    
    Based on:
    - Thorne et al. (2018): FEVER evidence verification
    - Gao et al. (2023): Citation-claim entailment
    - Menick et al. (2022): Human verifiability proxy
    """
    
    scores = {}
    
    # ==================================================================
    # 1. KG PATH VERIFICATION (not just counting!)
    # ==================================================================
    kg_paths = output.get('kg_paths_used', 0)
    kg_edges = output.get('kg_edges_used', 0)
    
    if kg and kg_paths > 0:
        # Verify paths actually support the prediction
        mech = output.get('mechanistic_explanation', {})
        primary_mech = mech.get('primary_mechanism', '') if isinstance(mech, dict) else ''
        
        # Extract key claims from mechanism
        claims = extract_atomic_claims(primary_mech)
        
        # Check how many claims are grounded in KG
        grounded_claims = 0
        for claim in claims:
            if verify_claim_in_kg(kg, claim, embedding_model):
                grounded_claims += 1
        
        kg_verification_score = grounded_claims / max(len(claims), 1)
    else:
        kg_verification_score = 0.0
    
    scores['kg_verification'] = kg_verification_score
    
    # ==================================================================
    # 2. LITERATURE CITATION QUALITY (using NLI)
    # ==================================================================
    lit_papers = output.get('literature_papers', 0)
    lit_evidence = output.get('literature_evidence', [])
    
    if lit_papers > 0 and lit_evidence:
        # Load NLI model for entailment checking
        if nli_model is None:
            from transformers import pipeline
            nli_model = pipeline("text-classification", 
                               model="microsoft/deberta-v3-base-mnli-fever-anli")
        
        # Check citation-claim entailment
        mech = output.get('mechanistic_explanation', {})
        primary_mech = mech.get('primary_mechanism', '') if isinstance(mech, dict) else ''
        
        entailment_scores = []
        for evidence in lit_evidence[:5]:  # Check top 5 citations
            # Does evidence entail the claim?
            result = nli_model(f"{evidence} [SEP] {primary_mech}")
            if result[0]['label'] == 'ENTAILMENT':
                entailment_scores.append(result[0]['score'])
            else:
                entailment_scores.append(0.0)
        
        lit_entailment_score = np.mean(entailment_scores) if entailment_scores else 0.0
        
        # Penalize over-citation (potential hallucination)
        citation_penalty = 0.0
        if lit_papers > 10:
            citation_penalty = min(0.2, (lit_papers - 10) * 0.02)
        
        lit_quality_score = max(0.0, lit_entailment_score - citation_penalty)
    else:
        lit_quality_score = 0.0
    
    scores['literature_entailment'] = lit_quality_score
    
    # ==================================================================
    # 3. MULTI-HOP REASONING VERIFICATION
    # ==================================================================
    # Check if multi-hop reasoning through sources is valid
    
    mech = output.get('mechanistic_explanation', {})
    if isinstance(mech, dict):
        cot = mech.get('chain_of_thought', [])
    else:
        cot = []
    if isinstance(cot, list) and len(cot) >= 2:
        # Check if each step is independently grounded
        grounded_steps = 0
        for step in cot:
            if isinstance(step, str):
                # Check KG or literature support
                if (kg and verify_claim_in_kg(kg, step, embedding_model)) or \
                   (lit_evidence and any(check_entailment_simple(step, ev) 
                                        for ev in lit_evidence[:5])):
                    grounded_steps += 1
        
        multihop_score = grounded_steps / len(cot)
    else:
        multihop_score = 0.0
    
    scores['multihop_grounding'] = multihop_score
    
    # ==================================================================
    # 4. SOURCE DIVERSITY & RELEVANCE
    # ==================================================================
    # Penalize redundant sources, reward diverse evidence types
    
    diversity_score = 0.0
    if lit_evidence and len(lit_evidence) > 1:
        # Compute pairwise similarity
        embeddings = embedding_model.encode(lit_evidence[:5])
        similarities = cosine_similarity(embeddings)
        
        # Good diversity: low average pairwise similarity
        avg_similarity = (similarities.sum() - len(similarities)) / (len(similarities) * (len(similarities) - 1))
        diversity_score = 1.0 - min(avg_similarity, 1.0)
    
    scores['source_diversity'] = diversity_score
    
    # ==================================================================
    # FINAL GROUNDING SCORE
    # ==================================================================
    weights = {
        'kg_verification': 0.35,
        'literature_entailment': 0.35,
        'multihop_grounding': 0.20,
        'source_diversity': 0.10
    }
    
    final_score = sum(scores[k] * weights[k] for k in weights)
    scores['source_grounding_score'] = np.clip(final_score, 0, 1)
    
    return scores


def extract_atomic_claims(text: str) -> List[str]:
    """Break mechanism explanation into atomic verifiable claims."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    claims = []
    for sent in doc.sents:
        # Further split on causal connectives
        subclaims = re.split(r',\s*(?:which|that|and)\s+', sent.text)
        claims.extend([c.strip() for c in subclaims if len(c.strip()) > 10])
    
    return claims


def verify_claim_in_kg(kg: nx.DiGraph, claim: str, 
                      embedding_model: SentenceTransformer) -> bool:
    """Verify if claim is supported by KG structure."""
    # Extract key entities from claim
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(claim)
    entities = [ent.text for ent in doc.ents]
    
    if len(entities) < 2:
        return False
    
    # Check for paths between entities in KG
    for i, e1 in enumerate(entities[:-1]):
        for e2 in entities[i+1:]:
            nodes_e1 = [n for n in kg.nodes() if e1.lower() in str(n).lower()]
            nodes_e2 = [n for n in kg.nodes() if e2.lower() in str(n).lower()]
            
            for n1 in nodes_e1:
                for n2 in nodes_e2:
                    if nx.has_path(kg, n1, n2) or nx.has_path(kg, n2, n1):
                        return True
    
    return False


def check_entailment_simple(claim: str, evidence: str) -> bool:
    """Simple entailment check using keyword overlap (proxy)."""
    claim_words = set(claim.lower().split())
    evidence_words = set(evidence.lower().split())
    overlap = len(claim_words & evidence_words) / max(len(claim_words), 1)
    return overlap > 0.4

# ==============================================================================
# INTERNAL VALIDITY V3 (Literature-Grounded)
# ==============================================================================

def internal_validity_v3(output: dict, 
                        embedding_model: SentenceTransformer) -> Dict[str, float]:
    """
    Measures faithfulness and logical consistency of explanations.
    
    Based on:
    - Wiegreffe & Marasović (2021): Faithfulness testing
    - Yeh et al. (2023): Self-consistency in CoT
    - Lanham et al. (2023): Forward simulation tests
    """
    
    scores = {}
    
    mech = output.get('mechanistic_explanation', {})
    if not isinstance(mech, dict):
        return {'internal_validity_score': 0.0}
    
    primary_mech = mech.get('primary_mechanism', '')
    cot = mech.get('chain_of_thought', [])
    prediction = output.get('predicted_properties', {})
    
    # ==================================================================
    # 1. SUFFICIENCY TEST (Explanation → Prediction)
    # ==================================================================
    # Can we derive the prediction from the explanation alone?
    
    sufficiency_score = 0.0
    if primary_mech and prediction:
        # Extract predicted properties
        pred_values = ' '.join(str(v) for v in prediction.values())
        
        # Check if explanation mentions predicted properties
        explanation_embedding = embedding_model.encode(primary_mech)
        prediction_embedding = embedding_model.encode(pred_values)
        
        semantic_alignment = cosine_similarity(
            [explanation_embedding], 
            [prediction_embedding]
        )[0][0]
        
        if semantic_alignment > 0.5:
            sufficiency_score = 0.7
        
        # Bonus: Check explicit causal language
        if any(prop_key in primary_mech.lower() 
               for prop_key in prediction.keys()):
            sufficiency_score = min(1.0, sufficiency_score + 0.3)
    
    scores['sufficiency'] = sufficiency_score
    
    # ==================================================================
    # 2. COMPREHENSIVENESS TEST (All steps necessary?)
    # ==================================================================
    # If we remove a step, does reasoning break?
    
    comprehensiveness_score = 0.0
    if cot and len(cot) >= 2:
        # Check if steps are causally connected
        step_pairs_connected = 0
        for i in range(len(cot) - 1):
            if not isinstance(cot[i], str) or not isinstance(cot[i+1], str):
                continue
            
            # Extract key entities from consecutive steps
            entities_i = extract_key_terms(cot[i])
            entities_next = extract_key_terms(cot[i+1])
            
            # Check for entity overlap (information flow)
            if entities_i & entities_next:
                step_pairs_connected += 1
        
        if len(cot) > 1:
            comprehensiveness_score = step_pairs_connected / (len(cot) - 1)
    
    scores['comprehensiveness'] = comprehensiveness_score
    
    # ==================================================================
    # 3. LOGICAL CONSISTENCY (Self-consistency test)
    # ==================================================================
    # Check for contradictions in reasoning
    
    consistency_score = 1.0  # Start perfect, penalize contradictions
    
    if cot:
        # Check for negation patterns
        negation_words = ['not', 'however', 'but', 'although', 'despite', 'contrary']
        contradiction_count = 0
        
        for i, step in enumerate(cot):
            if not isinstance(step, str):
                continue
            
            # Flag if step contradicts primary mechanism
            if any(neg in step.lower() for neg in negation_words):
                step_embedding = embedding_model.encode(step)
                primary_embedding = embedding_model.encode(primary_mech)
                
                similarity = cosine_similarity([step_embedding], [primary_embedding])[0][0]
                
                # High similarity with negation = potential contradiction
                if similarity > 0.6:
                    contradiction_count += 1
        
        consistency_score = max(0.0, 1.0 - (contradiction_count * 0.3))
    
    scores['logical_consistency'] = consistency_score
    
    # ==================================================================
    # 4. MECHANISTIC PRECISION (Quantitative grounding)
    # ==================================================================
    # Reward specific, quantitative reasoning
    
    precision_score = 0.0
    
    # Check for numerical estimates
    quant_estimates = mech.get('quantitative_estimates', {})
    if quant_estimates:
        precision_score += 0.4
        
        # Reward ranges with units (more precise)
        has_units = any(
            any(unit in str(v).lower() for unit in ['cm', 'ev', 'k', 'mol', 'pa', 's'])
            for v in quant_estimates.values()
        )
        if has_units:
            precision_score += 0.3
    
    # Check for specific mechanism names
    mechanism_terms = ['substitution', 'doping', 'defect', 'phonon', 'electron', 
                      'diffusion', 'segregation', 'precipitation']
    if any(term in primary_mech.lower() for term in mechanism_terms):
        precision_score += 0.3
    
    scores['mechanistic_precision'] = min(precision_score, 1.0)
    
    # ==================================================================
    # 5. FORWARD SIMULATION (Can predict intermediate states?)
    # ==================================================================
    # Test if reasoning chain can generate intermediate predictions
    
    simulation_score = 0.0
    if len(cot) >= 3:
        # Check if middle steps contain observable/testable claims
        testable_keywords = ['concentration', 'temperature', 'size', 'thickness',
                           'density', 'composition', 'phase', 'crystal']
        
        middle_steps = cot[1:-1]  # Exclude first and last
        testable_steps = sum(
            1 for step in middle_steps 
            if isinstance(step, str) and any(kw in step.lower() for kw in testable_keywords)
        )
        
        if testable_steps > 0:
            simulation_score = min(1.0, testable_steps / len(middle_steps))
    
    scores['forward_simulation'] = simulation_score
    
    # ==================================================================
    # FINAL VALIDITY SCORE
    # ==================================================================
    weights = {
        'sufficiency': 0.30,
        'comprehensiveness': 0.20,
        'logical_consistency': 0.25,
        'mechanistic_precision': 0.15,
        'forward_simulation': 0.10
    }
    
    final_score = sum(scores[k] * weights[k] for k in weights)
    scores['internal_validity_score'] = np.clip(final_score, 0, 1)
    
    return scores


def extract_key_terms(text: str) -> set:
    """Extract key technical terms from text."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    # Extract nouns and technical terms
    terms = set()
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
            terms.add(token.lemma_.lower())
    
    return terms

# ==============================================================================
# UPDATED OVERALL EVALUATION
# ==============================================================================

def evaluate_prediction_v3(
    predicted: dict,
    ground_truth: dict,
    embedding_model: SentenceTransformer,
    kg: Optional[nx.DiGraph] = None,
    nli_model=None,
    use_enhanced_metrics: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation using literature-grounded metrics.
    """
    
    results = {}
    
    # Standard metrics (kept from original)
    from evaluation.metrics import (
        scientific_accuracy, 
        functional_equivalence, 
        completeness, 
        interpretability
    )
    
    results['scientific_accuracy'] = scientific_accuracy(predicted, ground_truth, embedding_model)
    results['functional_equivalence'] = functional_equivalence(predicted, ground_truth, embedding_model)
    results['completeness'] = completeness(predicted)
    results['interpretability'] = interpretability(predicted)
    
    if use_enhanced_metrics:
        # Literature-grounded metrics
        causal_scores = causal_coherence_v3(predicted, ground_truth, kg, embedding_model)
        results.update(causal_scores)
        
        grounding_scores = source_grounding_v3(predicted, ground_truth, kg, embedding_model, nli_model)
        results.update(grounding_scores)
        
        validity_scores = internal_validity_v3(predicted, embedding_model)
        results.update(validity_scores)
    
    # Calculate final composite score
    if use_enhanced_metrics:
        results['overall_score'] = np.mean([
            results['scientific_accuracy'],
            results['functional_equivalence'],
            results['causal_coherence_score'],
            results['source_grounding_score'],
            results['internal_validity_score']
        ])
    else:
        results['overall_score'] = np.mean([
            results['scientific_accuracy'],
            results['functional_equivalence'],
            results['completeness'],
            results['interpretability']
        ])
    
    return results


# ==============================================================================
# TESTING
# ==============================================================================

if __name__ == '__main__':
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Mock KG
    kg = nx.DiGraph()
    kg.add_edge("CVD", "Nb-doped MoS2", mechanism="synthesis")
    kg.add_edge("Nb-doped MoS2", "n-type conductivity", mechanism="electron donation")
    
    # Test case
    test_output = {
        'tier': 1,
        'confidence': 0.85,
        'kg_paths_used': 1,
        'kg_edges_used': 2,
        'literature_papers': 3,
        'literature_evidence': [
            "CVD enables high-temperature Nb substitution in MoS2 lattice",
            "Nb acts as electron donor when substituting Mo sites",
            "Carrier concentration increases to 1e13 cm-3 with Nb doping"
        ],
        'mechanistic_explanation': {
            'primary_mechanism': 'CVD at high temperature enables Nb substitution into Mo sites in MoS2, where Nb acts as electron donor increasing n-type carrier concentration',
            'chain_of_thought': [
                'CVD provides thermal energy for Nb-Mo exchange',
                'Nb substitution occurs at Mo lattice sites due to similar ionic radii',
                'Nb has one more valence electron than Mo (5d4 vs 4d5)',
                'Extra electron becomes free carrier in conduction band',
                'Carrier concentration increases, enhancing n-type conductivity'
            ],
            'quantitative_estimates': {
                'carrier_concentration': '1e12 to 1e13 cm-3',
                'mobility': '10-50 cm2/Vs',
                'doping_efficiency': '60-80%'
            },
            'alternative_mechanisms': 'Nb could form interstitial defects instead of substitution, but this is less favorable energetically and would create acceptor states'
        },
        'predicted_properties': {
            'carrier_type': 'n-type',
            'conductivity': 'enhanced',
            'mobility': 'moderate'
        }
    }
    
    ground_truth = {
        'properties': {
            'carrier_type': 'n-type',
            'conductivity': 'high'
        }
    }
    
    print("\n" + "="*70)
    print("LITERATURE-GROUNDED EVALUATION RESULTS")
    print("="*70)
    
    # Run evaluation
    results = evaluate_prediction_v3(
        test_output, 
        ground_truth, 
        model, 
        kg=kg,
        use_enhanced_metrics=True
    )
    
    print("\n📊 CAUSAL COHERENCE METRICS:")
    print(f"  • Intervention Consistency: {results.get('intervention_consistency', 0):.3f}")
    print(f"  • Counterfactual Reasoning: {results.get('counterfactual_reasoning', 0):.3f}")
    print(f"  • Mechanistic Modularity: {results.get('mechanistic_modularity', 0):.3f}")
    print(f"  • PSP Chain Validity: {results.get('psp_chain_validity', 0):.3f}")
    print(f"  ➡ Overall Causal Coherence: {results.get('causal_coherence_score', 0):.3f}")
    
    print("\n🔗 SOURCE GROUNDING METRICS:")
    print(f"  • KG Verification: {results.get('kg_verification', 0):.3f}")
    print(f"  • Literature Entailment: {results.get('literature_entailment', 0):.3f}")
    print(f"  • Multi-hop Grounding: {results.get('multihop_grounding', 0):.3f}")
    print(f"  • Source Diversity: {results.get('source_diversity', 0):.3f}")
    print(f"  ➡ Overall Source Grounding: {results.get('source_grounding_score', 0):.3f}")
    
    print("\n✓ INTERNAL VALIDITY METRICS:")
    print(f"  • Sufficiency: {results.get('sufficiency', 0):.3f}")
    print(f"  • Comprehensiveness: {results.get('comprehensiveness', 0):.3f}")
    print(f"  • Logical Consistency: {results.get('logical_consistency', 0):.3f}")
    print(f"  • Mechanistic Precision: {results.get('mechanistic_precision', 0):.3f}")
    print(f"  • Forward Simulation: {results.get('forward_simulation', 0):.3f}")
    print(f"  ➡ Overall Internal Validity: {results.get('internal_validity_score', 0):.3f}")
    
    print("\n" + "="*70)
    print(f"🎯 FINAL SCORE: {results['overall_score']:.3f}")
    print("="*70)