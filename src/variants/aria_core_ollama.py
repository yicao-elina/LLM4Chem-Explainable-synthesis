"""
ARIA Core Implementation (Ollama-based)

This variant implements the full 3-tier hierarchical reasoning framework:
- Tier 1: Direct path matching (exact KG matches)
- Tier 2: Transfer learning (similarity-based analogy)
- Tier 3: Baseline fallback (pure LLM when no KG match)

Tests the contribution of hierarchical reasoning and transfer learning.

Author: ARIA Team
Date: 2026-02-01
"""

import json
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from ..ollama_client import get_ollama_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARIACoreOllama:
    """
    ARIA Core: 3-tier hierarchical causal reasoning.

    Tier 1 (Direct): Exact keyword matches in KG
    Tier 2 (Transfer): Similarity-based analogical reasoning
    Tier 3 (Fallback): Pure LLM when no KG applicability
    """

    def __init__(
        self,
        kg_file: str = "../../data/KG/outputs/kg_example_2d_doping_enriched.json",
        model: str = "deepseek-r1:8b",
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5
    ):
        """
        Initialize ARIA Core system.

        Args:
            kg_file: Path to knowledge graph JSON
            model: Ollama model for reasoning
            embedding_model: Sentence transformer for embeddings
            similarity_threshold: Minimum similarity for transfer learning
        """
        self.kg_file = Path(kg_file)
        self.model = model
        self.similarity_threshold = similarity_threshold

        # Initialize Ollama client
        self.ollama = get_ollama_client(model=model)

        # Load KG
        self.kg = self._load_kg()

        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self._precompute_node_embeddings()

        logger.info(f"ARIA Core initialized: {self.kg.number_of_nodes()} nodes, "
                   f"{self.kg.number_of_edges()} edges, 3-tier reasoning enabled")

    def _load_kg(self) -> nx.DiGraph:
        """Load knowledge graph from JSON."""
        if not self.kg_file.exists():
            raise FileNotFoundError(f"KG file not found: {self.kg_file}")

        with open(self.kg_file, 'r') as f:
            data = json.load(f)

        G = nx.DiGraph()

        relationships = data.get('causal_relationships', data if isinstance(data, list) else [])

        for rel in relationships:
            cause = (rel.get('cause_parameter') or '').strip()
            effect = (rel.get('effect_on_doping') or '').strip()

            if (cause and effect and
                'unknown' not in cause.lower() and
                'n/a' not in cause.lower() and
                'unknown' not in effect.lower() and
                'n/a' not in effect.lower()):

                G.add_edge(
                    cause,
                    effect,
                    mechanism=rel.get('mechanism_quote', ''),
                    affected_property=rel.get('affected_property', ''),
                    confidence=rel.get('confidence', 1.0)
                )

        return G

    def _precompute_node_embeddings(self):
        """Precompute embeddings for all KG nodes for fast similarity search."""
        logger.info("Precomputing node embeddings...")
        self.node_list = list(self.kg.nodes())

        if not self.node_list:
            self.node_embeddings = np.array([])
            logger.warning("No nodes in KG - embeddings empty")
            return

        self.node_embeddings = self.embedding_model.encode(
            self.node_list,
            convert_to_tensor=False
        )
        logger.info(f"Embeddings computed for {len(self.node_list)} nodes")

    def _find_most_similar_node(
        self,
        query: str,
        candidate_nodes: List[str]
    ) -> Tuple[Optional[str], float]:
        """
        Find most similar node to query using cosine similarity.

        Args:
            query: Query string
            candidate_nodes: Nodes to search among

        Returns:
            (best_node, similarity_score)
        """
        if not candidate_nodes or self.node_embeddings.size == 0:
            return None, 0.0

        # Encode query
        query_embedding = self.embedding_model.encode([query])

        # Get embeddings for candidates
        candidate_indices = [
            self.node_list.index(node) for node in candidate_nodes
            if node in self.node_list
        ]

        if not candidate_indices:
            return None, 0.0

        candidate_embeddings = self.node_embeddings[candidate_indices]

        # Compute similarities
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_node = candidate_nodes[best_idx]

        return best_node, best_score

    def _calculate_embedding_distance(self, text1: str, text2: str) -> float:
        """
        Calculate cosine distance (1 - similarity) between two texts.

        Returns:
            Distance from 0 (identical) to 2 (opposite)
        """
        if not text1 or not text2:
            return 1.0

        embeddings = self.embedding_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        distance = 1 - similarity

        return distance

    def _find_paths(
        self,
        start_keywords: List[str],
        end_keywords: List[str],
        reverse: bool = False
    ) -> List[str]:
        """
        Find paths in KG using keyword matching.

        Args:
            start_keywords: Keywords for start nodes
            end_keywords: Keywords for end nodes
            reverse: Whether to reverse the graph

        Returns:
            List of path strings ("node1 -> node2 -> node3")
        """
        # Match nodes
        start_nodes = {
            n for n in self.kg.nodes()
            if any(kw.lower() in n.lower() for kw in start_keywords)
        }
        end_nodes = {
            n for n in self.kg.nodes()
            if any(kw.lower() in n.lower() for kw in end_keywords)
        }

        # Setup graph and node roles
        graph = self.kg.reverse(copy=True) if reverse else self.kg
        sources = end_nodes if reverse else start_nodes
        targets = start_nodes if reverse else end_nodes

        # Find all simple paths
        valid_paths = []
        for source in sources:
            for target in targets:
                if nx.has_path(graph, source, target):
                    for path in nx.all_simple_paths(graph, source, target, cutoff=4):
                        path_str = " -> ".join(path)
                        valid_paths.append(path_str)

        return list(set(valid_paths))

    def _extract_mechanisms(self, paths: List[str]) -> List[str]:
        """Extract mechanism quotes from paths."""
        mechanisms = []
        for path_str in paths:
            nodes = path_str.split(" -> ")
            for i in range(len(nodes) - 1):
                if self.kg.has_edge(nodes[i], nodes[i+1]):
                    mech = self.kg[nodes[i]][nodes[i+1]].get('mechanism', '')
                    if mech and mech.strip():
                        mechanisms.append(mech.strip())

        return mechanisms

    def _tier1_forward_direct(
        self,
        synthesis_inputs: Dict[str, Any],
        paths: List[str],
        mechanisms: List[str]
    ) -> Dict[str, Any]:
        """
        Tier 1: Direct path reasoning with exact KG matches.

        Args:
            synthesis_inputs: Synthesis conditions
            paths: Found causal paths
            mechanisms: Extracted mechanisms

        Returns:
            Prediction dict
        """
        logger.info("Using Tier 1: Direct path reasoning")

        formatted_paths = "\n- ".join(paths)
        formatted_mechs = "\n- ".join(mechanisms) if mechanisms else "No mechanisms available"

        prompt = f"""You are an expert materials scientist AI. Your knowledge graph contains direct causal pathways relevant to the query.

**Synthesis Conditions:**
{json.dumps(synthesis_inputs, indent=2)}

**Direct Causal Pathways from Knowledge Graph:**
- {formatted_paths}

**Known Mechanisms from Knowledge Graph:**
- {formatted_mechs}

**Your Task:**
1. Explain the mechanistic pathway from synthesis conditions to material properties
2. Provide step-by-step logical reasoning
3. Give quantitative estimates where possible
4. Mention alternative pathways if relevant

Respond with ONLY valid JSON in this format:
{{
  "predicted_properties": {{
    "carrier_type": "n-type or p-type or null",
    "carrier_concentration": "value with units or null",
    "mobility": "value with units or null",
    "conductivity": "value with units or null",
    "band_gap": "value with units or null",
    "other_properties": {{}}
  }},
  "mechanistic_explanation": {{
    "primary_mechanism": "detailed explanation",
    "chain_of_thought": ["step 1", "step 2", "step 3"],
    "quantitative_estimates": {{}},
    "alternative_mechanisms": "other possibilities"
  }},
  "confidence": 0.0-1.0,
  "tier": 1,
  "reasoning_type": "direct_path"
}}"""

        try:
            response = self.ollama.generate_json(prompt, temperature=0.0)
            response['tier'] = 1
            response['reasoning_type'] = 'direct_path'
            response['kg_paths_used'] = len(paths)
            if 'mechanistic_explanation' in response:
                response['reasoning'] = json.dumps(response['mechanistic_explanation'])
            return response

        except Exception as e:
            logger.error(f"Tier 1 direct path failed: {e}")
            return self._tier3_fallback(synthesis_inputs, is_forward=True)

    def _tier2_forward_transfer(
        self,
        synthesis_inputs: Dict[str, Any],
        similar_node: str,
        similarity: float,
        paths: List[str],
        mechanisms: List[str]
    ) -> Dict[str, Any]:
        """
        Tier 2: Transfer learning with analogous cases.

        Args:
            synthesis_inputs: Synthesis conditions
            similar_node: Most similar KG node
            similarity: Similarity score
            paths: Analogous paths
            mechanisms: Extracted mechanisms

        Returns:
            Prediction dict
        """
        logger.info(f"Using Tier 2: Transfer learning (similarity={similarity:.3f})")

        formatted_path = paths[0] if paths else "No path found"
        formatted_mechs = "\n- ".join(mechanisms) if mechanisms else "No mechanisms available"

        prompt = f"""You are an expert materials scientist AI. No exact match found in knowledge graph, but an analogous case exists.

**Synthesis Conditions (Target):**
{json.dumps(synthesis_inputs, indent=2)}

**Most Similar Known Case:**
{similar_node} (similarity: {similarity:.3f})

**Analogous Causal Pathway:**
{formatted_path}

**Mechanisms from Analogous Case:**
- {formatted_mechs}

**Your Task:**
1. Adapt the analogous knowledge to the target case
2. Explain how the target differs from the known case
3. Adjust predictions based on these differences
4. Quantify uncertainty due to the analogy gap

Respond with ONLY valid JSON:
{{
  "predicted_properties": {{
    "carrier_type": "...",
    "carrier_concentration": "...",
    "mobility": "...",
    "other_properties": {{}}
  }},
  "mechanistic_explanation": {{
    "analogous_mechanism": "mechanism from similar case",
    "adaptation_reasoning": "how to adapt to target case",
    "similarity_analysis": {{
      "known_case": "{similar_node}",
      "target_case": "{json.dumps(synthesis_inputs)}",
      "key_differences": "...",
      "expected_impact": "..."
    }},
    "uncertainty_analysis": "sources of uncertainty"
  }},
  "confidence": {similarity},
  "tier": 2,
  "reasoning_type": "transfer_learning"
}}"""

        try:
            response = self.ollama.generate_json(prompt, temperature=0.0)
            response['tier'] = 2
            response['reasoning_type'] = 'transfer_learning'
            response['similarity_score'] = similarity
            response['analogous_node'] = similar_node
            if 'mechanistic_explanation' in response:
                response['reasoning'] = json.dumps(response['mechanistic_explanation'])
            return response

        except Exception as e:
            logger.error(f"Tier 2 transfer learning failed: {e}")
            return self._tier3_fallback(synthesis_inputs, is_forward=True)

    def _tier3_fallback(
        self,
        inputs: Dict[str, Any],
        is_forward: bool = True
    ) -> Dict[str, Any]:
        """
        Tier 3: Pure LLM fallback when no KG applicability.

        Args:
            inputs: Input dict (synthesis or properties)
            is_forward: Whether forward or inverse

        Returns:
            Prediction/recommendation dict
        """
        logger.info("Using Tier 3: Baseline fallback (no KG match)")

        if is_forward:
            prompt = f"""You are an expert materials scientist. Predict properties from synthesis conditions using fundamental principles (NO knowledge graph available).

**Synthesis Conditions:**
{json.dumps(inputs, indent=2)}

Respond with ONLY valid JSON:
{{
  "predicted_properties": {{"carrier_type": "...", "mobility": "...", ...}},
  "mechanistic_explanation": {{
    "reasoning": "based on fundamental principles"
  }},
  "confidence": 0.0-1.0,
  "tier": 3,
  "reasoning_type": "baseline_fallback"
}}"""

        else:  # Inverse
            prompt = f"""You are an expert materials scientist. Suggest synthesis conditions for desired properties using fundamental principles (NO knowledge graph available).

**Desired Properties:**
{json.dumps(inputs, indent=2)}

Respond with ONLY valid JSON:
{{
  "suggested_synthesis_conditions": {{"method": "...", "temperature_c": ..., ...}},
  "mechanistic_explanation": {{
    "reasoning": "based on fundamental principles"
  }},
  "confidence": 0.0-1.0,
  "tier": 3,
  "reasoning_type": "baseline_fallback_inverse"
}}"""

        try:
            response = self.ollama.generate_json(prompt, temperature=0.0)
            response['tier'] = 3
            response['kg_paths_used'] = 0
            if 'mechanistic_explanation' in response:
                response['reasoning'] = json.dumps(response['mechanistic_explanation'])
            return response

        except Exception as e:
            logger.error(f"Tier 3 fallback failed: {e}")
            return {
                'error': str(e),
                'confidence': 0.0,
                'tier': 3,
                'reasoning_type': 'error'
            }

    def forward_prediction(self, synthesis_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict properties from synthesis using 3-tier reasoning.

        Workflow:
        1. Try Tier 1: Direct path matching
        2. If no match, try Tier 2: Transfer learning
        3. If still no match, use Tier 3: Baseline fallback

        Args:
            synthesis_inputs: Synthesis conditions

        Returns:
            Prediction dict with tier information
        """
        logger.info("=== Forward Prediction (3-Tier) ===")

        # Extract keywords
        input_keywords = [str(v) for v in synthesis_inputs.values() if v is not None]

        # Find property nodes (leaf nodes)
        property_nodes = [n for n in self.kg.nodes() if self.kg.out_degree(n) == 0]

        # Tier 1: Try direct path matching
        paths = self._find_paths(input_keywords, property_nodes)

        if paths:
            mechanisms = self._extract_mechanisms(paths)
            return self._tier1_forward_direct(synthesis_inputs, paths, mechanisms)

        # Tier 2: Try transfer learning
        synthesis_nodes = [n for n in self.kg.nodes() if self.kg.in_degree(n) == 0]
        query_string = " and ".join(input_keywords)
        similar_node, score = self._find_most_similar_node(query_string, synthesis_nodes)

        if similar_node and score > self.similarity_threshold:
            analogous_paths = self._find_paths([similar_node], property_nodes)
            if analogous_paths:
                mechanisms = self._extract_mechanisms(analogous_paths)
                return self._tier2_forward_transfer(
                    synthesis_inputs, similar_node, score, analogous_paths, mechanisms
                )

        # Tier 3: Fallback
        return self._tier3_fallback(synthesis_inputs, is_forward=True)

    def inverse_design(self, desired_properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design synthesis from properties using 3-tier reasoning.

        Similar workflow to forward prediction but in reverse direction.

        Args:
            desired_properties: Target properties

        Returns:
            Synthesis recommendation dict with tier information
        """
        logger.info("=== Inverse Design (3-Tier) ===")

        # Extract keywords
        property_keywords = [str(v) for v in desired_properties.values() if v is not None]

        # Find synthesis nodes (root nodes)
        synthesis_nodes = [n for n in self.kg.nodes() if self.kg.in_degree(n) == 0]

        # Tier 1: Try direct path matching (reverse)
        paths = self._find_paths(synthesis_nodes, property_keywords, reverse=True)

        if paths:
            mechanisms = self._extract_mechanisms(paths)
            return self._tier1_inverse_direct(desired_properties, paths, mechanisms)

        # Tier 2: Try transfer learning
        property_nodes = [n for n in self.kg.nodes() if self.kg.out_degree(n) == 0]
        query_string = " and ".join(property_keywords)
        similar_node, score = self._find_most_similar_node(query_string, property_nodes)

        if similar_node and score > self.similarity_threshold:
            analogous_paths = self._find_paths(synthesis_nodes, [similar_node], reverse=True)
            if analogous_paths:
                mechanisms = self._extract_mechanisms(analogous_paths)
                embedding_dist = self._calculate_embedding_distance(query_string, similar_node)
                return self._tier2_inverse_transfer(
                    desired_properties, similar_node, score, embedding_dist,
                    analogous_paths, mechanisms
                )

        # Tier 3: Fallback
        return self._tier3_fallback(desired_properties, is_forward=False)

    def _tier1_inverse_direct(
        self,
        desired_properties: Dict[str, Any],
        paths: List[str],
        mechanisms: List[str]
    ) -> Dict[str, Any]:
        """Tier 1 inverse design with direct paths."""
        logger.info("Using Tier 1: Direct inverse design")

        formatted_paths = "\n- ".join(paths)
        formatted_mechs = "\n- ".join(mechanisms) if mechanisms else "No mechanisms"

        prompt = f"""Design synthesis conditions for desired properties using direct KG paths.

**Desired Properties:**
{json.dumps(desired_properties, indent=2)}

**Direct Causal Pathways (Reverse):**
- {formatted_paths}

**Known Mechanisms:**
- {formatted_mechs}

Respond with ONLY valid JSON:
{{
  "suggested_synthesis_conditions": {{
    "method": "...",
    "temperature_c": ...,
    "pressure_pa": ...,
    "time_hours": ...,
    "atmosphere": "...",
    "other_parameters": {{}}
  }},
  "mechanistic_explanation": {{
    "primary_mechanism": "...",
    "chain_of_thought": ["step 1", "step 2"],
    "confidence_factors": "..."
  }},
  "confidence": 0.0-1.0,
  "tier": 1,
  "reasoning_type": "direct_inverse"
}}"""

        try:
            response = self.ollama.generate_json(prompt, temperature=0.0)
            response['tier'] = 1
            response['reasoning_type'] = 'direct_inverse'
            response['kg_paths_used'] = len(paths)
            return response
        except Exception as e:
            logger.error(f"Tier 1 inverse failed: {e}")
            return self._tier3_fallback(desired_properties, is_forward=False)

    def _tier2_inverse_transfer(
        self,
        desired_properties: Dict[str, Any],
        similar_node: str,
        similarity: float,
        embedding_distance: float,
        paths: List[str],
        mechanisms: List[str]
    ) -> Dict[str, Any]:
        """Tier 2 inverse design with transfer learning."""
        logger.info(f"Using Tier 2: Transfer inverse (similarity={similarity:.3f})")

        formatted_path = paths[0] if paths else "No path"
        formatted_mechs = "\n- ".join(mechanisms) if mechanisms else "No mechanisms"

        prompt = f"""Design synthesis conditions using analogous knowledge.

**Desired Properties (Target):**
{json.dumps(desired_properties, indent=2)}

**Most Similar Known Property:**
{similar_node} (similarity: {similarity:.3f}, distance: {embedding_distance:.3f})

**Analogous Pathway:**
{formatted_path}

**Mechanisms:**
- {formatted_mechs}

Adapt the analogous synthesis to achieve target properties.

Respond with ONLY valid JSON:
{{
  "suggested_synthesis_conditions": {{
    "method": "...",
    "temperature_c": ...,
    "other_parameters": {{}}
  }},
  "mechanistic_explanation": {{
    "analogous_mechanism": "...",
    "adaptation_reasoning": "...",
    "similarity_analysis": {{
      "known_property": "{similar_node}",
      "target_property": "...",
      "required_adjustments": "..."
    }}
  }},
  "confidence": {similarity},
  "tier": 2,
  "reasoning_type": "transfer_inverse"
}}"""

        try:
            response = self.ollama.generate_json(prompt, temperature=0.0)
            response['tier'] = 2
            response['reasoning_type'] = 'transfer_inverse'
            response['similarity_score'] = similarity
            response['embedding_distance'] = embedding_distance
            return response
        except Exception as e:
            logger.error(f"Tier 2 inverse transfer failed: {e}")
            return self._tier3_fallback(desired_properties, is_forward=False)


def main():
    """Test ARIA Core implementation."""
    print("="*60)
    print("Testing ARIA Core (3-Tier Reasoning)")
    print("="*60)

    try:
        aria = ARIACoreOllama(
            kg_file="../../data/KG/outputs/kg_example_2d_doping_enriched.json",
            model="deepseek-r1:8b"
        )

        # Test 1: Forward prediction
        print("\n[Test 1] Forward Prediction")
        print("-" * 60)

        synthesis = {
            "method": "CVD",
            "temperature_c": 750,
            "material": "MoS2",
            "dopant": "Nb"
        }

        print(f"Input: {json.dumps(synthesis, indent=2)}")
        result = aria.forward_prediction(synthesis)

        print(f"\nOutput:")
        print(f"  Tier: {result.get('tier', 'unknown')}")
        print(f"  Reasoning type: {result.get('reasoning_type', 'unknown')}")
        print(f"  Confidence: {result.get('confidence', 0.0)}")
        print(f"  Properties: {json.dumps(result.get('predicted_properties', {}), indent=2)}")

        # Test 2: Inverse design
        print("\n[Test 2] Inverse Design")
        print("-" * 60)

        properties = {
            "carrier_type": "n-type",
            "mobility": "high (>50 cm2/V·s)",
            "material": "2D TMD"
        }

        print(f"Desired: {json.dumps(properties, indent=2)}")
        result = aria.inverse_design(properties)

        print(f"\nOutput:")
        print(f"  Tier: {result.get('tier', 'unknown')}")
        print(f"  Confidence: {result.get('confidence', 0.0)}")
        print(f"  Conditions: {json.dumps(result.get('suggested_synthesis_conditions', {}), indent=2)}")

        print("\n" + "="*60)
        print("✓ ARIA Core tests completed!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
