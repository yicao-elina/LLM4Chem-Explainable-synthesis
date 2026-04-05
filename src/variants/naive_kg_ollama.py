"""
Naive KG + LLM Implementation (Ollama-based)

This variant represents simple KG augmentation without advanced reasoning:
- Retrieves exact KG path matches (no analogy/transfer learning)
- No online searching
- Uses LLM to interpret retrieved KG paths
- Tests contribution of basic KG retrieval in isolation

Author: ARIA Team
Date: 2026-02-01
"""

import json
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from ..ollama_client import get_ollama_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NaiveKGOllama:
    """
    Naive KG + LLM variant for ablation study.

    Key characteristics:
    - Simple KG path retrieval (exact keyword matches only)
    - NO transfer learning or analogy reasoning
    - NO online search integration
    - Flat reasoning (no tier separation)
    - Just concatenate KG paths → LLM prompt
    """

    def __init__(
        self,
        kg_file: str = "../../data/KG/outputs/kg_example_2d_doping_enriched.json",
        model: str = "deepseek-r1:8b"
    ):
        """
        Initialize Naive KG system.

        Args:
            kg_file: Path to knowledge graph JSON
            model: Ollama model to use
        """
        self.kg_file = Path(kg_file)
        self.model = model
        self.ollama = get_ollama_client(model=model)

        # Load KG
        self.kg = self._load_kg()

        logger.info(f"NaiveKG initialized with {self.kg.number_of_nodes()} nodes, "
                   f"{self.kg.number_of_edges()} edges")

    def _load_kg(self) -> nx.DiGraph:
        """Load knowledge graph from JSON."""
        if not self.kg_file.exists():
            raise FileNotFoundError(f"KG file not found: {self.kg_file}")

        with open(self.kg_file, 'r') as f:
            data = json.load(f)

        # Build directed graph from causal relationships
        G = nx.DiGraph()

        relationships = data.get('causal_relationships', data if isinstance(data, list) else [])
        for rel in relationships:
            cause = (rel.get('cause_parameter') or '').strip()
            effect = (rel.get('effect_on_doping') or '').strip()

            # Filter out unknown/n/a entries
            if (cause and effect and
                'unknown' not in cause.lower() and
                'n/a' not in cause.lower() and
                'unknown' not in effect.lower() and
                'n/a' not in effect.lower()):

                # Add edge with mechanism metadata
                G.add_edge(
                    cause,
                    effect,
                    mechanism=rel.get('mechanism_quote', ''),
                    affected_property=rel.get('affected_property', ''),
                    source_file=rel.get('source_file', '')
                )

        return G

    def _find_exact_paths(self, start_keywords: List[str], end_keywords: List[str]) -> List[List[str]]:
        """
        Find paths in KG using exact keyword matching.

        NO similarity-based matching - only exact keyword presence.

        Args:
            start_keywords: Keywords for starting nodes
            end_keywords: Keywords for ending nodes

        Returns:
            List of paths (each path is list of nodes)
        """
        # Find nodes containing start keywords
        start_nodes = []
        for node in self.kg.nodes():
            node_lower = node.lower()
            if any(kw.lower() in node_lower for kw in start_keywords):
                start_nodes.append(node)

        # Find nodes containing end keywords
        end_nodes = []
        for node in self.kg.nodes():
            node_lower = node.lower()
            if any(kw.lower() in node_lower for kw in end_keywords):
                end_nodes.append(node)

        # Find all simple paths between start and end nodes
        all_paths = []
        for start in start_nodes:
            for end in end_nodes:
                try:
                    paths = list(nx.all_simple_paths(self.kg, start, end, cutoff=3))
                    all_paths.extend(paths)
                except nx.NetworkXNoPath:
                    continue

        return all_paths

    def _extract_path_mechanisms(self, paths: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Extract mechanisms and metadata from paths.

        Args:
            paths: List of node paths

        Returns:
            List of path dictionaries with mechanisms
        """
        path_data = []

        for path in paths:
            mechanisms = []
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]

                edge_data = self.kg.get_edge_data(source, target, {})
                mechanisms.append({
                    'from': source,
                    'to': target,
                    'mechanism': edge_data.get('mechanism', 'Not specified'),
                    'affected_property': edge_data.get('affected_property', ''),
                    'source_file': edge_data.get('source_file', '')
                })

            path_data.append({
                'path': path,
                'mechanisms': mechanisms
            })

        return path_data

    def _format_kg_context(self, path_data: List[Dict[str, Any]]) -> str:
        """
        Format KG paths for LLM prompt.
        Simple concatenation without sophisticated organization.

        Args:
            path_data: Extracted path data with mechanisms

        Returns:
            Formatted string for prompt
        """
        if not path_data:
            return "No relevant causal pathways found in the knowledge graph."

        formatted = "Retrieved causal pathways from knowledge graph:\n\n"

        for i, path_info in enumerate(path_data, 1):
            formatted += f"Pathway {i}:\n"
            formatted += f"  Path: {' → '.join(path_info['path'])}\n"
            formatted += f"  Mechanisms:\n"

            for mech in path_info['mechanisms']:
                formatted += f"    • {mech['from']} → {mech['to']}\n"
                formatted += f"      Mechanism: {mech['mechanism']}\n"
                if mech['affected_property']:
                    formatted += f"      Affects: {mech['affected_property']}\n"
            formatted += "\n"

        return formatted

    def forward_prediction(
        self,
        synthesis_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict material properties from synthesis conditions.

        Workflow:
        1. Extract keywords from synthesis inputs
        2. Find exact KG paths (no similarity matching)
        3. Format KG paths as context
        4. Prompt LLM with KG context + synthesis inputs
        5. Parse and return prediction

        Args:
            synthesis_conditions: Dict with synthesis parameters

        Returns:
            Prediction dict with properties and confidence
        """
        logger.info("Forward prediction (Naive KG)")

        # Extract keywords from inputs
        input_keywords = []
        for key, value in synthesis_conditions.items():
            if isinstance(value, str):
                input_keywords.extend(value.split())
            else:
                input_keywords.append(str(value))

        # Property keywords (target)
        property_keywords = ['mobility', 'conductivity', 'carrier', 'band gap', 'property']

        # Find paths
        paths = self._find_exact_paths(input_keywords, property_keywords)
        path_data = self._extract_path_mechanisms(paths)
        kg_context = self._format_kg_context(path_data)

        # Build prompt
        prompt = f"""You are a materials science expert. Predict the material properties that would result from the given synthesis conditions.

KNOWLEDGE GRAPH CONTEXT:
{kg_context}

SYNTHESIS CONDITIONS:
{json.dumps(synthesis_conditions, indent=2)}

TASK: Predict the resulting material properties based on:
1. The causal pathways from the knowledge graph
2. General materials science principles

Respond with ONLY valid JSON in this exact format:
{{
  "predicted_properties": {{
    "carrier_type": "n-type or p-type",
    "carrier_concentration": "value with units",
    "mobility": "value with units",
    "conductivity": "value with units",
    "band_gap": "value with units",
    "other_properties": {{}}
  }},
  "mechanistic_explanation": "Detailed explanation of how synthesis conditions lead to these properties",
  "confidence": 0.0-1.0,
  "kg_paths_used": {len(paths)},
  "reasoning_type": "naive_kg"
}}"""

        try:
            response = self.ollama.generate_json(prompt, temperature=0.0)

            # Ensure metadata is included
            if 'kg_paths_used' not in response:
                response['kg_paths_used'] = len(paths)
            if 'reasoning_type' not in response:
                response['reasoning_type'] = 'naive_kg'

            # Add raw KG context for transparency
            response['kg_context'] = kg_context

            return response

        except Exception as e:
            logger.error(f"Forward prediction failed: {e}")
            return {
                'predicted_properties': {},
                'mechanistic_explanation': f'Error: {str(e)}',
                'confidence': 0.0,
                'kg_paths_used': len(paths),
                'reasoning_type': 'naive_kg_error'
            }

    def inverse_design(
        self,
        desired_properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Design synthesis conditions to achieve desired properties.

        Workflow:
        1. Extract keywords from property targets
        2. Find exact KG paths (reverse direction)
        3. Format KG paths as context
        4. Prompt LLM with KG context + desired properties
        5. Parse and return synthesis recommendations

        Args:
            desired_properties: Dict with target properties

        Returns:
            Synthesis recommendations dict
        """
        logger.info("Inverse design (Naive KG)")

        # Extract property keywords
        property_keywords = []
        for key, value in desired_properties.items():
            if isinstance(value, str):
                property_keywords.extend(value.split())
            property_keywords.append(key)

        # Synthesis parameter keywords (sources)
        synthesis_keywords = ['temperature', 'pressure', 'time', 'atmosphere', 'method', 'CVD', 'MBE']

        # Find reverse paths (properties → synthesis)
        paths = self._find_exact_paths(property_keywords, synthesis_keywords)

        # Also try forward direction (synthesis → properties) for context
        forward_paths = self._find_exact_paths(synthesis_keywords, property_keywords)

        # Combine both directions
        all_paths = paths + forward_paths
        path_data = self._extract_path_mechanisms(all_paths)
        kg_context = self._format_kg_context(path_data)

        # Build prompt
        prompt = f"""You are a materials science expert. Design synthesis conditions to achieve the desired material properties.

KNOWLEDGE GRAPH CONTEXT:
{kg_context}

DESIRED PROPERTIES:
{json.dumps(desired_properties, indent=2)}

TASK: Suggest synthesis conditions that would produce these properties based on:
1. The causal pathways from the knowledge graph (in reverse)
2. General materials science principles

Respond with ONLY valid JSON in this exact format:
{{
  "suggested_synthesis_conditions": {{
    "method": "synthesis method",
    "temperature_c": value or range,
    "time_hours": value or range,
    "atmosphere": "atmospheric conditions",
    "pressure_pa": value or null,
    "precursors": ["list of precursors"],
    "other_parameters": {{}}
  }},
  "mechanistic_explanation": "Detailed explanation of why these conditions should produce the desired properties",
  "confidence": 0.0-1.0,
  "kg_paths_used": {len(all_paths)},
  "reasoning_type": "naive_kg_inverse"
}}"""

        try:
            response = self.ollama.generate_json(prompt, temperature=0.0)

            # Ensure metadata
            if 'kg_paths_used' not in response:
                response['kg_paths_used'] = len(all_paths)
            if 'reasoning_type' not in response:
                response['reasoning_type'] = 'naive_kg_inverse'

            response['kg_context'] = kg_context

            return response

        except Exception as e:
            logger.error(f"Inverse design failed: {e}")
            return {
                'suggested_synthesis_conditions': {},
                'mechanistic_explanation': f'Error: {str(e)}',
                'confidence': 0.0,
                'kg_paths_used': len(all_paths),
                'reasoning_type': 'naive_kg_inverse_error'
            }


def main():
    """Test the Naive KG implementation."""
    print("="*60)
    print("Testing Naive KG + Ollama Implementation")
    print("="*60)

    try:
        # Initialize
        naive_kg = NaiveKGOllama(
            kg_file="../../data/KG/outputs/kg_example_2d_doping_enriched.json",
            model="deepseek-r1:8b"
        )

        # Test 1: Forward prediction
        print("\n[Test 1] Forward Prediction")
        print("-" * 60)

        synthesis_input = {
            "method": "CVD",
            "temperature_c": 750,
            "material": "MoS2",
            "dopant": "Nb"
        }

        print(f"Input: {json.dumps(synthesis_input, indent=2)}")

        result = naive_kg.forward_prediction(synthesis_input)

        print(f"\nOutput:")
        print(f"  Predicted properties: {json.dumps(result.get('predicted_properties', {}), indent=2)}")
        print(f"  Confidence: {result.get('confidence', 0.0)}")
        print(f"  KG paths used: {result.get('kg_paths_used', 0)}")
        print(f"  Reasoning type: {result.get('reasoning_type', 'unknown')}")

        # Test 2: Inverse design
        print("\n[Test 2] Inverse Design")
        print("-" * 60)

        desired_props = {
            "carrier_type": "n-type",
            "mobility": "high",
            "material": "MoS2"
        }

        print(f"Desired properties: {json.dumps(desired_props, indent=2)}")

        result = naive_kg.inverse_design(desired_props)

        print(f"\nOutput:")
        print(f"  Suggested conditions: {json.dumps(result.get('suggested_synthesis_conditions', {}), indent=2)}")
        print(f"  Confidence: {result.get('confidence', 0.0)}")
        print(f"  KG paths used: {result.get('kg_paths_used', 0)}")

        print("\n" + "="*60)
        print("✓ Naive KG tests completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
