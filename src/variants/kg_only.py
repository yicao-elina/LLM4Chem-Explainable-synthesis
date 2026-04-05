"""
KG-Only Implementation (No LLM Enhancement)

This variant represents pure graph-based reasoning without LLM:
- Pure knowledge graph traversal
- Template-based response generation (no LLM reasoning)
- Deterministic outputs based solely on KG structure
- Tests KG quality and coverage in isolation

Author: ARIA Team
Date: 2026-02-01
"""

import json
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KGOnly:
    """
    Pure KG variant for ablation study.

    Key characteristics:
    - NO LLM reasoning (pure graph traversal)
    - Template-based output generation
    - Deterministic (same input = same output)
    - Tests KG coverage and quality in isolation
    - High confidence when path exists, zero otherwise
    """

    def __init__(self, kg_file: str = "../../data/KG/outputs/kg_example_2d_doping_enriched.json"):
        """
        Initialize KG-only system.

        Args:
            kg_file: Path to knowledge graph JSON
        """
        self.kg_file = Path(kg_file)
        self.kg = self._load_kg()

        logger.info(f"KGOnly initialized with {self.kg.number_of_nodes()} nodes, "
                   f"{self.kg.number_of_edges()} edges (NO LLM)")

    def _load_kg(self) -> nx.DiGraph:
        """Load knowledge graph from JSON."""
        if not self.kg_file.exists():
            raise FileNotFoundError(f"KG file not found: {self.kg_file}")

        with open(self.kg_file, 'r') as f:
            data = json.load(f)

        # Build directed graph
        G = nx.DiGraph()

        relationships = data.get('causal_relationships', data if isinstance(data, list) else [])
        for rel in relationships:
            cause = (rel.get('cause_parameter') or '').strip()
            effect = (rel.get('effect_on_doping') or '').strip()

            # Filter invalid entries
            if (cause and effect and
                'unknown' not in cause.lower() and
                'n/a' not in cause.lower() and
                'unknown' not in effect.lower() and
                'n/a' not in effect.lower()):

                # Store full metadata on edges
                G.add_edge(
                    cause,
                    effect,
                    mechanism=rel.get('mechanism_quote', ''),
                    affected_property=rel.get('affected_property', ''),
                    source_file=rel.get('source_file', ''),
                    source_doi=rel.get('source_doi', ''),
                    confidence=rel.get('confidence', 1.0)
                )

        return G

    def _find_paths(self, start_keywords: List[str], end_keywords: List[str]) -> List[List[str]]:
        """
        Find all paths in KG using keyword matching.

        Args:
            start_keywords: Keywords for source nodes
            end_keywords: Keywords for target nodes

        Returns:
            List of paths (each path is list of node names)
        """
        # Match source nodes
        start_nodes = []
        for node in self.kg.nodes():
            node_lower = node.lower()
            if any(kw.lower() in node_lower for kw in start_keywords):
                start_nodes.append(node)

        # Match target nodes
        end_nodes = []
        for node in self.kg.nodes():
            node_lower = node.lower()
            if any(kw.lower() in node_lower for kw in end_keywords):
                end_nodes.append(node)

        # Find all simple paths
        all_paths = []
        for start in start_nodes:
            for end in end_nodes:
                try:
                    # Find paths up to length 4 (to avoid exponential blowup)
                    paths = list(nx.all_simple_paths(self.kg, start, end, cutoff=4))
                    all_paths.extend(paths)
                except nx.NetworkXNoPath:
                    continue

        return all_paths

    def _extract_mechanisms_from_path(self, path: List[str]) -> List[Dict[str, Any]]:
        """
        Extract edge mechanisms along a path.

        Args:
            path: List of node names

        Returns:
            List of edge data dictionaries
        """
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
                'source_file': edge_data.get('source_file', ''),
                'confidence': edge_data.get('confidence', 1.0)
            })

        return mechanisms

    def _template_fill_forward(self, paths: List[List[str]]) -> Dict[str, Any]:
        """
        Generate forward prediction using pure template filling.

        NO LLM - just extract information from KG paths.

        Args:
            paths: Found causal paths

        Returns:
            Structured prediction dict
        """
        if not paths:
            return {
                'predicted_properties': {},
                'mechanistic_explanation': 'No causal pathway found in knowledge graph.',
                'confidence': 0.0,
                'paths_found': 0,
                'reasoning_type': 'kg_only_no_match'
            }

        # Extract all mechanisms
        all_mechanisms = []
        affected_properties = set()

        for path in paths:
            mechs = self._extract_mechanisms_from_path(path)
            all_mechanisms.extend(mechs)
            for m in mechs:
                if m['affected_property']:
                    affected_properties.add(m['affected_property'])

        # Template-based property extraction
        predicted_properties = {}

        # Extract properties from end nodes and mechanisms
        for prop in affected_properties:
            predicted_properties[prop] = "Affected (see mechanisms)"

        # Concatenate mechanisms as explanation
        explanation_parts = []
        for i, path in enumerate(paths, 1):
            path_str = ' → '.join(path)
            explanation_parts.append(f"Pathway {i}: {path_str}")

            mechs = self._extract_mechanisms_from_path(path)
            for m in mechs:
                explanation_parts.append(f"  • {m['from']} → {m['to']}: {m['mechanism']}")

        mechanistic_explanation = '\n'.join(explanation_parts)

        # Confidence = 1.0 when KG path exists (deterministic)
        avg_confidence = sum(m['confidence'] for m in all_mechanisms) / len(all_mechanisms) if all_mechanisms else 1.0

        return {
            'predicted_properties': predicted_properties,
            'mechanistic_explanation': mechanistic_explanation,
            'confidence': avg_confidence,
            'paths_found': len(paths),
            'reasoning_type': 'kg_only_template',
            'all_mechanisms': all_mechanisms
        }

    def _template_fill_inverse(self, paths: List[List[str]]) -> Dict[str, Any]:
        """
        Generate inverse design using pure template filling.

        NO LLM - extract synthesis conditions from reverse paths.

        Args:
            paths: Found causal paths (reversed)

        Returns:
            Structured synthesis dict
        """
        if not paths:
            return {
                'suggested_synthesis_conditions': {},
                'mechanistic_explanation': 'No causal pathway found in knowledge graph.',
                'confidence': 0.0,
                'paths_found': 0,
                'reasoning_type': 'kg_only_inverse_no_match'
            }

        # Extract synthesis parameters from path start nodes
        synthesis_conditions = {}
        all_mechanisms = []

        for path in paths:
            # First node = synthesis parameter (reversed path)
            # Last node = property (reversed path)
            mechs = self._extract_mechanisms_from_path(path)
            all_mechanisms.extend(mechs)

            # Extract parameters from path nodes
            for node in path:
                node_lower = node.lower()
                if 'temperature' in node_lower or 'temp' in node_lower:
                    synthesis_conditions['temperature'] = node
                elif 'pressure' in node_lower:
                    synthesis_conditions['pressure'] = node
                elif 'time' in node_lower or 'duration' in node_lower:
                    synthesis_conditions['time'] = node
                elif 'method' in node_lower or 'cvd' in node_lower or 'mbe' in node_lower:
                    synthesis_conditions['method'] = node
                elif 'atmosphere' in node_lower or 'gas' in node_lower:
                    synthesis_conditions['atmosphere'] = node

        # Concatenate mechanisms
        explanation_parts = []
        for i, path in enumerate(paths, 1):
            # Reverse for readability (property → synthesis)
            reversed_path = list(reversed(path))
            path_str = ' → '.join(reversed_path)
            explanation_parts.append(f"Pathway {i}: {path_str}")

            mechs = self._extract_mechanisms_from_path(path)
            for m in reversed(mechs):  # Reverse mechanism order too
                explanation_parts.append(f"  • {m['to']} ← {m['from']}: {m['mechanism']}")

        mechanistic_explanation = '\n'.join(explanation_parts)

        avg_confidence = sum(m['confidence'] for m in all_mechanisms) / len(all_mechanisms) if all_mechanisms else 1.0

        return {
            'suggested_synthesis_conditions': synthesis_conditions,
            'mechanistic_explanation': mechanistic_explanation,
            'confidence': avg_confidence,
            'paths_found': len(paths),
            'reasoning_type': 'kg_only_inverse_template',
            'all_mechanisms': all_mechanisms
        }

    def forward_prediction(self, synthesis_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict properties from synthesis using pure KG traversal.

        Args:
            synthesis_inputs: Synthesis parameters

        Returns:
            Prediction dict (template-filled from KG)
        """
        logger.info("Forward prediction (KG-only, no LLM)")

        # Extract keywords from inputs
        input_keywords = []
        for key, value in synthesis_inputs.items():
            if isinstance(value, str):
                input_keywords.extend(value.split())
            else:
                input_keywords.append(str(value))

        # Property keywords (targets)
        property_keywords = [
            'mobility', 'conductivity', 'carrier', 'band gap', 'doping',
            'property', 'electronic', 'optical', 'thermal', 'mechanical'
        ]

        # Find paths
        paths = self._find_paths(input_keywords, property_keywords)

        # Template fill
        result = self._template_fill_forward(paths)

        return result

    def inverse_design(self, desired_properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design synthesis from properties using pure KG traversal.

        Args:
            desired_properties: Target properties

        Returns:
            Synthesis dict (template-filled from KG)
        """
        logger.info("Inverse design (KG-only, no LLM)")

        # Extract property keywords
        property_keywords = []
        for key, value in desired_properties.items():
            if isinstance(value, str):
                property_keywords.extend(value.split())
            property_keywords.append(key)

        # Synthesis parameter keywords
        synthesis_keywords = [
            'temperature', 'pressure', 'time', 'atmosphere', 'method',
            'CVD', 'MBE', 'PVD', 'annealing', 'oxidation', 'reduction'
        ]

        # Find reverse paths (property → synthesis)
        paths = self._find_paths(property_keywords, synthesis_keywords)

        # Template fill
        result = self._template_fill_inverse(paths)

        return result

    def get_kg_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        return {
            'num_nodes': self.kg.number_of_nodes(),
            'num_edges': self.kg.number_of_edges(),
            'avg_degree': sum(dict(self.kg.degree()).values()) / self.kg.number_of_nodes(),
            'is_dag': nx.is_directed_acyclic_graph(self.kg),
            'weakly_connected_components': nx.number_weakly_connected_components(self.kg),
            'nodes': list(self.kg.nodes()),
            'edges': list(self.kg.edges())
        }


def main():
    """Test the KG-only implementation."""
    print("="*60)
    print("Testing KG-Only Implementation (No LLM)")
    print("="*60)

    try:
        # Initialize
        kg_only = KGOnly(kg_file="../../data/KG/outputs/kg_example_2d_doping_enriched.json")

        # Print KG stats
        stats = kg_only.get_kg_statistics()
        print(f"\nKG Statistics:")
        print(f"  Nodes: {stats['num_nodes']}")
        print(f"  Edges: {stats['num_edges']}")
        print(f"  Avg degree: {stats['avg_degree']:.2f}")
        print(f"  Is DAG: {stats['is_dag']}")
        print(f"  Node list: {stats['nodes'][:5]}..." if len(stats['nodes']) > 5 else f"  Node list: {stats['nodes']}")

        # Test 1: Forward prediction
        print("\n[Test 1] Forward Prediction")
        print("-" * 60)

        synthesis_input = {
            "method": "CVD",
            "temperature": "750°C",
            "material": "MoS2",
            "dopant": "Nb"
        }

        print(f"Input: {json.dumps(synthesis_input, indent=2)}")

        result = kg_only.forward_prediction(synthesis_input)

        print(f"\nOutput:")
        print(f"  Predicted properties: {json.dumps(result.get('predicted_properties', {}), indent=2)}")
        print(f"  Confidence: {result.get('confidence', 0.0)}")
        print(f"  Paths found: {result.get('paths_found', 0)}")
        print(f"  Reasoning type: {result.get('reasoning_type', 'unknown')}")
        print(f"  Explanation: {result.get('mechanistic_explanation', 'N/A')[:200]}...")

        # Test 2: Inverse design
        print("\n[Test 2] Inverse Design")
        print("-" * 60)

        desired_props = {
            "carrier_type": "n-type",
            "mobility": "high",
            "material": "MoS2"
        }

        print(f"Desired properties: {json.dumps(desired_props, indent=2)}")

        result = kg_only.inverse_design(desired_props)

        print(f"\nOutput:")
        print(f"  Suggested conditions: {json.dumps(result.get('suggested_synthesis_conditions', {}), indent=2)}")
        print(f"  Confidence: {result.get('confidence', 0.0)}")
        print(f"  Paths found: {result.get('paths_found', 0)}")
        print(f"  Reasoning type: {result.get('reasoning_type', 'unknown')}")

        # Test 3: Test with query that should match KG
        print("\n[Test 3] Query Matching KG Content")
        print("-" * 60)

        # Use actual node from KG
        if stats['nodes']:
            test_node = stats['nodes'][0]
            print(f"Testing with KG node: {test_node}")

            synthesis_with_match = {
                "parameter": test_node
            }

            result = kg_only.forward_prediction(synthesis_with_match)
            print(f"  Paths found: {result.get('paths_found', 0)}")
            print(f"  Confidence: {result.get('confidence', 0.0)}")

        print("\n" + "="*60)
        print("✓ KG-Only tests completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
