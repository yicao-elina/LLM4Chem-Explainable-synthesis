"""
Knowledge Graph Quality Diagnostics

Analyzes KG structure, density, coverage, and quality metrics.
Generates comprehensive report for ARIA evaluation.

Author: ARIA Team
Date: 2026-02-01
"""

import json
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KGDiagnostics:
    """Comprehensive KG quality analysis."""

    def __init__(self, kg_file: str):
        """
        Initialize KG diagnostics.

        Args:
            kg_file: Path to KG JSON file
        """
        self.kg_file = Path(kg_file)
        self.kg = None
        self.raw_data = None
        self.embedding_model = None

        self._load_kg()

    def _load_kg(self):
        """Load KG from JSON."""
        logger.info(f"Loading KG from: {self.kg_file}")

        if not self.kg_file.exists():
            raise FileNotFoundError(f"KG file not found: {self.kg_file}")

        with open(self.kg_file, 'r') as f:
            self.raw_data = json.load(f)

        # Build NetworkX graph
        G = nx.DiGraph()

        relationships = self.raw_data.get('causal_relationships', self.raw_data if isinstance(self.raw_data, list) else [])

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
                    confidence=rel.get('confidence', 1.0),
                    source_file=rel.get('source_file', '')
                )

        self.kg = G
        logger.info(f"KG loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    def analyze_structure(self) -> Dict[str, Any]:
        """Analyze graph structure metrics."""
        logger.info("Analyzing graph structure...")

        G = self.kg

        # Basic counts
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        # Density (actual edges / possible edges)
        density = nx.density(G)

        # Degree statistics
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        total_degrees = dict(G.degree())

        avg_in_degree = np.mean(list(in_degrees.values())) if in_degrees else 0
        avg_out_degree = np.mean(list(out_degrees.values())) if out_degrees else 0
        avg_degree = np.mean(list(total_degrees.values())) if total_degrees else 0

        # Node types
        root_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]  # Synthesis params
        leaf_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]  # Properties
        intermediate_nodes = [n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0]

        # Path statistics
        is_dag = nx.is_directed_acyclic_graph(G)
        weakly_connected = nx.number_weakly_connected_components(G)
        strongly_connected = nx.number_strongly_connected_components(G)

        # Longest path
        try:
            longest_path_length = max(len(nx.dag_longest_path(G)) if is_dag else 0, 0)
        except:
            longest_path_length = 0

        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'avg_in_degree': avg_in_degree,
            'avg_out_degree': avg_out_degree,
            'avg_degree': avg_degree,
            'max_in_degree': max(in_degrees.values()) if in_degrees else 0,
            'max_out_degree': max(out_degrees.values()) if out_degrees else 0,
            'num_root_nodes': len(root_nodes),
            'num_leaf_nodes': len(leaf_nodes),
            'num_intermediate_nodes': len(intermediate_nodes),
            'is_dag': is_dag,
            'weakly_connected_components': weakly_connected,
            'strongly_connected_components': strongly_connected,
            'longest_path_length': longest_path_length,
            'root_nodes': root_nodes[:10],  # Sample
            'leaf_nodes': leaf_nodes[:10]   # Sample
        }

    def analyze_content(self) -> Dict[str, Any]:
        """Analyze content quality metrics."""
        logger.info("Analyzing content quality...")

        G = self.kg

        # Mechanism coverage
        edges_with_mechanism = 0
        edges_without_mechanism = 0
        mechanism_lengths = []

        for u, v, data in G.edges(data=True):
            mechanism = data.get('mechanism', '')
            if mechanism and mechanism.strip():
                edges_with_mechanism += 1
                mechanism_lengths.append(len(mechanism))
            else:
                edges_without_mechanism += 1

        mechanism_coverage = edges_with_mechanism / G.number_of_edges() if G.number_of_edges() > 0 else 0

        # Affected property coverage
        edges_with_property = sum(1 for u, v, d in G.edges(data=True) if d.get('affected_property'))
        property_coverage = edges_with_property / G.number_of_edges() if G.number_of_edges() > 0 else 0

        # Confidence scores
        confidences = [d.get('confidence', 1.0) for u, v, d in G.edges(data=True)]
        avg_confidence = np.mean(confidences) if confidences else 0

        # Unique affected properties
        affected_properties = [d.get('affected_property', '') for u, v, d in G.edges(data=True) if d.get('affected_property')]
        unique_properties = set(affected_properties)

        return {
            'mechanism_coverage': mechanism_coverage,
            'edges_with_mechanism': edges_with_mechanism,
            'edges_without_mechanism': edges_without_mechanism,
            'avg_mechanism_length': np.mean(mechanism_lengths) if mechanism_lengths else 0,
            'property_coverage': property_coverage,
            'edges_with_property': edges_with_property,
            'avg_confidence': avg_confidence,
            'num_unique_properties': len(unique_properties),
            'unique_properties': list(unique_properties)[:20]  # Sample
        }

    def analyze_coverage(self) -> Dict[str, Any]:
        """Analyze KG coverage for typical queries."""
        logger.info("Analyzing query coverage...")

        G = self.kg

        # Test queries (common materials science questions)
        test_queries = [
            {'type': 'forward', 'keywords': ['temperature', 'CVD'], 'target': ['mobility', 'conductivity']},
            {'type': 'forward', 'keywords': ['pressure', 'doping'], 'target': ['carrier', 'concentration']},
            {'type': 'forward', 'keywords': ['annealing', 'oxygen'], 'target': ['defect', 'property']},
            {'type': 'inverse', 'keywords': ['n-type', 'high mobility'], 'target': ['temperature', 'method']},
            {'type': 'inverse', 'keywords': ['p-type', 'doping'], 'target': ['dopant', 'concentration']},
        ]

        coverage_stats = {
            'total_queries': len(test_queries),
            'queries_with_match': 0,
            'queries_without_match': 0,
            'avg_paths_per_query': 0,
            'query_details': []
        }

        total_paths = 0

        for query in test_queries:
            # Find nodes matching keywords
            start_nodes = set()
            end_nodes = set()

            for node in G.nodes():
                node_lower = node.lower()
                if any(kw.lower() in node_lower for kw in query['keywords']):
                    start_nodes.add(node)
                if any(kw.lower() in node_lower for kw in query['target']):
                    end_nodes.add(node)

            # Find paths
            paths = []
            for start in start_nodes:
                for end in end_nodes:
                    if nx.has_path(G, start, end):
                        for path in nx.all_simple_paths(G, start, end, cutoff=4):
                            paths.append(path)

            num_paths = len(paths)
            total_paths += num_paths

            if num_paths > 0:
                coverage_stats['queries_with_match'] += 1

            coverage_stats['query_details'].append({
                'type': query['type'],
                'keywords': query['keywords'],
                'target': query['target'],
                'num_paths': num_paths,
                'has_match': num_paths > 0
            })

        coverage_stats['queries_without_match'] = coverage_stats['total_queries'] - coverage_stats['queries_with_match']
        coverage_stats['avg_paths_per_query'] = total_paths / len(test_queries) if test_queries else 0
        coverage_stats['coverage_rate'] = coverage_stats['queries_with_match'] / coverage_stats['total_queries'] if test_queries else 0

        return coverage_stats

    def analyze_diversity(self) -> Dict[str, Any]:
        """Analyze semantic diversity of nodes."""
        logger.info("Analyzing semantic diversity...")

        # Load embedding model
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        G = self.kg
        nodes = list(G.nodes())

        if len(nodes) < 2:
            return {'diversity_score': 0, 'avg_similarity': 0, 'note': 'Too few nodes'}

        # Compute embeddings
        embeddings = self.embedding_model.encode(nodes)

        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)

        # Remove diagonal (self-similarity)
        np.fill_diagonal(similarities, 0)

        # Average similarity (lower = more diverse)
        avg_similarity = np.mean(similarities)
        diversity_score = 1 - avg_similarity  # Higher = more diverse

        # Find most similar pairs
        similar_pairs = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                similar_pairs.append((nodes[i], nodes[j], similarities[i][j]))

        similar_pairs.sort(key=lambda x: x[2], reverse=True)

        return {
            'diversity_score': diversity_score,
            'avg_similarity': avg_similarity,
            'most_similar_pairs': [(p[0], p[1], float(p[2])) for p in similar_pairs[:5]],
            'least_similar_pairs': [(p[0], p[1], float(p[2])) for p in similar_pairs[-5:]]
        }

    def estimate_kg_gaps(self, structure: Dict, coverage: Dict) -> Dict[str, Any]:
        """Estimate how much data is needed to improve KG."""
        logger.info("Estimating KG gaps...")

        current_edges = structure['num_edges']
        coverage_rate = coverage['coverage_rate']

        # Estimate needed edges for different coverage targets
        estimates = {}

        for target_coverage in [0.5, 0.7, 0.9]:
            if coverage_rate > 0:
                needed_edges = int(current_edges * (target_coverage / coverage_rate))
                additional_edges = max(0, needed_edges - current_edges)
            else:
                additional_edges = "unknown (current coverage = 0)"

            estimates[f'{int(target_coverage*100)}%_coverage'] = {
                'target_edges': needed_edges if coverage_rate > 0 else "unknown",
                'additional_edges_needed': additional_edges
            }

        # Estimate papers needed (assuming ~2-5 edges per paper)
        papers_needed = {}
        for target, data in estimates.items():
            if isinstance(data['additional_edges_needed'], int):
                papers_needed[target] = {
                    'min_papers': data['additional_edges_needed'] // 5,
                    'max_papers': data['additional_edges_needed'] // 2
                }
            else:
                papers_needed[target] = "unknown"

        return {
            'current_coverage': coverage_rate,
            'coverage_estimates': estimates,
            'papers_needed_estimates': papers_needed,
            'recommendation': self._get_recommendation(structure, coverage)
        }

    def _get_recommendation(self, structure: Dict, coverage: Dict) -> str:
        """Generate recommendation based on KG quality."""
        edges = structure['num_edges']
        coverage_rate = coverage['coverage_rate']

        if edges < 50:
            return "CRITICAL: KG too small (<50 edges). Need 100-500 edges for meaningful evaluation."
        elif edges < 100:
            return "LOW: KG small (50-100 edges). Recommend enrichment to 200+ edges."
        elif coverage_rate < 0.3:
            return "MODERATE: Decent size but low coverage. Focus on improving coverage of common queries."
        elif coverage_rate < 0.5:
            return "GOOD: Good size and coverage. Can proceed with testing. Enrichment will improve results."
        else:
            return "EXCELLENT: High coverage. Proceed with testing confidently."

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive KG quality report."""
        logger.info("Generating comprehensive report...")

        report = {
            'kg_file': str(self.kg_file),
            'timestamp': '2026-02-01',
            'structure': self.analyze_structure(),
            'content': self.analyze_content(),
            'coverage': self.analyze_coverage(),
            'diversity': self.analyze_diversity()
        }

        report['gaps'] = self.estimate_kg_gaps(report['structure'], report['coverage'])

        return report

    def print_report(self, report: Dict[str, Any]):
        """Print formatted report."""
        print("\n" + "="*70)
        print("KNOWLEDGE GRAPH QUALITY DIAGNOSTIC REPORT")
        print("="*70)

        print(f"\nKG File: {report['kg_file']}")

        # Structure
        print("\n" + "-"*70)
        print("1. GRAPH STRUCTURE")
        print("-"*70)
        s = report['structure']
        print(f"  Nodes:                 {s['num_nodes']}")
        print(f"  Edges:                 {s['num_edges']}")
        print(f"  Density:               {s['density']:.4f}")
        print(f"  Avg Degree:            {s['avg_degree']:.2f}")
        print(f"  Root Nodes:            {s['num_root_nodes']} (synthesis parameters)")
        print(f"  Leaf Nodes:            {s['num_leaf_nodes']} (properties)")
        print(f"  Intermediate Nodes:    {s['num_intermediate_nodes']}")
        print(f"  Is DAG:                {s['is_dag']}")
        print(f"  Longest Path:          {s['longest_path_length']} hops")
        print(f"  Weakly Connected:      {s['weakly_connected_components']} component(s)")

        # Content
        print("\n" + "-"*70)
        print("2. CONTENT QUALITY")
        print("-"*70)
        c = report['content']
        print(f"  Mechanism Coverage:    {c['mechanism_coverage']:.1%} ({c['edges_with_mechanism']}/{c['edges_with_mechanism']+c['edges_without_mechanism']})")
        print(f"  Avg Mechanism Length:  {c['avg_mechanism_length']:.0f} chars")
        print(f"  Property Coverage:     {c['property_coverage']:.1%}")
        print(f"  Avg Confidence:        {c['avg_confidence']:.2f}")
        print(f"  Unique Properties:     {c['num_unique_properties']}")

        # Coverage
        print("\n" + "-"*70)
        print("3. QUERY COVERAGE")
        print("-"*70)
        cov = report['coverage']
        print(f"  Test Queries:          {cov['total_queries']}")
        print(f"  Queries with Match:    {cov['queries_with_match']} ({cov['coverage_rate']:.1%})")
        print(f"  Queries without Match: {cov['queries_without_match']}")
        print(f"  Avg Paths per Query:   {cov['avg_paths_per_query']:.1f}")

        # Diversity
        print("\n" + "-"*70)
        print("4. SEMANTIC DIVERSITY")
        print("-"*70)
        d = report['diversity']
        print(f"  Diversity Score:       {d['diversity_score']:.3f} (higher = more diverse)")
        print(f"  Avg Node Similarity:   {d['avg_similarity']:.3f} (lower = more diverse)")

        # Gaps
        print("\n" + "-"*70)
        print("5. KG GAPS & RECOMMENDATIONS")
        print("-"*70)
        g = report['gaps']
        print(f"  Current Coverage:      {g['current_coverage']:.1%}")
        print(f"\n  To achieve 50% coverage:")
        if isinstance(g['coverage_estimates']['50%_coverage']['additional_edges_needed'], int):
            print(f"    Additional edges:    {g['coverage_estimates']['50%_coverage']['additional_edges_needed']}")
            print(f"    Papers needed:       {g['papers_needed_estimates']['50%_coverage']['min_papers']}-{g['papers_needed_estimates']['50%_coverage']['max_papers']}")
        else:
            print(f"    {g['coverage_estimates']['50%_coverage']['additional_edges_needed']}")

        print(f"\n  To achieve 70% coverage:")
        if isinstance(g['coverage_estimates']['70%_coverage']['additional_edges_needed'], int):
            print(f"    Additional edges:    {g['coverage_estimates']['70%_coverage']['additional_edges_needed']}")
            print(f"    Papers needed:       {g['papers_needed_estimates']['70%_coverage']['min_papers']}-{g['papers_needed_estimates']['70%_coverage']['max_papers']}")
        else:
            print(f"    {g['coverage_estimates']['70%_coverage']['additional_edges_needed']}")

        print(f"\n  RECOMMENDATION: {g['recommendation']}")

        print("\n" + "="*70)

    def save_report(self, report: Dict[str, Any], output_file: str = "kg_quality_report.json"):
        """Save report to JSON."""
        import numpy as np

        # Convert numpy types to Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj

        report_clean = convert_numpy(report)

        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(report_clean, f, indent=2)
        logger.info(f"Report saved to: {output_path}")


def main():
    """Run KG diagnostics."""
    kg_file = "/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD/KG/outputs/combined_doping_data.json"

    print("\n" + "="*70)
    print("Starting KG Quality Diagnostics...")
    print("="*70)

    try:
        diagnostics = KGDiagnostics(kg_file)
        report = diagnostics.generate_report()
        diagnostics.print_report(report)
        diagnostics.save_report(report, "kg_quality_report.json")

        print("\n✅ Diagnostics complete!")
        print(f"Report saved to: kg_quality_report.json")

    except Exception as e:
        print(f"\n❌ Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
