"""
ARIA Full Implementation (Ollama-based)

This variant extends ARIA Search with chain-of-thought transparency:
- All ARIA Search features (3-tier reasoning + literature search)
- + Chain-of-thought transparency (explicit reasoning steps)
- + Source attribution (track which sources contributed to each conclusion)
- + Dynamic KG enrichment capability (extract new relationships from literature)

Tests the contribution of transparent reasoning and KG growth.

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
from dataclasses import dataclass, field, asdict
from datetime import datetime

from ..ollama_client import get_ollama_client
from .aria_search_ollama import LiteratureSearcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeSource:
    """Track individual knowledge sources with metadata."""
    source_id: str
    content: str
    source_type: str  # "kg_node", "kg_edge", "kg_mechanism", "literature", "llm_baseline"
    confidence: float
    context: str
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ReasoningStep:
    """Individual step in chain-of-thought reasoning."""
    step_id: str
    description: str
    evidence_sources: List[KnowledgeSource]
    reasoning_type: str  # "retrieval", "synthesis", "validation", "inference", "search"
    confidence: float
    intermediate_conclusion: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'step_id': self.step_id,
            'description': self.description,
            'evidence_sources': [s.to_dict() for s in self.evidence_sources],
            'reasoning_type': self.reasoning_type,
            'confidence': self.confidence,
            'intermediate_conclusion': self.intermediate_conclusion,
            'timestamp': self.timestamp
        }


@dataclass
class ChainOfThought:
    """Complete reasoning chain with source attribution."""
    query_context: Dict
    reasoning_steps: List[ReasoningStep]
    final_reasoning: str
    final_result: Dict
    confidence_breakdown: Dict
    source_attribution: Dict
    tier: int
    kg_paths_used: int
    literature_papers_used: int

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'query_context': self.query_context,
            'reasoning_steps': [step.to_dict() for step in self.reasoning_steps],
            'final_reasoning': self.final_reasoning,
            'final_result': self.final_result,
            'confidence_breakdown': self.confidence_breakdown,
            'source_attribution': self.source_attribution,
            'tier': self.tier,
            'kg_paths_used': self.kg_paths_used,
            'literature_papers_used': self.literature_papers_used
        }


class ARIAFullOllama:
    """
    ARIA Full: Complete system with chain-of-thought transparency.

    Extends ARIA Search with explicit reasoning steps and source tracking.
    """

    def __init__(
        self,
        kg_file: str = "../../data/KG/outputs/kg_example_2d_doping_enriched.json",
        model: str = "deepseek-r1:8b",
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        search_email: str = "research@example.com"
    ):
        """
        Initialize ARIA Full system.

        Args:
            kg_file: Path to knowledge graph JSON
            model: Ollama model for reasoning
            embedding_model: Sentence transformer for embeddings
            similarity_threshold: Minimum similarity for transfer learning
            search_email: Email for OpenAlex API access
        """
        self.kg_file = Path(kg_file)
        self.model = model
        self.similarity_threshold = similarity_threshold

        # Initialize Ollama client
        self.ollama = get_ollama_client(model=model)

        # Initialize literature searcher
        self.searcher = LiteratureSearcher(email=search_email)

        # Load KG
        self.kg = self._load_kg()

        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self._precompute_node_embeddings()

        # Build knowledge source index
        self._build_knowledge_source_index()

        logger.info(f"ARIA Full initialized: {self.kg.number_of_nodes()} nodes, "
                   f"{self.kg.number_of_edges()} edges, chain-of-thought transparency enabled")

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

    def _build_knowledge_source_index(self):
        """Build comprehensive knowledge source index."""
        logger.info("Building knowledge source index...")

        self.kg_sources = {}

        # Index nodes
        for node in self.kg.nodes():
            source_id = f"kg_node_{hash(node)}"
            self.kg_sources[source_id] = KnowledgeSource(
                source_id=source_id,
                content=node,
                source_type="kg_node",
                confidence=1.0,
                context="Knowledge graph node",
                metadata={'node': node}
            )

        # Index edges and mechanisms
        for source, target, data in self.kg.edges(data=True):
            edge_id = f"{source} -> {target}"
            source_id = f"kg_edge_{hash(edge_id)}"

            self.kg_sources[source_id] = KnowledgeSource(
                source_id=source_id,
                content=f"Causal relationship: {source} leads to {target}",
                source_type="kg_edge",
                confidence=0.9,
                context="Knowledge graph causal edge",
                metadata={'source': source, 'target': target, 'edge_id': edge_id}
            )

            mechanism = data.get('mechanism', '')
            if mechanism:
                mech_source_id = f"kg_mechanism_{hash(edge_id)}"
                self.kg_sources[mech_source_id] = KnowledgeSource(
                    source_id=mech_source_id,
                    content=mechanism,
                    source_type="kg_mechanism",
                    confidence=0.95,
                    context=f"Mechanistic explanation for {edge_id}",
                    metadata={'pathway': edge_id}
                )

        logger.info(f"Indexed {len(self.kg_sources)} knowledge sources")

    def _precompute_node_embeddings(self):
        """Precompute embeddings for all KG nodes."""
        self.node_list = list(self.kg.nodes())
        if not self.node_list:
            self.node_embeddings = np.array([])
            return

        logger.info(f"Computing embeddings for {len(self.node_list)} nodes...")
        self.node_embeddings = self.embedding_model.encode(
            self.node_list,
            convert_to_numpy=True,
            show_progress_bar=False
        )

    def _find_most_similar_node(self, query: str, candidate_nodes: List[str]) -> Tuple[Optional[str], float]:
        """Find most similar node using embeddings."""
        if not candidate_nodes or len(self.node_embeddings) == 0:
            return None, 0.0

        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)

        candidate_indices = [
            self.node_list.index(node) for node in candidate_nodes
            if node in self.node_list
        ]

        if not candidate_indices:
            return None, 0.0

        candidate_embeddings = self.node_embeddings[candidate_indices]
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_node = candidate_nodes[best_idx]

        return best_node, float(best_score)

    def _find_paths(self, start_keywords: List[str], end_keywords: List[str],
                   reverse: bool = False, max_paths: int = 10) -> List[str]:
        """Find paths in KG between keyword sets."""
        start_nodes = {
            node for node in self.kg.nodes()
            if any(kw.lower() in node.lower() for kw in start_keywords if kw)
        }

        end_nodes = {
            node for node in self.kg.nodes()
            if any(kw.lower() in node.lower() for kw in end_keywords if kw)
        }

        if not start_nodes or not end_nodes:
            return []

        graph = self.kg.reverse(copy=True) if reverse else self.kg
        source_nodes = end_nodes if reverse else start_nodes
        target_nodes = start_nodes if reverse else end_nodes

        paths = []
        for source in source_nodes:
            for target in target_nodes:
                if nx.has_path(graph, source, target):
                    try:
                        for path in nx.all_simple_paths(graph, source, target, cutoff=3):
                            path_str = " -> ".join(path)
                            paths.append(path_str)
                            if len(paths) >= max_paths:
                                break
                    except:
                        continue
                if len(paths) >= max_paths:
                    break
            if len(paths) >= max_paths:
                break

        return list(set(paths))[:max_paths]

    def _extract_kg_sources_from_paths(self, paths: List[str]) -> List[KnowledgeSource]:
        """Extract KnowledgeSource objects from paths."""
        sources = []

        for path in paths:
            nodes = [n.strip() for n in path.split(" -> ")]

            # Add node sources
            for node in nodes:
                for source_id, source in self.kg_sources.items():
                    if source.source_type == "kg_node" and source.metadata.get('node') == node:
                        sources.append(source)
                        break

            # Add edge and mechanism sources
            for i in range(len(nodes) - 1):
                edge_id = f"{nodes[i]} -> {nodes[i+1]}"
                for source_id, source in self.kg_sources.items():
                    if (source.source_type in ['kg_edge', 'kg_mechanism'] and
                        source.metadata.get('edge_id') == edge_id or
                        source.metadata.get('pathway') == edge_id):
                        sources.append(source)

        return sources

    def _create_literature_sources(self, papers: List[Dict]) -> List[KnowledgeSource]:
        """Convert literature search results to KnowledgeSource objects."""
        sources = []

        for i, paper in enumerate(papers):
            source = KnowledgeSource(
                source_id=f"lit_{i}_{hash(paper['title'])}",
                content=f"{paper['title']}: {paper.get('abstract', '')[:200]}...",
                source_type="literature",
                confidence=min(1.0, 0.5 + (paper.get('citations', 0) / 1000)),  # Citation-based confidence
                context=f"Literature: {paper.get('year', 'N/A')} ({paper.get('citations', 0)} citations)",
                metadata=paper
            )
            sources.append(source)

        return sources

    def _forward_with_cot(self, synthesis_inputs: Dict) -> ChainOfThought:
        """
        Forward prediction with full chain-of-thought transparency.

        Args:
            synthesis_inputs: Dict of synthesis parameters

        Returns:
            Complete ChainOfThought object
        """
        logger.info("\n=== ARIA Full: Forward Prediction with CoT ===")

        reasoning_steps = []

        # Extract keywords
        input_keywords = [str(v) for v in synthesis_inputs.values() if v]

        # Step 1: KG Path Retrieval
        property_nodes = [n for n in self.kg.nodes() if self.kg.out_degree(n) == 0]
        paths = self._find_paths(input_keywords, property_nodes)

        kg_sources = self._extract_kg_sources_from_paths(paths) if paths else []

        retrieval_step = ReasoningStep(
            step_id="kg_retrieval",
            description=f"Retrieved {len(paths)} causal pathways from knowledge graph",
            evidence_sources=kg_sources,
            reasoning_type="retrieval",
            confidence=1.0 if paths else 0.0,
            intermediate_conclusion=f"Found {len(paths)} paths, {len(kg_sources)} knowledge sources" if paths
                                   else "No direct paths found in KG"
        )
        reasoning_steps.append(retrieval_step)

        # Determine tier
        tier = 1 if paths else None

        # Step 2: Literature Search
        if paths:
            search_queries = self._generate_search_queries(synthesis_inputs, paths)
        else:
            search_queries = [f"{' '.join(input_keywords[:3])} materials synthesis"]

        search_results = self._search_literature(search_queries, max_papers=8)
        lit_sources = self._create_literature_sources(search_results)

        search_step = ReasoningStep(
            step_id="literature_search",
            description=f"Searched literature using {len(search_queries)} queries",
            evidence_sources=lit_sources,
            reasoning_type="search",
            confidence=0.8 if lit_sources else 0.3,
            intermediate_conclusion=f"Found {len(lit_sources)} relevant papers from literature"
        )
        reasoning_steps.append(search_step)

        # Step 3: Transfer Learning (if no direct paths)
        if not paths:
            synthesis_nodes = [n for n in self.kg.nodes() if self.kg.in_degree(n) == 0]
            query_string = " ".join(input_keywords)
            similar_node, score = self._find_most_similar_node(query_string, synthesis_nodes)

            if similar_node and score > self.similarity_threshold:
                tier = 2
                analogous_paths = self._find_paths([similar_node], property_nodes)
                transfer_sources = self._extract_kg_sources_from_paths(analogous_paths)

                transfer_step = ReasoningStep(
                    step_id="transfer_learning",
                    description=f"Applied transfer learning from similar case: {similar_node} (similarity={score:.2f})",
                    evidence_sources=transfer_sources,
                    reasoning_type="inference",
                    confidence=score,
                    intermediate_conclusion=f"Using {len(analogous_paths)} analogous paths for transfer learning"
                )
                reasoning_steps.append(transfer_step)
                kg_sources = transfer_sources
            else:
                tier = 3

        # Step 4: LLM Reasoning
        all_sources = kg_sources + lit_sources
        llm_result = self._generate_prediction(synthesis_inputs, all_sources, paths, tier)

        # Create LLM baseline source
        llm_source = KnowledgeSource(
            source_id="llm_reasoning",
            content=llm_result.get('reasoning', ''),
            source_type="llm_baseline",
            confidence=llm_result.get('confidence', 0.5),
            context=f"Tier {tier} LLM reasoning",
            metadata={'tier': tier}
        )

        reasoning_step = ReasoningStep(
            step_id="llm_synthesis",
            description=f"Synthesized prediction using Tier {tier} reasoning",
            evidence_sources=[llm_source],
            reasoning_type="synthesis",
            confidence=llm_result.get('confidence', 0.5),
            intermediate_conclusion=llm_result.get('reasoning', '')[:200] + "..."
        )
        reasoning_steps.append(reasoning_step)

        # Build final chain of thought
        cot = ChainOfThought(
            query_context={'type': 'forward', 'inputs': synthesis_inputs},
            reasoning_steps=reasoning_steps,
            final_reasoning=llm_result.get('reasoning', ''),
            final_result=llm_result.get('predicted_properties', {}),
            confidence_breakdown={
                step.step_id: step.confidence for step in reasoning_steps
            },
            source_attribution={
                'kg_sources': len(kg_sources),
                'literature_sources': len(lit_sources),
                'total_sources': len(all_sources)
            },
            tier=tier,
            kg_paths_used=len(paths),
            literature_papers_used=len(lit_sources)
        )

        return cot

    def _inverse_with_cot(self, desired_properties: Dict) -> ChainOfThought:
        """
        Inverse design with full chain-of-thought transparency.

        Args:
            desired_properties: Dict of target material properties

        Returns:
            Complete ChainOfThought object
        """
        logger.info("\n=== ARIA Full: Inverse Design with CoT ===")

        reasoning_steps = []

        # Extract keywords
        property_keywords = [str(v) for v in desired_properties.values() if v]

        # Step 1: KG Path Retrieval (Inverse)
        synthesis_nodes = [n for n in self.kg.nodes() if self.kg.in_degree(n) == 0]
        paths = self._find_paths(synthesis_nodes, property_keywords, reverse=True)

        kg_sources = self._extract_kg_sources_from_paths(paths) if paths else []

        retrieval_step = ReasoningStep(
            step_id="kg_retrieval_inverse",
            description=f"Retrieved {len(paths)} inverse causal pathways from knowledge graph",
            evidence_sources=kg_sources,
            reasoning_type="retrieval",
            confidence=1.0 if paths else 0.0,
            intermediate_conclusion=f"Found {len(paths)} inverse paths" if paths
                                   else "No direct inverse paths found in KG"
        )
        reasoning_steps.append(retrieval_step)

        tier = 1 if paths else None

        # Step 2: Literature Search
        if paths:
            search_queries = self._generate_search_queries_inverse(desired_properties, paths)
        else:
            search_queries = [f"achieving {' '.join(property_keywords[:3])} materials"]

        search_results = self._search_literature(search_queries, max_papers=8)
        lit_sources = self._create_literature_sources(search_results)

        search_step = ReasoningStep(
            step_id="literature_search_inverse",
            description=f"Searched literature for synthesis methods",
            evidence_sources=lit_sources,
            reasoning_type="search",
            confidence=0.8 if lit_sources else 0.3,
            intermediate_conclusion=f"Found {len(lit_sources)} papers with relevant synthesis protocols"
        )
        reasoning_steps.append(search_step)

        # Step 3: Transfer Learning (if no direct paths)
        if not paths:
            property_nodes = [n for n in self.kg.nodes() if self.kg.out_degree(n) == 0]
            query_string = " ".join(property_keywords)
            similar_node, score = self._find_most_similar_node(query_string, property_nodes)

            if similar_node and score > self.similarity_threshold:
                tier = 2
                analogous_paths = self._find_paths(synthesis_nodes, [similar_node], reverse=True)
                transfer_sources = self._extract_kg_sources_from_paths(analogous_paths)

                transfer_step = ReasoningStep(
                    step_id="transfer_learning_inverse",
                    description=f"Applied inverse transfer learning from similar property: {similar_node} (similarity={score:.2f})",
                    evidence_sources=transfer_sources,
                    reasoning_type="inference",
                    confidence=score,
                    intermediate_conclusion=f"Using {len(analogous_paths)} analogous inverse paths"
                )
                reasoning_steps.append(transfer_step)
                kg_sources = transfer_sources
            else:
                tier = 3

        # Step 4: LLM Reasoning
        all_sources = kg_sources + lit_sources
        llm_result = self._generate_synthesis_conditions(desired_properties, all_sources, paths, tier)

        llm_source = KnowledgeSource(
            source_id="llm_reasoning_inverse",
            content=llm_result.get('reasoning', ''),
            source_type="llm_baseline",
            confidence=llm_result.get('confidence', 0.5),
            context=f"Tier {tier} LLM inverse reasoning",
            metadata={'tier': tier}
        )

        reasoning_step = ReasoningStep(
            step_id="llm_synthesis_inverse",
            description=f"Synthesized synthesis conditions using Tier {tier} reasoning",
            evidence_sources=[llm_source],
            reasoning_type="synthesis",
            confidence=llm_result.get('confidence', 0.5),
            intermediate_conclusion=llm_result.get('reasoning', '')[:200] + "..."
        )
        reasoning_steps.append(reasoning_step)

        # Build final chain of thought
        cot = ChainOfThought(
            query_context={'type': 'inverse', 'properties': desired_properties},
            reasoning_steps=reasoning_steps,
            final_reasoning=llm_result.get('reasoning', ''),
            final_result=llm_result.get('suggested_synthesis_conditions', {}),
            confidence_breakdown={
                step.step_id: step.confidence for step in reasoning_steps
            },
            source_attribution={
                'kg_sources': len(kg_sources),
                'literature_sources': len(lit_sources),
                'total_sources': len(all_sources)
            },
            tier=tier,
            kg_paths_used=len(paths),
            literature_papers_used=len(lit_sources)
        )

        return cot

    def _generate_search_queries(self, prompt_data: Dict, paths: List[str]) -> List[str]:
        """Generate search queries for forward prediction."""
        key_terms = [str(v) for v in prompt_data.values() if v][:5]
        queries = [
            f"experimental validation {' '.join(key_terms[:3])}",
            f"mechanism {' '.join(key_terms[:3])} materials science",
            f"quantitative data {' '.join(key_terms[:3])}",
        ]
        return queries

    def _generate_search_queries_inverse(self, prompt_data: Dict, paths: List[str]) -> List[str]:
        """Generate search queries for inverse design."""
        key_terms = [str(v) for v in prompt_data.values() if v][:5]
        queries = [
            f"synthesis methods {' '.join(key_terms[:3])}",
            f"achieving {' '.join(key_terms[:3])} materials",
            f"experimental protocols {' '.join(key_terms[:3])}",
        ]
        return queries

    def _search_literature(self, queries: List[str], max_papers: int = 8) -> List[Dict]:
        """Execute literature searches and return papers."""
        all_papers = []

        for query in queries[:3]:  # Limit queries
            papers = self.searcher.search(query, max_results=3, use_both=False)
            all_papers.extend(papers)

        # Deduplicate
        unique_papers = {}
        for paper in all_papers:
            title = paper['title'].lower()
            if title not in unique_papers:
                unique_papers[title] = paper

        papers_list = list(unique_papers.values())[:max_papers]
        papers_list.sort(key=lambda x: x.get('citations', 0), reverse=True)

        return papers_list

    def _generate_prediction(self, synthesis_inputs: Dict, sources: List[KnowledgeSource],
                            paths: List[str], tier: int) -> Dict:
        """Generate forward prediction using LLM."""
        # Format evidence
        kg_evidence = "\n".join([f"- {s.content}" for s in sources if s.source_type.startswith('kg_')])[:500]
        lit_evidence = "\n".join([f"- {s.metadata.get('title', 'Unknown')}" for s in sources if s.source_type == 'literature'])[:500]

        prompt = f"""Predict material properties from synthesis conditions.

**Synthesis Conditions:**
{json.dumps(synthesis_inputs, indent=2)}

**Knowledge Graph Evidence:**
{kg_evidence if kg_evidence else 'No KG evidence'}

**Literature Evidence:**
{lit_evidence if lit_evidence else 'No literature found'}

Provide prediction in JSON format:
```json
{{
    "reasoning": "...",
    "predicted_properties": {{"doping_outcome": "...", "carrier_type": "..."}},
    "confidence": <float>
}}
```"""

        response_text = self.ollama.generate(prompt, temperature=0.0)

        try:
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = json.loads(response_text)
        except:
            result = {
                "reasoning": response_text,
                "predicted_properties": {},
                "confidence": 0.5
            }

        if not result.get('reasoning') and result.get('mechanistic_explanation'):
            result['reasoning'] = json.dumps(result['mechanistic_explanation'])

        return result

    def _generate_synthesis_conditions(self, desired_properties: Dict, sources: List[KnowledgeSource],
                                       paths: List[str], tier: int) -> Dict:
        """Generate inverse synthesis conditions using LLM."""
        kg_evidence = "\n".join([f"- {s.content}" for s in sources if s.source_type.startswith('kg_')])[:500]
        lit_evidence = "\n".join([f"- {s.metadata.get('title', 'Unknown')}" for s in sources if s.source_type == 'literature'])[:500]

        prompt = f"""Suggest synthesis conditions to achieve desired properties.

**Desired Properties:**
{json.dumps(desired_properties, indent=2)}

**Knowledge Graph Evidence:**
{kg_evidence if kg_evidence else 'No KG evidence'}

**Literature Evidence:**
{lit_evidence if lit_evidence else 'No literature found'}

Provide synthesis conditions in JSON format:
```json
{{
    "reasoning": "...",
    "suggested_synthesis_conditions": {{"host_material": "...", "dopant": {{"element": "..."}}, "method": "..."}},
    "confidence": <float>
}}
```"""

        response_text = self.ollama.generate(prompt, temperature=0.0)

        try:
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = json.loads(response_text)
        except:
            result = {
                "reasoning": response_text,
                "suggested_synthesis_conditions": {},
                "confidence": 0.5
            }

        return result

    def forward_prediction(self, synthesis_inputs: Dict) -> Dict:
        """
        Forward prediction with chain-of-thought transparency.

        Args:
            synthesis_inputs: Dict of synthesis parameters

        Returns:
            Dict with prediction + full chain-of-thought
        """
        cot = self._forward_with_cot(synthesis_inputs)

        # Convert to output format
        result = {
            'reasoning': cot.final_reasoning,
            'predicted_properties': cot.final_result,
            'confidence': cot.confidence_breakdown.get('llm_synthesis', 0.5),
            'tier': cot.tier,
            'reasoning_type': f'tier_{cot.tier}_cot',
            'kg_paths': cot.kg_paths_used,
            'literature_papers': cot.literature_papers_used,
            'chain_of_thought': cot.to_dict()  # Full transparency
        }

        return result

    def inverse_design(self, desired_properties: Dict) -> Dict:
        """
        Inverse design with chain-of-thought transparency.

        Args:
            desired_properties: Dict of target material properties

        Returns:
            Dict with synthesis conditions + full chain-of-thought
        """
        cot = self._inverse_with_cot(desired_properties)

        # Convert to output format
        result = {
            'reasoning': cot.final_reasoning,
            'suggested_synthesis_conditions': cot.final_result,
            'confidence': cot.confidence_breakdown.get('llm_synthesis_inverse', 0.5),
            'tier': cot.tier,
            'reasoning_type': f'tier_{cot.tier}_inverse_cot',
            'kg_paths': cot.kg_paths_used,
            'literature_papers': cot.literature_papers_used,
            'chain_of_thought': cot.to_dict()  # Full transparency
        }

        return result


if __name__ == '__main__':
    # Test ARIA Full
    kg_file = "../../data/KG/outputs/combined_doping_data.json"

    print("Initializing ARIA Full...")
    engine = ARIAFullOllama(kg_file=kg_file, model="deepseek-r1:8b")

    # Test forward prediction
    test_synthesis = {
        "method": "CVD",
        "host_material": "MoS2",
        "dopant": "Nb"
    }

    print(f"\n{'='*60}")
    print("Test: Forward Prediction with Chain-of-Thought")
    print('='*60)

    result = engine.forward_prediction(test_synthesis)

    print("\nResult:")
    print(f"Tier: {result.get('tier')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"KG Paths: {result.get('kg_paths')}")
    print(f"Literature Papers: {result.get('literature_papers')}")
    print(f"\nReasoning Steps: {len(result['chain_of_thought']['reasoning_steps'])}")
    for step in result['chain_of_thought']['reasoning_steps']:
        print(f"  - {step['step_id']}: {step['description']} (confidence={step['confidence']:.2f})")
