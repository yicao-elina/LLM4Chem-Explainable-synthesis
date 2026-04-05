"""
ARIA Search Implementation (Ollama-based)

This variant extends ARIA Core with online literature search validation:
- All ARIA Core features (3-tier reasoning)
- + Online literature search (OpenAlex, Semantic Scholar)
- + Citation extraction and grounding analysis
- + Validation of KG paths against published literature
- + Quantitative data extraction from papers

Tests the contribution of literature-based validation.

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
import requests
import time
from datetime import datetime
from collections import defaultdict

from ..ollama_client import get_ollama_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiteratureSearcher:
    """
    Handles literature search using OpenAlex and Semantic Scholar APIs.
    """

    def __init__(self, email: str = "research@example.com"):
        """
        Initialize literature searcher.

        Args:
            email: Email for OpenAlex API (polite pool access)
        """
        self.email = email
        self.openalex_base = "https://api.openalex.org/works"
        self.s2_base = "https://api.semanticscholar.org/graph/v1/paper/search"

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

    def _rate_limit(self):
        """Ensure we don't exceed API rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def search_openalex(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search OpenAlex for papers.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of paper metadata dicts
        """
        self._rate_limit()

        params = {
            'search': query,
            'per_page': max_results,
            'mailto': self.email,
            'sort': 'cited_by_count:desc'  # Most cited first
        }

        try:
            response = requests.get(self.openalex_base, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            papers = []
            for work in data.get('results', []):
                paper = {
                    'title': work.get('title', 'Untitled'),
                    'abstract': work.get('abstract', ''),
                    'url': work.get('id', ''),
                    'year': work.get('publication_year'),
                    'citations': work.get('cited_by_count', 0),
                    'authors': [author.get('author', {}).get('display_name', 'Unknown')
                               for author in work.get('authorships', [])[:3]],  # First 3 authors
                    'source': 'OpenAlex'
                }
                papers.append(paper)

            return papers

        except Exception as e:
            logger.warning(f"OpenAlex search failed for '{query}': {e}")
            return []

    def search_semantic_scholar(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search Semantic Scholar for papers.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of paper metadata dicts
        """
        self._rate_limit()

        params = {
            'query': query,
            'limit': max_results,
            'fields': 'title,abstract,url,year,citationCount,authors'
        }

        try:
            response = requests.get(self.s2_base, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            papers = []
            for paper in data.get('data', []):
                paper_dict = {
                    'title': paper.get('title', 'Untitled'),
                    'abstract': paper.get('abstract', ''),
                    'url': paper.get('url', ''),
                    'year': paper.get('year'),
                    'citations': paper.get('citationCount', 0),
                    'authors': [author.get('name', 'Unknown')
                               for author in paper.get('authors', [])[:3]],  # First 3 authors
                    'source': 'Semantic Scholar'
                }
                papers.append(paper_dict)

            return papers

        except Exception as e:
            logger.warning(f"Semantic Scholar search failed for '{query}': {e}")
            return []

    def search(self, query: str, max_results: int = 10, use_both: bool = True) -> List[Dict]:
        """
        Search both APIs and combine results.

        Args:
            query: Search query
            max_results: Maximum total results
            use_both: If True, search both APIs and combine

        Returns:
            Combined and deduplicated results
        """
        results = []

        if use_both:
            # Search both APIs
            openalex_results = self.search_openalex(query, max_results // 2)
            s2_results = self.search_semantic_scholar(query, max_results // 2)
            results = openalex_results + s2_results
        else:
            # Just use OpenAlex (more comprehensive)
            results = self.search_openalex(query, max_results)

        # Deduplicate by title similarity
        unique_results = []
        seen_titles = set()

        for paper in results:
            title_lower = paper['title'].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_results.append(paper)

        # Sort by citations (most cited first)
        unique_results.sort(key=lambda x: x.get('citations', 0), reverse=True)

        return unique_results[:max_results]


class ARIASearchOllama:
    """
    ARIA Search: ARIA Core + literature search validation.

    Extends 3-tier reasoning with online literature search to:
    - Validate KG paths against published research
    - Extract quantitative data from papers
    - Detect contradictions in literature
    - Ground predictions in citable sources
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
        Initialize ARIA Search system.

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

        logger.info(f"ARIA Search initialized: {self.kg.number_of_nodes()} nodes, "
                   f"{self.kg.number_of_edges()} edges, 3-tier reasoning + literature search enabled")

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

    def _extract_mechanisms(self, path_str: str) -> List[str]:
        """Extract mechanism information from path."""
        nodes = [n.strip() for n in path_str.split(" -> ")]
        mechanisms = []

        for i in range(len(nodes) - 1):
            if self.kg.has_edge(nodes[i], nodes[i+1]):
                mech = self.kg[nodes[i]][nodes[i+1]].get('mechanism', '')
                if mech:
                    mechanisms.append(f"{nodes[i]} → {nodes[i+1]}: {mech}")

        return mechanisms

    def _generate_search_queries(self, prompt_data: Dict, paths: List[str],
                                query_type: str) -> List[str]:
        """
        Generate targeted search queries for literature validation.

        Args:
            prompt_data: Original query data (synthesis or properties)
            paths: Causal paths found in KG
            query_type: 'forward' or 'inverse'

        Returns:
            List of search query strings
        """
        queries = []

        # Extract key terms from prompt
        key_terms = []
        for value in prompt_data.values():
            if value:
                terms = str(value).replace('_', ' ').split()
                key_terms.extend(terms)

        # Limit to most important terms
        key_terms = key_terms[:5]

        # 1. Direct validation queries for paths
        for path in paths[:3]:  # Top 3 paths
            path_terms = path.replace(" -> ", " ").replace("_", " ")
            queries.append(f"experimental validation {path_terms}")
            queries.append(f"mechanism {path_terms} materials science")

        # 2. Quantitative data queries
        queries.extend([
            f"quantitative data {' '.join(key_terms)} experimental",
            f"numerical results {' '.join(key_terms)} measurements",
        ])

        # 3. Recent research (temporal awareness)
        current_year = datetime.now().year
        queries.extend([
            f"recent advances {' '.join(key_terms)} {current_year}",
            f"latest research {' '.join(key_terms)} {current_year-1}-{current_year}",
        ])

        # 4. Alternative mechanisms
        queries.extend([
            f"alternative mechanisms {' '.join(key_terms)}",
            f"competing theories {' '.join(key_terms)} doping",
        ])

        return queries[:12]  # Limit total queries

    def _search_and_summarize(self, queries: List[str], max_papers: int = 10) -> Dict:
        """
        Execute searches and summarize findings.

        Args:
            queries: List of search queries
            max_papers: Maximum papers to retrieve

        Returns:
            Dict with search results and summary
        """
        all_papers = []
        query_results = {}

        logger.info(f"Executing {len(queries)} literature searches...")

        for query in queries[:5]:  # Limit to top 5 queries to avoid rate limits
            papers = self.searcher.search(query, max_results=3, use_both=False)
            query_results[query] = papers
            all_papers.extend(papers)

        # Deduplicate by title
        unique_papers = {}
        for paper in all_papers:
            title = paper['title'].lower()
            if title not in unique_papers:
                unique_papers[title] = paper

        papers_list = list(unique_papers.values())[:max_papers]

        # Sort by citations
        papers_list.sort(key=lambda x: x.get('citations', 0), reverse=True)

        logger.info(f"Found {len(papers_list)} unique papers from literature search")

        return {
            'total_papers': len(papers_list),
            'papers': papers_list,
            'queries_used': list(query_results.keys()),
            'papers_per_query': {q: len(p) for q, p in query_results.items()}
        }

    def _format_literature_context(self, search_results: Dict) -> str:
        """Format search results for inclusion in LLM prompt."""
        if not search_results['papers']:
            return "No relevant literature found."

        context = f"**Literature Search Results ({search_results['total_papers']} papers found):**\n\n"

        for i, paper in enumerate(search_results['papers'][:10], 1):
            authors = ', '.join(paper.get('authors', ['Unknown']))
            year = paper.get('year', 'N/A')
            title = paper.get('title', 'Untitled')
            abstract = paper.get('abstract', 'No abstract available.')
            citations = paper.get('citations', 0)

            # Truncate abstract
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."

            context += f"{i}. **{title}** ({year})\n"
            context += f"   Authors: {authors}\n"
            context += f"   Citations: {citations}\n"
            context += f"   Abstract: {abstract}\n"
            context += f"   Source: {paper.get('source', 'Unknown')}\n\n"

        return context

    def _tier1_forward_direct(self, synthesis_inputs: Dict, paths: List[str]) -> Dict:
        """Tier 1: Direct path reasoning with literature validation."""
        logger.info(f"Tier 1 (Direct): Found {len(paths)} paths, conducting literature search...")

        # Extract mechanisms
        all_mechanisms = []
        for path in paths:
            mechanisms = self._extract_mechanisms(path)
            all_mechanisms.extend(mechanisms)

        # Generate and execute search queries
        search_queries = self._generate_search_queries(synthesis_inputs, paths, 'forward')
        search_results = self._search_and_summarize(search_queries, max_papers=10)
        literature_context = self._format_literature_context(search_results)

        # Build prompt with KG + literature
        prompt = f"""You are an expert materials scientist. Predict the material properties resulting from the given synthesis conditions.

**Synthesis Conditions:**
{json.dumps(synthesis_inputs, indent=2)}

**Knowledge Graph Evidence:**
Causal Pathways Found:
{chr(10).join(f"- {path}" for path in paths[:5])}

Known Mechanisms:
{chr(10).join(f"- {mech}" for mech in all_mechanisms[:5]) if all_mechanisms else "- No mechanisms provided"}

{literature_context}

**Instructions:**
1. Analyze the causal pathways and mechanisms from the knowledge graph
2. Validate these pathways using the literature search results above
3. Identify any contradictions or supporting evidence from papers
4. Synthesize a comprehensive prediction combining KG evidence and literature
5. Provide confidence based on literature support

**Output Format:**
```json
{{
    "reasoning": "Step-by-step analysis incorporating KG and literature evidence...",
    "predicted_properties": {{
        "doping_outcome": "...",
        "carrier_type": "...",
        "conductivity_change": "...",
        "other_properties": "..."
    }},
    "confidence": <float between 0 and 1>,
    "literature_support": "Brief summary of key supporting papers",
    "contradictions": "Any conflicting evidence found (or 'none')"
}}
```

Respond with valid JSON only."""

        response_text = self.ollama.generate(prompt, temperature=0.0)

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = {
                    "reasoning": response_text,
                    "predicted_properties": {},
                    "confidence": 0.5
                }

        result['tier'] = 1
        result['reasoning_type'] = 'direct_path_with_search'
        result['kg_paths'] = len(paths)
        result['literature_papers'] = search_results['total_papers']
        result['search_queries'] = search_results['queries_used']

        if not result.get('reasoning') and result.get('mechanistic_explanation'):
            result['reasoning'] = json.dumps(result['mechanistic_explanation'])

        return result

    def _tier1_inverse_direct(self, desired_properties: Dict, paths: List[str]) -> Dict:
        """Tier 1: Direct inverse reasoning with literature validation."""
        logger.info(f"Tier 1 (Inverse Direct): Found {len(paths)} paths, conducting literature search...")

        # Extract mechanisms
        all_mechanisms = []
        for path in paths:
            mechanisms = self._extract_mechanisms(path)
            all_mechanisms.extend(mechanisms)

        # Generate and execute search queries
        search_queries = self._generate_search_queries(desired_properties, paths, 'inverse')
        search_results = self._search_and_summarize(search_queries, max_papers=10)
        literature_context = self._format_literature_context(search_results)

        prompt = f"""You are an expert materials scientist. Suggest synthesis conditions to achieve the desired material properties.

**Desired Properties:**
{json.dumps(desired_properties, indent=2)}

**Knowledge Graph Evidence:**
Causal Pathways Found (Inverse):
{chr(10).join(f"- {path}" for path in paths[:5])}

Known Mechanisms:
{chr(10).join(f"- {mech}" for mech in all_mechanisms[:5]) if all_mechanisms else "- No mechanisms provided"}

{literature_context}

**Instructions:**
1. Analyze the reverse causal pathways from the knowledge graph
2. Validate these pathways using the literature search results above
3. Identify experimental protocols from the papers that achieved similar properties
4. Synthesize a comprehensive synthesis strategy combining KG and literature
5. Provide confidence based on literature support

**Output Format:**
```json
{{
    "reasoning": "Step-by-step analysis incorporating KG and literature evidence...",
    "suggested_synthesis_conditions": {{
        "host_material": "...",
        "dopant": {{
            "element": "...",
            "concentration": "..."
        }},
        "method": "...",
        "temperature_c": <number or null>,
        "other_parameters": {{}}
    }},
    "confidence": <float between 0 and 1>,
    "literature_support": "Brief summary of key supporting papers",
    "experimental_precedents": "Papers that achieved similar outcomes"
}}
```

Respond with valid JSON only."""

        response_text = self.ollama.generate(prompt, temperature=0.0)

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = {
                    "reasoning": response_text,
                    "suggested_synthesis_conditions": {},
                    "confidence": 0.5
                }

        result['tier'] = 1
        result['reasoning_type'] = 'direct_inverse_with_search'
        result['kg_paths'] = len(paths)
        result['literature_papers'] = search_results['total_papers']
        result['search_queries'] = search_results['queries_used']

        return result

    def _tier2_forward_transfer(self, synthesis_inputs: Dict, similar_node: str,
                               score: float, analogous_paths: List[str]) -> Dict:
        """Tier 2: Transfer learning with literature validation."""
        logger.info(f"Tier 2 (Transfer): Using analogous paths (similarity={score:.2f}), conducting literature search...")

        # Extract mechanisms from analogous paths
        all_mechanisms = []
        for path in analogous_paths:
            mechanisms = self._extract_mechanisms(path)
            all_mechanisms.extend(mechanisms)

        # Generate search queries for transfer learning
        transfer_queries = [
            f"similar mechanisms {json.dumps(synthesis_inputs)} materials science",
            f"analogous processes {analogous_paths[0].replace(' -> ', ' ')}",
            f"transfer learning materials synthesis",
        ]

        search_results = self._search_and_summarize(transfer_queries, max_papers=8)
        literature_context = self._format_literature_context(search_results)

        prompt = f"""You are an expert materials scientist conducting transfer learning analysis. Predict properties by reasoning from analogous cases.

**Target Synthesis Conditions:**
{json.dumps(synthesis_inputs, indent=2)}

**Most Similar Known Case:** {similar_node} (similarity: {score:.2f})

**Analogous Causal Pathways:**
{chr(10).join(f"- {path}" for path in analogous_paths[:3])}

**Known Mechanisms:**
{chr(10).join(f"- {mech}" for mech in all_mechanisms[:5]) if all_mechanisms else "- No mechanisms provided"}

{literature_context}

**Instructions:**
1. Analyze the analogous pathways from similar cases
2. Use literature to validate transfer learning assumptions
3. Identify key similarities and differences
4. Adapt the analogous knowledge to the target case
5. Provide conservative confidence reflecting transfer uncertainty

**Output Format:**
```json
{{
    "reasoning": "Transfer learning analysis with literature validation...",
    "predicted_properties": {{
        "doping_outcome": "...",
        "carrier_type": "...",
        "other_properties": "..."
    }},
    "confidence": <float between 0 and 1>,
    "transfer_validity": "Assessment of transfer learning applicability",
    "literature_support": "Papers supporting the transfer"
}}
```

Respond with valid JSON only."""

        response_text = self.ollama.generate(prompt, temperature=0.0)

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = {
                    "reasoning": response_text,
                    "predicted_properties": {},
                    "confidence": 0.4
                }

        result['tier'] = 2
        result['reasoning_type'] = 'transfer_with_search'
        result['similarity_score'] = score
        result['similar_node'] = similar_node
        result['kg_paths'] = len(analogous_paths)
        result['literature_papers'] = search_results['total_papers']

        if not result.get('reasoning') and result.get('mechanistic_explanation'):
            result['reasoning'] = json.dumps(result['mechanistic_explanation'])

        return result

    def _tier2_inverse_transfer(self, desired_properties: Dict, similar_node: str,
                               score: float, analogous_paths: List[str]) -> Dict:
        """Tier 2: Inverse transfer learning with literature validation."""
        logger.info(f"Tier 2 (Inverse Transfer): Using analogous paths (similarity={score:.2f}), conducting literature search...")

        all_mechanisms = []
        for path in analogous_paths:
            mechanisms = self._extract_mechanisms(path)
            all_mechanisms.extend(mechanisms)

        transfer_queries = [
            f"similar properties {json.dumps(desired_properties)} materials science",
            f"analogous outcomes {analogous_paths[0].replace(' -> ', ' ')}",
        ]

        search_results = self._search_and_summarize(transfer_queries, max_papers=8)
        literature_context = self._format_literature_context(search_results)

        prompt = f"""You are an expert materials scientist conducting inverse transfer learning. Suggest synthesis conditions based on analogous cases.

**Target Properties:**
{json.dumps(desired_properties, indent=2)}

**Most Similar Known Property:** {similar_node} (similarity: {score:.2f})

**Analogous Causal Pathways (Inverse):**
{chr(10).join(f"- {path}" for path in analogous_paths[:3])}

**Known Mechanisms:**
{chr(10).join(f"- {mech}" for mech in all_mechanisms[:5]) if all_mechanisms else "- No mechanisms provided"}

{literature_context}

**Instructions:**
1. Analyze analogous inverse pathways from similar property targets
2. Use literature to validate and adapt the approach
3. Identify synthesis protocols that achieved similar outcomes
4. Adapt conditions for the target properties
5. Provide conservative confidence

**Output Format:**
```json
{{
    "reasoning": "Inverse transfer learning with literature validation...",
    "suggested_synthesis_conditions": {{
        "host_material": "...",
        "dopant": {{"element": "...", "concentration": "..."}},
        "method": "...",
        "temperature_c": <number or null>,
        "other_parameters": {{}}
    }},
    "confidence": <float between 0 and 1>,
    "transfer_validity": "Assessment of inverse transfer applicability",
    "literature_support": "Papers with similar property targets"
}}
```

Respond with valid JSON only."""

        response_text = self.ollama.generate(prompt, temperature=0.0)

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = {
                    "reasoning": response_text,
                    "suggested_synthesis_conditions": {},
                    "confidence": 0.4
                }

        result['tier'] = 2
        result['reasoning_type'] = 'transfer_inverse_with_search'
        result['similarity_score'] = score
        result['similar_node'] = similar_node
        result['kg_paths'] = len(analogous_paths)
        result['literature_papers'] = search_results['total_papers']

        return result

    def _tier3_forward_fallback(self, synthesis_inputs: Dict) -> Dict:
        """Tier 3: Baseline fallback with literature search."""
        logger.info("Tier 3 (Fallback): No KG match, using baseline reasoning with literature search...")

        # Generate generic search queries
        key_terms = [str(v) for v in synthesis_inputs.values() if v]
        fallback_queries = [
            f"{' '.join(key_terms[:3])} materials synthesis",
            f"{' '.join(key_terms[:3])} doping properties",
        ]

        search_results = self._search_and_summarize(fallback_queries, max_papers=5)
        literature_context = self._format_literature_context(search_results)

        prompt = f"""You are an expert materials scientist. Predict the material properties from synthesis conditions using your baseline knowledge and literature.

**Synthesis Conditions:**
{json.dumps(synthesis_inputs, indent=2)}

{literature_context}

**Instructions:**
Predict the resulting properties based on fundamental materials science principles and the literature above.

**Output Format:**
```json
{{
    "reasoning": "Analysis based on fundamental principles and literature...",
    "predicted_properties": {{
        "doping_outcome": "...",
        "carrier_type": "...",
        "other_properties": "..."
    }},
    "confidence": <float between 0 and 1>,
    "literature_support": "Key papers used"
}}
```

Respond with valid JSON only."""

        response_text = self.ollama.generate(prompt, temperature=0.0)

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = {
                    "reasoning": response_text,
                    "predicted_properties": {},
                    "confidence": 0.3
                }

        result['tier'] = 3
        result['reasoning_type'] = 'baseline_fallback_with_search'
        result['kg_paths'] = 0
        result['literature_papers'] = search_results['total_papers']

        if not result.get('reasoning') and result.get('mechanistic_explanation'):
            result['reasoning'] = json.dumps(result['mechanistic_explanation'])

        return result

    def _tier3_inverse_fallback(self, desired_properties: Dict) -> Dict:
        """Tier 3: Inverse baseline fallback with literature search."""
        logger.info("Tier 3 (Inverse Fallback): No KG match, using baseline reasoning with literature search...")

        key_terms = [str(v) for v in desired_properties.values() if v]
        fallback_queries = [
            f"{' '.join(key_terms[:3])} synthesis methods",
            f"achieving {' '.join(key_terms[:3])} materials",
        ]

        search_results = self._search_and_summarize(fallback_queries, max_papers=5)
        literature_context = self._format_literature_context(search_results)

        prompt = f"""You are an expert materials scientist. Suggest synthesis conditions to achieve the desired properties using your baseline knowledge and literature.

**Desired Properties:**
{json.dumps(desired_properties, indent=2)}

{literature_context}

**Instructions:**
Suggest synthesis conditions based on fundamental materials science principles and the literature above.

**Output Format:**
```json
{{
    "reasoning": "Analysis based on fundamental principles and literature...",
    "suggested_synthesis_conditions": {{
        "host_material": "...",
        "dopant": {{"element": "...", "concentration": "..."}},
        "method": "...",
        "temperature_c": <number or null>,
        "other_parameters": {{}}
    }},
    "confidence": <float between 0 and 1>,
    "literature_support": "Key papers used"
}}
```

Respond with valid JSON only."""

        response_text = self.ollama.generate(prompt, temperature=0.0)

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = {
                    "reasoning": response_text,
                    "suggested_synthesis_conditions": {},
                    "confidence": 0.3
                }

        result['tier'] = 3
        result['reasoning_type'] = 'baseline_inverse_fallback_with_search'
        result['kg_paths'] = 0
        result['literature_papers'] = search_results['total_papers']

        return result

    def forward_prediction(self, synthesis_inputs: Dict) -> Dict:
        """
        Forward prediction with 3-tier reasoning + literature search.

        Args:
            synthesis_inputs: Dict of synthesis parameters

        Returns:
            Prediction result with tier and search metadata
        """
        logger.info("\n=== ARIA Search: Forward Prediction ===")

        # Extract keywords
        input_keywords = [str(v) for v in synthesis_inputs.values() if v]

        # Get all property nodes (sink nodes)
        property_nodes = [n for n in self.kg.nodes() if self.kg.out_degree(n) == 0]

        # Try Tier 1: Direct path matching
        paths = self._find_paths(input_keywords, property_nodes)

        if paths:
            return self._tier1_forward_direct(synthesis_inputs, paths)

        # Try Tier 2: Transfer learning
        synthesis_nodes = [n for n in self.kg.nodes() if self.kg.in_degree(n) == 0]
        query_string = " ".join(input_keywords)
        similar_node, score = self._find_most_similar_node(query_string, synthesis_nodes)

        if similar_node and score > self.similarity_threshold:
            analogous_paths = self._find_paths([similar_node], property_nodes)
            if analogous_paths:
                return self._tier2_forward_transfer(
                    synthesis_inputs, similar_node, score, analogous_paths
                )

        # Tier 3: Fallback
        return self._tier3_forward_fallback(synthesis_inputs)

    def inverse_design(self, desired_properties: Dict) -> Dict:
        """
        Inverse design with 3-tier reasoning + literature search.

        Args:
            desired_properties: Dict of target material properties

        Returns:
            Synthesis suggestion with tier and search metadata
        """
        logger.info("\n=== ARIA Search: Inverse Design ===")

        # Extract keywords
        property_keywords = [str(v) for v in desired_properties.values() if v]

        # Get all synthesis nodes (source nodes)
        synthesis_nodes = [n for n in self.kg.nodes() if self.kg.in_degree(n) == 0]

        # Try Tier 1: Direct inverse path matching
        paths = self._find_paths(synthesis_nodes, property_keywords, reverse=True)

        if paths:
            return self._tier1_inverse_direct(desired_properties, paths)

        # Try Tier 2: Inverse transfer learning
        property_nodes = [n for n in self.kg.nodes() if self.kg.out_degree(n) == 0]
        query_string = " ".join(property_keywords)
        similar_node, score = self._find_most_similar_node(query_string, property_nodes)

        if similar_node and score > self.similarity_threshold:
            analogous_paths = self._find_paths(synthesis_nodes, [similar_node], reverse=True)
            if analogous_paths:
                return self._tier2_inverse_transfer(
                    desired_properties, similar_node, score, analogous_paths
                )

        # Tier 3: Inverse fallback
        return self._tier3_inverse_fallback(desired_properties)


if __name__ == '__main__':
    # Test ARIA Search
    kg_file = "../../data/KG/outputs/combined_doping_data.json"

    print("Initializing ARIA Search...")
    engine = ARIASearchOllama(kg_file=kg_file, model="deepseek-r1:8b")

    # Test cases
    test_cases = [
        {
            "name": "Forward: CVD MoS2 Nb Doping",
            "synthesis": {
                "method": "CVD",
                "host_material": "MoS2",
                "dopant": "Nb"
            }
        },
        {
            "name": "Inverse: N-type High Mobility",
            "properties": {
                "carrier_type": "n-type",
                "mobility": "high"
            }
        }
    ]

    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test_case['name']}")
        print('='*60)

        if 'synthesis' in test_case:
            result = engine.forward_prediction(test_case['synthesis'])
            print("\nForward Prediction Result:")
        else:
            result = engine.inverse_design(test_case['properties'])
            print("\nInverse Design Result:")

        print(json.dumps(result, indent=2))
        print(f"\nTier: {result.get('tier')}")
        print(f"Reasoning Type: {result.get('reasoning_type')}")
        print(f"KG Paths: {result.get('kg_paths', 0)}")
        print(f"Literature Papers: {result.get('literature_papers', 0)}")
        print(f"Confidence: {result.get('confidence', 0)}")
