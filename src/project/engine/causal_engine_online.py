import json
import networkx as nx
import os
import google.generativeai as genai
from google.generativeai.types import Tool
from pathlib import Path
import textwrap
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import re

# --- Helper function to extract JSON from model response ---
def extract_json_from_response(text: str):
    """
    Extracts a JSON object from a string that contains a JSON markdown block.
    This version is more robust against malformed JSON from the LLM.
    """
    import re
    # Match the JSON block
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)
    
    if not match:
        # If no JSON block is found at all, return a warning and the raw response.
        return {"warning": "No JSON object found in the response.", "raw_response": text}
    
    json_string = match.group(1)
    
    try:
        # Attempt to parse the extracted string as JSON
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # If parsing fails, return a detailed error message including the problematic text.
        print("\n--- JSON DECODE ERROR ---")
        print(f"Failed to parse JSON from the model's response. Error: {e}")
        print("Problematic text received from model:")
        print(json_string)
        print("-------------------------\n")
        return {
            "error": "Failed to decode JSON from model response.",
            "details": str(e),
            "malformed_json_string": json_string
        }


class CausalReasoningEngine:
    """
    Enhanced version with improved search capabilities, citation granularity,
    contradiction detection, quantitative validation, and temporal awareness.
    """
    def __init__(self, json_file_path: str, model_id: str = "gemini-1.5-pro-latest", embedding_model: str = 'all-MiniLM-L6-v2'):
        self.graph = self._build_graph(json_file_path)
        self._configure_api(model_id)
        print(f"Loading sentence transformer model ('{embedding_model}')...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self._precompute_node_embeddings()
        print("Enhanced Causal Reasoning Engine initialized.")
        print(f"Knowledge graph contains {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def _configure_api(self, model_id: str):
        api_key = os.getenv("GOOGLE_API_KEY")
        # Enhanced grounding tool configuration for better search coverage
        grounding_tool = Tool(google_search_retrieval={
            'dynamic_retrieval_config': {
                'mode': 'MODE_DYNAMIC',
                'dynamic_threshold': 0.7
            }
        })

        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY")
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(model_id, tools=[grounding_tool])

    def _build_graph(self, json_file_path: str):
        input_path = Path(json_file_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found at {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both formats
        if "causal_relationships" in data:
            relationships = data.get("causal_relationships", [])
        else: # Assume the root is the list of relationships
            relationships = data

        G = nx.DiGraph()
        for rel in relationships:
            cause_text = rel.get("cause_parameter") or "Unknown Cause"
            effect_text = rel.get("effect_on_doping") or "Unknown Effect"
            affected_property = rel.get("affected_property")
            cause = textwrap.fill(cause_text.strip(), width=25)
            if affected_property and affected_property.strip():
                effect_label = f"{effect_text.strip()}\n({affected_property.strip()})"
            else:
                effect_label = effect_text.strip()
            effect = textwrap.fill(effect_label, width=25)
            if 'unknown' not in cause.lower() and 'n/a' not in cause.lower() and \
               'unknown' not in effect.lower() and 'n/a' not in effect.lower():
                mechanism = rel.get("mechanism_quote", "")
                G.add_edge(cause, effect, mechanism=mechanism)
        return G

    def _precompute_node_embeddings(self):
        print("Pre-computing node embeddings...")
        self.node_list = list(self.graph.nodes())
        if not self.node_list:
            self.node_embeddings = np.array([])
            return
        self.node_embeddings = self.embedding_model.encode(self.node_list, convert_to_tensor=False)
        print("Embeddings computed.")

    def _find_most_similar_node(self, query: str, candidate_nodes: list):
        if not candidate_nodes or self.node_embeddings.size == 0:
            return None, 0.0
        query_embedding = self.embedding_model.encode([query])
        candidate_indices = [self.node_list.index(node) for node in candidate_nodes if node in self.node_list]
        if not candidate_indices:
            return None, 0.0
        candidate_embeddings = self.node_embeddings[candidate_indices]
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        best_match_local_index = np.argmax(similarities)
        best_score = similarities[best_match_local_index]
        best_node = candidate_nodes[best_match_local_index]
        return best_node, best_score

    def _calculate_embedding_difference(self, text1: str, text2: str):
        """
        Calculates the cosine distance (1 - cosine similarity) between two text embeddings.
        Returns a value between 0 (identical) and 2 (completely opposite).
        """
        if not text1 or not text2: return 1.0
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        distance = 1 - similarity
        return distance

    def _find_relevant_paths(self, start_keywords: list, end_keywords: list, reverse: bool = False):
        start_keywords_str = [str(kw).lower() for kw in start_keywords if kw is not None]
        end_keywords_str = [str(kw).lower() for kw in end_keywords if kw is not None]
        start_nodes = {n for n in self.graph.nodes if any(kw in n.lower() for kw in start_keywords_str)}
        end_nodes = {n for n in self.graph.nodes if any(kw in n.lower() for kw in end_keywords_str)}
        valid_paths = []
        graph_to_search = self.graph.reverse(copy=True) if reverse else self.graph
        source_nodes, target_nodes = (end_nodes, start_nodes) if reverse else (start_nodes, end_nodes)
        for source in source_nodes:
            for target in target_nodes:
                if nx.has_path(graph_to_search, source, target):
                    for path in nx.all_simple_paths(graph_to_search, source=source, target=target):
                        valid_paths.append(" -> ".join(path))
        return list(set(valid_paths))

    def _get_path_mechanisms(self, path_str: str):
        """
        Extract mechanism information for edges in a path.
        """
        nodes = [node.strip() for node in path_str.split(" -> ")]
        mechanisms = []
        for i in range(len(nodes) - 1):
            if self.graph.has_edge(nodes[i], nodes[i+1]):
                mechanism = self.graph[nodes[i]][nodes[i+1]].get('mechanism', '')
                if mechanism:
                    mechanisms.append(f"{nodes[i]} → {nodes[i+1]}: {mechanism}")
        return mechanisms

    def _generate_comprehensive_search_queries(self, original_prompt_data: dict, causal_paths: list, query_type: str) -> List[str]:
        """
        Generate multiple targeted search queries for comprehensive validation.
        """
        queries = []
        
        # Extract key terms from the prompt and paths
        key_terms = []
        for value in original_prompt_data.values():
            if value:
                key_terms.extend(str(value).split())
        
        path_terms = []
        for path in causal_paths:
            path_terms.extend(path.replace(" -> ", " ").split())
        
        # 1. Validation queries
        for path in causal_paths[:3]:  # Limit to top 3 paths
            queries.append(f"experimental validation {path.replace(' -> ', ' ')}")
            queries.append(f"mechanism {path.replace(' -> ', ' ')} materials science")
        
        # 2. Contradiction detection queries
        queries.extend([
            f"contradictory results {' '.join(key_terms[:5])}",
            f"alternative mechanisms {' '.join(key_terms[:5])}",
            f"limitations {' '.join(key_terms[:5])} materials science"
        ])
        
        # 3. Quantitative data queries
        queries.extend([
            f"quantitative data {' '.join(key_terms[:5])} experimental values",
            f"numerical results {' '.join(key_terms[:5])} measurements",
            f"experimental parameters {' '.join(key_terms[:5])}"
        ])
        
        # 4. Recent research queries (temporal awareness)
        current_year = datetime.now().year
        queries.extend([
            f"recent advances {' '.join(key_terms[:5])} {current_year}",
            f"latest research {' '.join(key_terms[:5])} {current_year-1}-{current_year}",
            f"2023-2024 {' '.join(key_terms[:5])} breakthrough"
        ])
        
        return queries[:15]  # Limit total queries to avoid overwhelming

    def _extract_detailed_citations(self, response, sources: List[Dict]) -> Dict:
        """
        Enhanced citation extraction with better granularity.
        """
        citations = {
            "total_sources": len(sources),
            "source_details": sources,
            "citation_mapping": [],
            "coverage_analysis": {
                "well_cited_claims": [],
                "poorly_cited_claims": [],
                "uncited_claims": []
            }
        }
        
        if not (response.candidates and response.candidates[0].grounding_metadata):
            return citations
            
        metadata = response.candidates[0].grounding_metadata
        
        if hasattr(metadata, 'grounding_supports') and sources:
            for i, support in enumerate(metadata.grounding_supports):
                segment_text = support.segment.text
                confidence = getattr(support, 'confidence_scores', [0.0])[0] if hasattr(support, 'confidence_scores') else 0.0
                
                supporting_sources = []
                for chunk_index in support.grounding_chunk_indices:
                    if chunk_index < len(sources):
                        supporting_sources.append({
                            "index": chunk_index,
                            "title": sources[chunk_index]["title"],
                            "uri": sources[chunk_index]["uri"]
                        })
                
                citation_entry = {
                    "claim": segment_text,
                    "confidence": confidence,
                    "sources": supporting_sources,
                    "citation_quality": "high" if len(supporting_sources) >= 2 else "medium" if len(supporting_sources) == 1 else "low"
                }
                
                citations["citation_mapping"].append(citation_entry)
                
                # Categorize claims by citation quality
                if len(supporting_sources) >= 2:
                    citations["coverage_analysis"]["well_cited_claims"].append(segment_text)
                elif len(supporting_sources) == 1:
                    citations["coverage_analysis"]["poorly_cited_claims"].append(segment_text)
                else:
                    citations["coverage_analysis"]["uncited_claims"].append(segment_text)
        
        return citations

    def _enhanced_direct_path_query(self, original_prompt_data: dict, causal_paths: list, query_type: str):
        """
        Significantly enhanced query with all five improvements implemented.
        """
        print(f"Found {len(causal_paths)} direct causal paths. Conducting comprehensive validation...")

        # Generate comprehensive search queries
        search_queries = self._generate_comprehensive_search_queries(original_prompt_data, causal_paths, query_type)
        
        # Mechanism extraction
        all_mechanisms = []
        for path in causal_paths:
            mechanisms = self._get_path_mechanisms(path)
            if mechanisms:
                all_mechanisms.extend(mechanisms)

        if query_type == "forward":
            task_description = "predict the resulting material properties from the given synthesis conditions"
            input_data_label = "Synthesis Conditions"
            output_format_label = "predicted_properties"
            example_output = '{"carrier_type": "p-type", "band_gap_ev": 1.5, "carrier_concentration": "increased"}'
        else: # inverse
            task_description = "propose synthesis conditions to achieve the desired material properties"
            input_data_label = "Desired Material Properties"
            output_format_label = "suggested_synthesis_conditions"
            example_output = '{"method": "Surface oxidation", "temperature_c": 200, "duration_hours": 2}'

        formatted_paths = "\n- ".join(causal_paths)
        formatted_mechanisms = "\n- ".join(all_mechanisms) if all_mechanisms else "No specific mechanisms provided in knowledge graph"
        formatted_search_queries = "\n- ".join(search_queries)

        # Enhanced prompt with all five improvements
        prompt = f"""
        You are a world-class materials science research AI conducting a comprehensive peer review. Your task is to validate, critique, and enhance a hypothesis from a knowledge graph using systematic literature search and analysis.

        **CRITICAL INSTRUCTIONS - YOU MUST FOLLOW ALL OF THESE:**

        1. **MANDATORY SEARCH USAGE**: You MUST actively use your search tool for EVERY major claim you make. Do not rely solely on your training data.

        2. **MULTIPLE SOURCE VALIDATION**: For each key finding, search for and cite AT LEAST 2-3 independent sources when possible.

        3. **CONTRADICTION DETECTION**: Actively search for studies that contradict or challenge the knowledge graph's claims.

        4. **QUANTITATIVE FOCUS**: Prioritize finding numerical data, experimental values, and measurement ranges.

        5. **TEMPORAL AWARENESS**: Emphasize recent studies (2022-2024) and note if older findings have been superseded.

        **User's Goal:**
        The user wants to {task_description}.

        **Context from User's Query:**
        - **{input_data_label}:** {json.dumps(original_prompt_data, indent=2)}

        **Hypothesis from Knowledge Graph:**
        The knowledge graph suggests the following causal pathway(s) and mechanism(s):
        - **Path(s):** {formatted_paths}
        - **Known Mechanism(s):** {formatted_mechanisms}

        **Suggested Search Queries (use these as starting points):**
        - {formatted_search_queries}

        **Your Systematic Analysis Process:**

        1. **VALIDATION WITH MULTIPLE SOURCES**: 
           - Search for experimental studies that confirm each step in the causal pathway
           - Find at least 2-3 independent sources for major claims
           - Cite specific papers with authors, years, and URLs
           - Note the quality and recency of sources

        2. **CONTRADICTION AND LIMITATION ANALYSIS**:
           - Actively search for studies that contradict the proposed mechanism
           - Look for boundary conditions where the mechanism fails
           - Identify conflicting experimental results
           - Search for review papers that discuss controversies

        3. **QUANTITATIVE DATA MINING**:
           - Search specifically for numerical values: carrier concentrations, band gaps, formation energies, reaction rates
           - Find experimental parameter ranges: temperatures, pressures, time scales
           - Look for statistical analyses and error bars
           - Compare values across different studies

        4. **RECENT DEVELOPMENTS**:
           - Prioritize papers from 2022-2024
           - Search for "recent advances", "latest research", "breakthrough"
           - Note if recent work contradicts older findings
           - Identify emerging trends or new understanding

        5. **MECHANISTIC DEPTH**:
           - Search for detailed reaction mechanisms and pathways
           - Find computational studies (DFT, molecular dynamics)
           - Look for in-situ characterization studies
           - Search for kinetic and thermodynamic analyses

        **Output Format Requirements:**

        Structure your response as a comprehensive research report with detailed citations:

        ### Comprehensive Research Analysis

        **1. Multi-Source Validation**
        *(For each major claim, provide 2-3 citations with specific details: "According to Smith et al. (2023) [URL], the carrier concentration increases by X. This is confirmed by Johnson et al. (2024) [URL] who found Y. However, Brown et al. (2022) [URL] reported conflicting results...")*

        **2. Contradiction and Controversy Analysis**
        *(Explicitly discuss conflicting findings: "While the knowledge graph suggests X, recent work by [Author] challenges this by showing Y. The discrepancy may be due to...")*

        **3. Quantitative Data Summary**
        *(Provide specific numerical ranges with citations: "Experimental values for hole concentration range from X to Y cm^-2 (Source A), with typical values around Z cm^-2 (Source B)...")*

        **4. Temporal Analysis**
        *(Discuss how understanding has evolved: "Early work (2018-2020) suggested X, but recent studies (2023-2024) have refined this to Y...")*

        **5. Mechanistic Synthesis**
        *(Integrate findings into a coherent mechanism with proper attribution)*

        **6. Confidence Assessment and Limitations**
        *(Provide nuanced confidence scores based on source quality, consensus, and recency)*

        ```json
        {{
          "{output_format_label}": {example_output},
          "validation_summary": {{
            "supporting_sources": ["Source 1 with specific findings", "Source 2 with specific findings"],
            "contradicting_sources": ["Source X showing conflicting result Y"],
            "consensus_level": "high/medium/low",
            "key_controversies": ["Controversy 1", "Controversy 2"]
          }},
          "quantitative_data": {{
            "parameter_1": {{"value": "X ± Y", "unit": "cm^-2", "source": "Author et al. 2024", "confidence": "high"}},
            "parameter_2": {{"range": "A-B", "unit": "eV", "source": "Multiple studies", "confidence": "medium"}}
          }},
          "temporal_analysis": {{
            "recent_developments": ["2024 finding 1", "2023 finding 2"],
            "paradigm_shifts": ["Change from old understanding to new"],
            "emerging_trends": ["Trend 1", "Trend 2"]
          }},
          "mechanistic_explanation": {{
            "validated_steps": ["Step 1 confirmed by Source A", "Step 2 confirmed by Source B"],
            "disputed_steps": ["Step X disputed by Source Y"],
            "missing_links": ["Gap 1 needs more research", "Gap 2 unclear"]
          }},
          "confidence_analysis": {{
            "overall_confidence": 0.85,
            "confidence_factors": {{
              "source_quality": 0.9,
              "source_quantity": 0.8,
              "consensus_level": 0.85,
              "recency": 0.9,
              "quantitative_support": 0.8
            }}
          }},
          "limitations_and_caveats": ["Limitation 1", "Limitation 2", "Caveat 1"],
          "summary_of_reasoning": "Comprehensive analysis based on X sources, with Y supporting and Z contradicting studies..."
        }}
        ```

        **REMEMBER**: Every major scientific claim must be backed by specific, cited sources. Use your search tool extensively!
        """

        print("Conducting comprehensive literature search and analysis...")
        response = self.model.generate_content(prompt)
        print(response.text)
        
        # Enhanced citation analysis
        sources = []
        if response.candidates and response.candidates[0].grounding_metadata:
            metadata = response.candidates[0].grounding_metadata
            print(f"\n📊 ENHANCED SEARCH ANALYSIS:")
            
            # Search queries analysis
            if hasattr(metadata, 'web_search_queries'):
                print(f"\n🔍 Search Queries Used ({len(metadata.web_search_queries)} total):")
                for i, query in enumerate(metadata.web_search_queries, 1):
                    print(f"  {i}. '{query}'")
            
            # Source extraction
            if hasattr(metadata, 'grounding_chunks'):
                for chunk in metadata.grounding_chunks:
                    sources.append({
                        "title": chunk.web.title,
                        "uri": chunk.web.uri
                    })
                print(f"\n📚 Sources Found: {len(sources)} unique sources")
            
            # Detailed citation analysis
            citations = self._extract_detailed_citations(response, sources)
            
            print(f"\n✅ CITATION QUALITY ANALYSIS:")
            print(f"  - Well-cited claims: {len(citations['coverage_analysis']['well_cited_claims'])}")
            print(f"  - Poorly-cited claims: {len(citations['coverage_analysis']['poorly_cited_claims'])}")
            print(f"  - Uncited claims: {len(citations['coverage_analysis']['uncited_claims'])}")
            
            # Detailed citation mapping
            print(f"\n📖 DETAILED CITATION MAPPING:")
            for i, citation in enumerate(citations["citation_mapping"][:5], 1):  # Show first 5
                print(f"\n[{i}] Claim: \"{citation['claim'][:100]}...\"")
                print(f"    Quality: {citation['citation_quality']}")
                print(f"    Sources: {len(citation['sources'])}")
                for source in citation['sources']:
                    print(f"      - {source['title']} ({source['uri']})")
            
            # Store citation data in response
            response._citation_analysis = citations
        else:
            print("\n⚠️ WARNING: Search tool was not used effectively!")
        
        return extract_json_from_response(response.text)

    def _transfer_learning_query(self, original_prompt_data: dict, analogous_context: str, confidence: float, query_type: str, 
                                 similar_node: str = None, query_string: str = None):
        """
        Enhanced transfer learning query with improved search and validation.
        """
        print(f"WARNING: No exact path found. Using enhanced transfer learning with confidence {confidence:.2f}")

        if query_type == "forward":
            task_description = "predict the resulting material properties with mechanistic explanation"
            input_data_label = "Target Synthesis Conditions"
            output_format_label = "predicted_properties"
            example_output = '{"carrier_type": "p-type", "band_gap_ev": 1.5}'
        else: # inverse
            task_description = "suggest synthesis conditions with mechanistic justification"
            input_data_label = "Target Material Properties"
            output_format_label = "suggested_synthesis_conditions"
            example_output = '{"method": "Chemical Vapor Deposition", "temperature_c": 800}'
        
        property_embedding_diff = 1.0
        if query_type == "inverse" and similar_node and query_string:
            property_embedding_diff = self._calculate_embedding_difference(query_string, similar_node)

        # Extract mechanisms for the analogous path
        mechanisms = self._get_path_mechanisms(analogous_context)
        formatted_mechanisms = "\n- ".join(mechanisms) if mechanisms else "No specific mechanisms provided"

        # Generate search queries for transfer learning
        transfer_queries = [
            f"similar mechanisms {query_string} materials science",
            f"analogous processes {analogous_context.replace(' -> ', ' ')}",
            f"transfer learning {query_string} experimental validation",
            f"comparative study {query_string} vs {similar_node}",
            f"recent advances {query_string} 2023 2024"
        ]

        prompt = f"""
        You are an expert materials scientist AI conducting transfer learning analysis. Your knowledge graph lacks exact pathways, but you've identified analogous information that requires careful validation and adaptation.

        **MANDATORY REQUIREMENTS:**
        1. **EXTENSIVE SEARCH**: Use your search tool to find studies on both the analogous system AND the target system
        2. **COMPARATIVE ANALYSIS**: Search for direct comparisons or studies that bridge the two systems
        3. **UNCERTAINTY QUANTIFICATION**: Be explicit about confidence levels and limitations
        4. **RECENT VALIDATION**: Prioritize recent studies (2022-2024) that might validate or refute the analogy

        **Task:**
        Based on analogous information and comprehensive literature search, {task_description} for the user's target.

        **{input_data_label} (User's Query):**
        {json.dumps(original_prompt_data, indent=2)}

        **Most Similar Known Causal Pathway:**
        - {analogous_context}

        **Known Mechanisms for Similar Pathway:**
        - {formatted_mechanisms}

        **Similarity Analysis:**
        - Embedding distance: {property_embedding_diff:.4f} (0=identical, 2=opposite)
        - Most similar known case: {similar_node}

        **Suggested Search Queries:**
        - {chr(10).join(f"  - {q}" for q in transfer_queries)}

        **Your Analysis Process:**

        1. **VALIDATE THE ANALOGY**: Search for studies that directly compare or relate the analogous system to your target system
        
        2. **IDENTIFY KEY DIFFERENCES**: Search for fundamental differences that might invalidate the analogy
        
        3. **ADAPTATION STRATEGY**: Based on literature findings, explain how to adapt the known mechanism
        
        4. **EXPERIMENTAL VALIDATION**: Search for any experimental work on your specific target system
        
        5. **UNCERTAINTY ASSESSMENT**: Quantify confidence based on literature support

        **Output Format:**
        Provide comprehensive analysis with extensive citations:

        ```json
        {{
          "{output_format_label}": {example_output},
          "analogy_validation": {{
            "supporting_evidence": ["Study 1 showing similarity", "Study 2 confirming mechanism"],
            "contradicting_evidence": ["Study X showing key difference"],
            "analogy_strength": "strong/moderate/weak",
            "key_similarities": ["Similarity 1", "Similarity 2"],
            "critical_differences": ["Difference 1", "Difference 2"]
          }},
          "adapted_mechanism": {{
            "original_mechanism": "Mechanism from analogous system",
            "adaptation_rationale": "Why and how to modify for target system",
            "modified_mechanism": "Adapted mechanism for target system",
            "literature_support": ["Supporting study 1", "Supporting study 2"]
          }},
          "confidence_analysis": {{
            "overall_confidence": {confidence:.4f},
            "confidence_breakdown": {{
              "analogy_validity": 0.0,
              "literature_support": 0.0,
              "mechanistic_understanding": 0.0,
              "experimental_validation": 0.0
            }},
            "major_uncertainties": ["Uncertainty 1", "Uncertainty 2"]
          }},
          "experimental_recommendations": [
            "Experiment 1 to validate analogy",
            "Experiment 2 to test adapted mechanism"
          ],
          "property_embedding_distance": {property_embedding_diff:.4f},
          "analogous_path_used": "{analogous_context}"
        }}
        ```
        """

        print("Conducting enhanced transfer learning analysis...")
        response = self.model.generate_content(prompt)
        return extract_json_from_response(response.text)

    def forward_prediction(self, synthesis_inputs: dict):
        """
        Enhanced forward prediction with comprehensive search and validation.
        """
        print("\n--- Starting Enhanced Forward Prediction ---")
        input_keywords = [str(v) for v in synthesis_inputs.values() if v is not None]
        query_string = " and ".join(input_keywords)
        
        all_properties = [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]
        
        causal_context_list = self._find_relevant_paths(input_keywords, all_properties)

        if causal_context_list:
            return self._enhanced_direct_path_query(synthesis_inputs, causal_context_list, "forward")
        else:
            all_synthesis_params = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
            similar_node, score = self._find_most_similar_node(query_string, all_synthesis_params)
            
            if similar_node and score > 0.5:
                analogous_paths = self._find_relevant_paths([similar_node], all_properties)
                if analogous_paths:
                    return self._transfer_learning_query(
                        synthesis_inputs, analogous_paths[0], score, "forward",
                        similar_node=similar_node, query_string=query_string
                    )

        print("No direct path or sufficiently similar node found. Using general knowledge fallback.")
        return {
            "error": "Failed to find a direct or analogous path in the knowledge graph.",
            "confidence": 0.0,
            "suggestion": "Consider expanding the knowledge graph or use a general-purpose query."
        }

    def inverse_design(self, desired_properties: dict):
        """
        Enhanced inverse design with comprehensive search and validation.
        """
        print("\n--- Starting Enhanced Inverse Design ---")
        property_keywords = [str(v) for v in desired_properties.values() if v is not None]
        query_string = " and ".join(property_keywords)

        all_synthesis_params = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]

        causal_context_list = self._find_relevant_paths(all_synthesis_params, property_keywords, reverse=True)
        
        if causal_context_list:
            return self._enhanced_direct_path_query(desired_properties, causal_context_list, "inverse")
        else:
            all_properties = [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]
            similar_node, score = self._find_most_similar_node(query_string, all_properties)

            if similar_node and score > 0.5:
                analogous_paths = self._find_relevant_paths(all_synthesis_params, [similar_node], reverse=True)
                if analogous_paths:
                    return self._transfer_learning_query(
                        desired_properties, analogous_paths[0], score, "inverse",
                        similar_node=similar_node, query_string=query_string
                    )

        print("No direct path or sufficiently similar node found. Using general knowledge fallback.")
        return {
            "error": "Failed to find a direct or analogous path in the knowledge graph.",
            "confidence": 0.0,
            "suggestion": "Consider expanding the knowledge graph or use a general-purpose query."
        }


if __name__ == '__main__':
    # Test the enhanced engine
    json_file = '../outputs/filtered_combined_doping_data.json'
    
    try:
        engine = EnhancedCausalReasoningEngine(json_file)

        # Test with enhanced capabilities
        synthesis_params_exact = {
            "temperature": "200°C",
            "method": "Oxidation"
        }
        predicted_props = engine.forward_prediction(synthesis_params_exact)
        print("\nEnhanced Forward Prediction Result:")
        print(json.dumps(predicted_props, indent=2))

    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error initializing or running enhanced engine: {e}")