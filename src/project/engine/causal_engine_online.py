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

from project.utils.parser import extract_json_from_response

class CausalReasoningEngine:
    """
    Enhanced version with improved search capabilities, citation granularity,
    contradiction detection, quantitative validation, and temporal awareness.
    """
    def __init__(self, json_file_path: str, model_id: str = "gemini-1.5-pro-latest", api_type: str = "gemini", embedding_model: str = 'all-MiniLM-L6-v2'):
        self.graph = self._build_graph(json_file_path)
        self._configure_api(model_id, api_type)
        print(f"Loading sentence transformer model ('{embedding_model}')...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self._precompute_node_embeddings()
        print("Enhanced Causal Reasoning Engine initialized.")
        print(f"Knowledge graph contains {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        self.forward_output_format = """```json
        {{
            "reasoning": "...",
            "predicted_properties": {{
            "doping_outcome": "...",
            "structure_changes": "...",
            "phase_transition": "...",
            "defect_formation": "...",
            "distribution_characteristics": "...",
            "property_changes": "...",
            "thermal": "...",
            "mechanical": "...",
            "optical": "...",
            "...": "..."
            }},
            "confidence": a float between 0 and 1
        }}
        ```"""
        self.inverse_output_format = """```json
        {{
            "reasoning": "...",
            "suggested_synthesis_conditions": {{
            "host_material": "...",
            "dopant":{{
            "element": "...",
            "concentration": "...",
            "precursor": "..."}},
            "method": "...",
            "temperature_c": ...,
            "pressure_pa": ...,
            "time_hours": ...,
            "atmosphere": ...,
            "electric_field": ...,
            "cooling_rate_c_min": ...,
            "substrate_pretreatment": ...,
            "additional_parameters": ...,
            "...": "..."
            }},
            "confidence": a float between 0 and 1
        }}
        ```"""

    def _configure_api(self, model_id: str, api_type: str):
        api_key = os.getenv("GOOGLE_API_KEY")
        # Enhanced grounding tool configuration for better search coverage
        
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY")
        genai.configure(api_key=api_key)

        self.api_type = api_type
        self.model_id = model_id
        grounding_tool = Tool(google_search_retrieval={
            'dynamic_retrieval_config': {
                'mode': 'MODE_DYNAMIC',
                'dynamic_threshold': 0.7
            }
        })
        self.model = genai.GenerativeModel(self.model_id, tools=[grounding_tool])
    def _generate_content(self, prompt: str):
        return self.model.generate_content(prompt)
    
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

    def _extract_and_analyze_citations(self, response) -> Dict:
        """
        Enhanced citation extraction with analysis, printing, and result preparation.
        """
        sources = []
        citations = {
            "total_sources": 0,
            "source_details": [],
            "citation_mapping": [],
            "coverage_analysis": {
                "well_cited_claims": [],
                "poorly_cited_claims": [],
                "uncited_claims": []
            }
        }
        
        if not (response.candidates and response.candidates[0].grounding_metadata):
            print("\n⚠️ WARNING: Search tool was not used effectively!")
            return None
            
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
        
        # Update citations with source info
        citations["total_sources"] = len(sources)
        citations["source_details"] = sources
        
        # Detailed citation extraction
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

    def _direct_path_query(self, original_prompt_data: dict, causal_paths: list, query_type: str):
        """
        Significantly enhanced query with all five improvements implemented.
        """
        print(f"Found {len(causal_paths)} direct causal paths. Conducting comprehensive validation...")

        # Generate comprehensive search queries
        search_queries = self._generate_comprehensive_search_queries(original_prompt_data, causal_paths, query_type)
        
        # Mechanism extraction
        all_mechanisms = []
        for path in causal_paths:
            mechanisms = self._get_path_mechanisms(path_str=path)
            if mechanisms:
                all_mechanisms.extend(mechanisms)

        if query_type == "forward":
            task_desc = "predict material properties from synthesis conditions"
            input_label = "Synthesis Conditions"
            output_format = self.forward_output_format
        else:
            task_desc = "suggest synthesis conditions for desired properties"
            input_label = "Desired Properties"
            output_format = self.inverse_output_format
        
        # Format DAG knowledge
        formatted_paths = "\n- ".join(causal_paths)
        all_mechanisms = []
        for path in causal_paths:
            mechanisms = self._get_path_mechanisms(path)
            all_mechanisms.extend(mechanisms)
        formatted_mechanisms = "\n- ".join(all_mechanisms) if all_mechanisms else "No specific mechanisms provided"
        # Enhanced prompt with all five improvements
        prompt = f"""
        You are an expert materials scientist with access to a specialized knowledge graph and online search tool.
        Your task is to {task_desc} by intelligently combining your baseline scientific knowledge with relevant research findings.
        
        **CRITICAL INSTRUCTION: Your final answer must be AT LEAST as good as pure baseline reasoning. Use DAG knowledge and online search tool to ENHANCE, not replace, fundamental principles.**

        **{input_label}:**
        {json.dumps(original_prompt_data, indent=2)}
        
        **Relevant Research Knowledge from Literature:**
        Causal Pathways:
        - {formatted_paths}
        
        Known Mechanisms:
        - {formatted_mechanisms}
        
        **Your Instructions:**

        1.  **Validate and Search:** Critically evaluate the proposed pathway. Use your search tool to find scientific papers or reliable sources that confirm, contradict, or add nuance to this hypothesis. Cite your key sources with URLs.
        2.  **Synthesize and Explain:** Based on your search findings and internal knowledge, write a comprehensive mechanistic explanation. Do not just repeat the graph data; synthesize it into a coherent scientific narrative. 
        3.  **Provide Quantitative Analysis:** Search for and include relevant quantitative data. 
        4.  **Discuss Alternatives and Nuances:** Briefly mention any alternative or competing mechanisms that could be at play. What are the key assumptions or boundary conditions for this process to work?

        **Output Requirements:**
        - Reason step by step until you reach the final answer, put the final answer in the json format
        - Your answer must be logical, scientifically rigorous and practically useful
        - Include mechanistic explanations combining both knowledge sources
        - Provide confidence levels and uncertainty analysis

        **JSON Output Format:**
        {output_format}
        """

        print("Conducting comprehensive literature search and analysis...")
        response = self._generate_content(prompt)
        response_text = response.text
        result = extract_json_from_response(response_text)
        
        # Enhanced citation analysis using unified function
        citations = self._extract_and_analyze_citations(response)
        result["citation_analysis"] = citations
        
        result["raw_response"] = response_text
        result["method"] = "online"
        result["dag_enhancement"] = False
        return result

    def _transfer_learning_query(self, original_prompt_data: dict, analogous_context: str, confidence: float, query_type: str, 
                                 similar_node: str = None, query_string: str = None):
        """
        Enhanced transfer learning query with improved search and validation.
        """
        print(f"WARNING: No exact path found. Using enhanced transfer learning with confidence {confidence:.2f}")

        if query_type == "forward":
            task_description = "predict the resulting material properties with mechanistic explanation"
            input_data_label = "Target Synthesis Conditions"
            output_format = self.forward_output_format
        else: # inverse
            task_description = "suggest synthesis conditions with mechanistic justification"
            input_data_label = "Target Material Properties"
            output_format = self.inverse_output_format
        
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
        ]

        prompt = f"""
        You are an expert materials scientist AI conducting transfer learning analysis. Your task is to {task_description} by intelligently combining your baseline scientific knowledge with relevant research findings. But your knowledge graph lacks exact pathways, but you've identified analogous information that requires careful validation and adaptation.

        **Your Instructions:**

        1.  **Validate and Search:** Critically evaluate the proposed pathway. Use your search tool to find scientific papers or reliable sources that confirm, contradict, or add nuance to this hypothesis. Cite your key sources with URLs.
        2.  **Synthesize and Explain:** Based on your search findings and internal knowledge, write a comprehensive mechanistic explanation. Do not just repeat the graph data; synthesize it into a coherent scientific narrative. 
        3.  **Provide Quantitative Analysis:** Search for and include relevant quantitative data. 
        4.  **Discuss Alternatives and Nuances:** Briefly mention any alternative or competing mechanisms that could be at play. What are the key assumptions or boundary conditions for this process to work?

        **Output Requirements:**
        - Reason step by step until you reach the final answer, put the final answer in the json format
        - Your answer must be logical, scientifically rigorous and practically useful
        - Include mechanistic explanations combining both knowledge sources
        - Provide confidence levels and uncertainty analysis

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

        **Your Instructions:**

        1.  **Validate and Search:** Critically evaluate the proposed pathway. Use your search tool to find scientific papers or reliable sources that confirm, contradict, or add nuance to this hypothesis. Cite your key sources with URLs.
        2.  **Synthesize and Explain:** Based on your search findings and internal knowledge, write a comprehensive mechanistic explanation. Do not just repeat the graph data; synthesize it into a coherent scientific narrative. 
        3.  **Provide Quantitative Analysis:** Search for and include relevant quantitative data. 
        4.  **Discuss Alternatives and Nuances:** Briefly mention any alternative or competing mechanisms that could be at play. What are the key assumptions or boundary conditions for this process to work?

        **Output Requirements:**
        - First, analysis the provided causal paths and mechanisms (if provided), then reason step by step until you reach the final answer, put the final answer in the json format
        - Your answer must be logical, scientifically rigorous and practically useful
        - Include mechanistic explanations combining both knowledge sources
        - Provide confidence levels and uncertainty analysis

        **JSON Output Format:**
        {output_format}
        """

        print("Conducting enhanced transfer learning analysis...")
        response = self._generate_content(prompt)
        response_text = response.text
        result = extract_json_from_response(response_text)
        
        # Enhanced citation analysis using unified function
        citations = self._extract_and_analyze_citations(response)
        result["citation_analysis"] = citations
        
        result["raw_response"] = response_text
        result["method"] = "online"
        result["dag_enhancement"] = False
        return result
    
    def _baseline_fallback_query(self, original_prompt_data: dict, query_type: str, reason: str = "No suitable DAG knowledge found"):
        """
        Pure baseline query that matches the baseline model performance exactly. This ensures we never underperform the baseline.
        """
        print(f"🔄 Using baseline fallback: {reason}")
        
        if query_type == "forward":
            task_desc = "predict the resulting material properties"

            output_format = self.forward_output_format
       
        else:
            task_desc = "suggest synthesis conditions to achieve the desired properties"

            output_format = self.inverse_output_format
        
        baseline_prompt = f"""
        You are an expert materials scientist. Based on the following {'synthesis conditions' if query_type == 'forward' else 'desired properties'}, 
        {task_desc}.
        
        {'Synthesis Conditions' if query_type == 'forward' else 'Desired Properties'}:
        {json.dumps(original_prompt_data, indent=2)}
        
        Search for relevant literature and reason step by step until you reach the final answer, then provide your answer in a structured JSON format within a ```json block.
        
        **JSON Output Format:**
        {output_format}
        """
        
        response = self._generate_content(baseline_prompt)
        response_text = response.text
        result = extract_json_from_response(response_text)
        
        # Enhanced citation analysis using unified function
        citations = self._extract_and_analyze_citations(response)
        result["citation_analysis"] = citations
        
        # Add metadata to indicate this is baseline performance
        result["raw_response"] = response_text
        result["method"] = "baseline_fallback"
        result["dag_enhancement"] = False
        return result

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
            return self._direct_path_query(synthesis_inputs, causal_context_list, "forward")
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
        return self._baseline_fallback_query(synthesis_inputs, "forward")

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
            return self._direct_path_query(desired_properties, causal_context_list, "inverse")
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
        return self._baseline_fallback_query(desired_properties, "inverse") 


if __name__ == '__main__':
    # Test the enhanced engine
    json_file = 'datas/filtered_combined_doping_data.json'
    
    engine = CausalReasoningEngine(json_file, api_type="gemini")
    test_cases = [
            # {
            #     "name": "Direct Match Test",
            #     "synthesis": {"temperature": "200°C", "method": "Oxidation"},
            #     "properties": {"doping": "Controllable p-type doping"}
            # },
            {
                "name": "Similarity Match Test", 
                "synthesis": {"temperature": "210°C", "method": "Air annealing"},
                "properties": {"doping": "p-type conductivity enhancement"}
            },
            {
                "name": "Low Similarity Test",
                "synthesis": {"temperature": "1000°C", "method": "Plasma treatment"},
                "properties": {"doping": "quantum dot formation"}
            },
            {
                "name": "Edge Case Test",
                "synthesis": {"temperature": None, "method": ""},
                "properties": {"doping": ""}
            }
        ]
        
    for test_case in test_cases:
        print(f"\n🧪 {test_case['name']}")

        # Forward prediction
        print("Forward Prediction:")
        forward_result = engine.forward_prediction(test_case['synthesis'])
        print(forward_result)
        
        # Inverse design
        print("\nInverse Design:")
        inverse_result = engine.inverse_design(test_case['properties'])
        print(inverse_result)
    # # Test with enhanced capabilities
    # synthesis_params_exact = {
    #     "temperature": "200°C",
    #     "method": "Oxidation"
    # }
    # predicted_props = engine.forward_prediction(synthesis_params_exact)
    # print("\nEnhanced Forward Prediction Result:")
    # print(json.dumps(predicted_props, indent=2))