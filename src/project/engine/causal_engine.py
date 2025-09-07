import json
import networkx as nx
import os
import google.generativeai as genai
from pathlib import Path
import textwrap
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException
import re

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from project.prompts.forward_design import ForwardTransferChain
from project.prompts.inverse_design import InverseTransferChain
from project.prompts.forward_design import ForwardDirectChain
from project.prompts.inverse_design import InverseDirectChain

class CausalReasoningEngine:
    """
    Integrates a Directed Acyclic Graph (DAG) of scientific knowledge with the 
    Gemini LLM to perform constrained causal reasoning for materials science.
    Includes a similarity-based confidence score and a transfer-learning fallback mechanism.
    """
    def __init__(self, json_file_path: str, model_id: str = "gemini-1.5-pro-latest", embedding_model: str = 'all-MiniLM-L6-v2', config_dir: str = None):
        self.graph = self._build_graph(json_file_path)
        logger.info(f"Loading sentence transformer model ('{embedding_model}')...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self._precompute_node_embeddings()
        
        # Initialize prompt configuration loader
        self.forward_direct_chain = ForwardDirectChain()
        self.forward_transfer_chain = ForwardTransferChain()
        self.inverse_direct_chain = InverseDirectChain()
        self.inverse_transfer_chain = InverseTransferChain()
        
        logger.info("Causal Reasoning Engine initialized.")
        logger.info(f"Knowledge graph contains {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def _build_graph(self, json_file_path: str):
        # This function remains the same
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
        # This function remains the same
        print("Pre-computing node embeddings...")
        self.node_list = list(self.graph.nodes())
        if not self.node_list:
            self.node_embeddings = np.array([])
            return
        self.node_embeddings = self.embedding_model.encode(self.node_list, convert_to_tensor=False)
        print("Embeddings computed.")

    def _find_most_similar_node(self, query: str, candidate_nodes: list):
        # This function remains the same
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

    # --- NEW: Helper function to calculate embedding distance ---
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
        # This function remains the same
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


    def forward_prediction(self, synthesis_inputs: dict):
        """
        Predicts material properties based on synthesis conditions.
        First attempts an exact match, then falls back to similarity-based transfer learning.
        """
        print("\n--- Starting Forward Prediction ---")
        # Convert all values to strings and filter out None before joining
        input_keywords = [str(v) for v in synthesis_inputs.values() if v is not None]
        query_string = " and ".join(input_keywords)
        
        all_properties = [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]
        causal_context_list = self._find_relevant_paths(input_keywords, all_properties)

        if causal_context_list:
            print(f"Found {len(causal_context_list)} direct causal paths to constrain reasoning.")
            formatted_context = "\n- ".join(causal_context_list)
            
            # Extract mechanisms from the graph edges
            mechanisms = []
            for path in causal_context_list:
                nodes = path.split(" -> ")
                for i in range(len(nodes) - 1):
                    if self.graph.has_edge(nodes[i], nodes[i+1]):
                        mechanism = self.graph[nodes[i]][nodes[i+1]].get('mechanism', 'No mechanism available')
                        if mechanism and mechanism.strip():
                            mechanisms.append(mechanism)
            formatted_mechanisms = "\n- ".join(mechanisms) if mechanisms else "No specific mechanisms available"
            
            # Use LangChain forward direct prompt
            
            result = self.forward_direct_chain.get_result({
                "synthesis_conditions": json.dumps(synthesis_inputs, indent=2),
                "causal_paths": formatted_context,
                "mechanisms": formatted_mechanisms
            })
            return result

        else:
            all_synthesis_params = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
            similar_node, score = self._find_most_similar_node(query_string, all_synthesis_params)
            
            if similar_node and score > 0.5:  # Confidence threshold
                analogous_paths = self._find_relevant_paths([similar_node], all_properties)
                if analogous_paths:
                    # Extract mechanisms for transfer learning
                    mechanisms = []
                    path = analogous_paths[0]
                    nodes = path.split(" -> ")
                    for i in range(len(nodes) - 1):
                        if self.graph.has_edge(nodes[i], nodes[i+1]):
                            mechanism = self.graph[nodes[i]][nodes[i+1]].get('mechanism', '')
                            if mechanism and mechanism.strip():
                                mechanisms.append(mechanism)
                    formatted_mechanisms = "\n- ".join(mechanisms) if mechanisms else "No specific mechanisms available"
                    
                    # Use LangChain forward transfer prompt

                    result = self.forward_transfer_chain.get_result({
                        "synthesis_conditions": json.dumps(synthesis_inputs, indent=2),
                        "analogous_context": analogous_paths[0],
                        "mechanisms": formatted_mechanisms,
                        "confidence": score
                    })
                    return result
                    

        print("No direct path or sufficiently similar node found.")
        return {
            "error": "Failed to find a direct or analogous path in the knowledge graph.",
            "confidence": 0.0,
            "suggestion": "Consider expanding the knowledge graph or use a general-purpose query."
        }


    def inverse_design(self, desired_properties: dict):
        """
        Suggests synthesis conditions to achieve desired material properties.
        First attempts an exact match, then falls back to similarity-based transfer learning.
        """
        print("\n--- Starting Inverse Design ---")
        # Convert all values to strings and filter out None before joining
        property_keywords = [str(v) for v in desired_properties.values() if v is not None]
        query_string = " and ".join(property_keywords)

        all_synthesis_params = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
        causal_context_list = self._find_relevant_paths(all_synthesis_params, property_keywords, reverse=True)
        
        if causal_context_list:
            print(f"Found {len(causal_context_list)} direct causal paths for inverse design.")
            formatted_context = "\n- ".join(causal_context_list)
            
            # Extract mechanisms from the graph edges
            mechanisms = []
            for path in causal_context_list:
                nodes = path.split(" -> ")
                for i in range(len(nodes) - 1):
                    if self.graph.has_edge(nodes[i], nodes[i+1]):
                        mechanism = self.graph[nodes[i]][nodes[i+1]].get('mechanism', 'No mechanism available')
                        if mechanism and mechanism.strip():
                            mechanisms.append(mechanism)
            formatted_mechanisms = "\n- ".join(mechanisms) if mechanisms else "No specific mechanisms available"
            
            # Use LangChain inverse direct prompt
            result = self.inverse_direct_chain.get_result({
                "desired_properties": json.dumps(desired_properties, indent=2),
                "causal_paths": formatted_context,
                "mechanisms": formatted_mechanisms
            })
            return result
        
        else:
            all_properties = [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]
            similar_node, score = self._find_most_similar_node(query_string, all_properties)

            if similar_node and score > 0.5:  # Confidence threshold
                analogous_paths = self._find_relevant_paths(all_synthesis_params, [similar_node], reverse=True)
                if analogous_paths:
                    # Extract mechanisms for transfer learning
                    mechanisms = []
                    path = analogous_paths[0]
                    nodes = path.split(" -> ")
                    for i in range(len(nodes) - 1):
                        if self.graph.has_edge(nodes[i], nodes[i+1]):
                            mechanism = self.graph[nodes[i]][nodes[i+1]].get('mechanism', '')
                            if mechanism and mechanism.strip():
                                mechanisms.append(mechanism)
                    formatted_mechanisms = "\n- ".join(mechanisms) if mechanisms else "No specific mechanisms available"
                    
                    # Calculate embedding distance for quantitative reasoning
                    property_embedding_diff = self._calculate_embedding_difference(query_string, similar_node)
                    
                    # Use LangChain inverse transfer prompt
                    result = self.inverse_transfer_chain.get_result({
                        "desired_properties": json.dumps(desired_properties, indent=2),
                        "analogous_context": analogous_paths[0],
                        "mechanisms": formatted_mechanisms,
                        "confidence": score,
                        "property_embedding_distance": property_embedding_diff
                    })
                    return result
                    

        print("No direct path or sufficiently similar node found.")
        return {
            "error": "Failed to find a direct or analogous path in the knowledge graph.",
            "confidence": 0.0,
            "suggestion": "Consider expanding the knowledge graph or use a general-purpose query."
        }


if __name__ == '__main__':
    # Ensure the JSON file and API key are correctly set up
    json_file = 'outputs/combined_doping_data.json' # Make sure this file exists
    
    try:
        engine = CausalReasoningEngine(json_file)

        # --- Example 1: Forward Prediction (Exact Match) ---
        # This should find a direct path in the graph.
        synthesis_params_exact = {
            "temperature": "200°C",
            "method": "Oxidation"
        }
        predicted_props = engine.forward_prediction(synthesis_params_exact)
        print("\nForward Prediction Result (Exact Match):")
        print(json.dumps(predicted_props, indent=2))

        # --- Example 2: Forward Prediction (Analogous/Transfer Learning) ---
        # This query is semantically similar but not identical to the one above.
        # It should trigger the transfer learning fallback.
        synthesis_params_analogous = {
            "temperature": "210°C",
            "method": "Annealing in an oxygen atmosphere"
        }
        predicted_props_analogous = engine.forward_prediction(synthesis_params_analogous)
        print("\nForward Prediction Result (Analogous Match):")
        print(json.dumps(predicted_props_analogous, indent=2))

        # --- Example 3: Inverse Design (Exact Match) ---
        target_properties_exact = {
            "doping": "Controllable p-type doping",
        }
        suggested_synthesis = engine.inverse_design(target_properties_exact)
        print("\nInverse Design Result (Exact Match):")
        print(json.dumps(suggested_synthesis, indent=2))
        
        # --- Example 4: Inverse Design (Analogous/Transfer Learning) ---
        # This query is semantically similar to a property in the graph but uses different wording.
        target_properties_analogous = {
            "doping": "Achieve tunable hole-based conductivity",
        }
        suggested_synthesis_analogous = engine.inverse_design(target_properties_analogous)
        print("\nInverse Design Result (Analogous Match):")
        print(json.dumps(suggested_synthesis_analogous, indent=2))


    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error initializing or running engine: {e}")
