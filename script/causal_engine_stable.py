import json
import networkx as nx
import os
import google.generativeai as genai
from pathlib import Path
import textwrap
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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


# (Helper functions like extract_json_from_response remain the same)
def extract_json_from_response(text: str):
    import re
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if not match:
        return {"warning": "No JSON object found in the response.", "raw_response": text}
    json_string = match.group(1)
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"\n--- JSON DECODE ERROR ---\n{e}\nProblematic text:\n{json_string}\n-------------------------\n")
        return {"error": "Failed to decode JSON", "details": str(e), "malformed_json_string": json_string}


class CausalReasoningEngine:
    """
    Integrates a Directed Acyclic Graph (DAG) of scientific knowledge with the 
    Gemini LLM to perform constrained causal reasoning for materials science.
    Includes a similarity-based confidence score and a transfer-learning fallback mechanism.
    """
    def __init__(self, json_file_path: str, model_id: str = "gemini-1.5-pro-latest", embedding_model: str = 'all-MiniLM-L6-v2'):
        self.graph = self._build_graph(json_file_path)
        self._configure_api(model_id)
        print(f"Loading sentence transformer model ('{embedding_model}')...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self._precompute_node_embeddings()
        print("Causal Reasoning Engine initialized.")
        print(f"Knowledge graph contains {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def _configure_api(self, model_id: str):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id)

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

    def _transfer_learning_query(self, original_prompt_data: dict, analogous_context: str, confidence: float, query_type: str, 
                                 similar_node: str = None, query_string: str = None):
        """
        --- MODIFIED: This prompt is now much more explicit about the reasoning process. ---
        Performs a query to Gemini by asking it to reason by analogy and show its work.
        """
        print(f"WARNING: No exact path found. Using most similar context with confidence {confidence:.2f} for transfer learning.")

        if query_type == "forward":
            task_description = "predict the resulting material properties"
            input_data_label = "Target Synthesis Conditions"
            output_format_label = "predicted_properties"
            example_output = '{"carrier_type": "p-type", "band_gap_ev": 1.5}'
        else: # inverse
            task_description = "suggest synthesis conditions to achieve the desired properties"
            input_data_label = "Target Material Properties"
            output_format_label = "suggested_synthesis_conditions"
            example_output = '{"method": "Chemical Vapor Deposition", "temperature_c": 800}'
        
        # --- NEW: Calculate embedding distance for quantitative reasoning ---
        property_embedding_diff = 1.0 # Default value
        if query_type == "inverse" and similar_node and query_string:
            property_embedding_diff = self._calculate_embedding_difference(query_string, similar_node)

        prompt = f"""
        You are an expert materials scientist AI. Your task is to reason from analogous data using a structured, interpretable process.
        Your knowledge graph does not contain an exact causal pathway for the user's query.
        However, you have identified the most semantically similar information available.

        **Task:**
        Based on the provided analogous information, {task_description} for the user's target.
        You must explicitly detail your reasoning process.

        **{input_data_label} (User's Query):**
        {json.dumps(original_prompt_data, indent=2)}

        **Most Similar Known Causal Pathway (from Knowledge Graph):**
        - {analogous_context}

        **Quantitative Analysis:**
        The embedding distance between the user's desired properties and the most similar known property is {property_embedding_diff:.4f} (where 0 is identical and 2 is opposite).

        **Your Reasoning Process (Mandatory):**
        1.  **Analyze & Compare:** Briefly compare the User's Query with the Known Pathway. What are the key similarities and, more importantly, the key differences (e.g., opposite doping type, different materials, different conditions)?
        2.  **Formulate Hypothesis:** Based on the differences and the quantitative embedding distance, state a hypothesis. For example: "The known pathway describes p-doping via oxidation. The user wants n-doping. The large embedding distance of ~1.0 confirms these concepts are dissimilar. Therefore, a reductive process or a dopant with more valence electrons is required, making the known pathway a poor analogy for direct parameter transfer."
        3.  **Extrapolate or Diverge:** Decide if you can adjust the parameters from the known pathway (extrapolate) or if you must suggest a completely different approach (diverge). Justify this decision using the embedding distance. A small distance (< 0.4) suggests extrapolation is viable; a large distance (> 0.7) suggests divergence is necessary.
        4.  **Synthesize Final Answer:** Based on your hypothesis, construct the final prediction/suggestion.

        **Output Format:**
        Provide your answer in a structured JSON format within a ```json block.
        The JSON MUST include a detailed 'transfer_learning_analysis' object containing your step-by-step reasoning.

        Example JSON output:
        {{
          "{output_format_label}": {example_output},
          "reasoning": "The user desires n-type properties, but the most similar context is for p-type doping via oxidation. Acknowledging this contradiction (embedding distance > 1.0), I suggest CVD with a Re precursor, a known n-type doping strategy, rather than adapting the oxidation parameters.",
          "confidence": {confidence:.4f},
          "analogous_path_used": "{analogous_context}",
          "property_embedding_distance": {property_embedding_diff:.4f},
          "transfer_learning_analysis": {{
              "1_comparison": "The user's query is for electron-doping (n-type). The known pathway is for hole-doping (p-type) via oxidation. The core electronic goal is opposite.",
              "2_hypothesis": "Because the desired electronic outcome is the inverse of the known pathway, confirmed by a large embedding distance, the synthesis method must also be fundamentally different. Simple parameter adjustments to the oxidation process are inappropriate.",
              "3_decision": "Diverge. The provided context serves as a counterexample. I will ignore its specific parameters and instead access my general knowledge about n-type doping for 2D materials.",
              "4_synthesis": "The final suggestion is based on general knowledge of using Group 7 elements like Rhenium (Re) to n-dope MoS2-like materials, typically via CVD."
          }},
          "suggested_next_steps": "Perform DFT calculations to model Re doping. Experimentally, synthesize via CVD and characterize using Hall effect and XPS."
        }}
        """
        # --- NEW: Print the full prompt for complete transparency ---
        print("\n" + "="*80)
        print("TRANSFER LEARNING PROMPT SENT TO GEMINI:")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")

        print("Querying Gemini with an interpretable transfer learning prompt...")
        response = self.model.generate_content(prompt)
        return extract_json_from_response(response.text)

    def forward_prediction(self, synthesis_inputs: dict):
        print("\n--- Starting Forward Prediction ---")
        input_keywords = [str(v) for v in synthesis_inputs.values() if v is not None]
        query_string = " and ".join(input_keywords)
        all_properties = [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]
        causal_context_list = self._find_relevant_paths(input_keywords, all_properties)
        if causal_context_list:
            formatted_context = "\n- ".join(causal_context_list)
            prompt = f"""
            You are an expert materials scientist AI. Based on the following synthesis conditions and known causal pathways from a scientific knowledge graph, predict the resulting material properties.

            **Synthesis Conditions:**
            {json.dumps(synthesis_inputs, indent=2)}

            **Known Causal Pathways (from Knowledge Graph):**
            - {formatted_context}

            **Task:**
            Predict the likely outcomes. Provide your answer in a structured JSON format within a ```json block, including a 'confidence' of '1.0' (exact match) and a 'reasoning' field.
            """
            print("Querying Gemini with high-confidence constrained prompt...")
            response = self.model.generate_content(prompt)
            return extract_json_from_response(response.text)
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
        return {"error": "No path found", "confidence": 0.0}


    def inverse_design(self, desired_properties: dict):
        print("\n--- Starting Inverse Design ---")
        property_keywords = [str(v) for v in desired_properties.values() if v is not None]
        query_string = " and ".join(property_keywords)
        all_synthesis_params = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
        causal_context_list = self._find_relevant_paths(all_synthesis_params, property_keywords, reverse=True)
        if causal_context_list:
            formatted_context = "\n- ".join(causal_context_list)
            prompt = f"""
            You are an expert materials scientist AI. Your task is to design a synthesis protocol to achieve specific material properties, guided by a knowledge graph.

            **Desired Material Properties:**
            {json.dumps(desired_properties, indent=2)}

            **Known Causal Pathways (from Knowledge Graph):**
            - {formatted_context}

            **Task:**
            Suggest a set of synthesis conditions. Provide your answer in a structured JSON format within a ```json block, including a 'confidence' of '1.0' (exact match) and a 'reasoning' field.
            """
            print("Querying Gemini with high-confidence constrained prompt...")
            response = self.model.generate_content(prompt)
            return extract_json_from_response(response.text)
        else:
            all_properties = [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]
            similar_node, score = self._find_most_similar_node(query_string, all_properties)
            if similar_node and score > 0.5:
                analogous_paths = self._find_relevant_paths(all_synthesis_params, [similar_node], reverse=True)
                if analogous_paths:
                    return self._transfer_learning_query(
                        desired_properties, analogous_paths[0], score, "inverse",
                        similar_node=similar_node, query_string=query_string
                    )
        return {"error": "No path found", "confidence": 0.0}


    def forward_prediction(self, synthesis_inputs: dict):
        """
        Predicts material properties based on synthesis conditions.
        First attempts an exact match, then falls back to similarity-based transfer learning.
        """
        print("\n--- Starting Forward Prediction ---")
        # --- FIX: Convert all values to strings and filter out None before joining ---
        input_keywords = [str(v) for v in synthesis_inputs.values() if v is not None]
        query_string = " and ".join(input_keywords)
        
        all_properties = [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]
        
        causal_context_list = self._find_relevant_paths(input_keywords, all_properties)

        if causal_context_list:
            print(f"Found {len(causal_context_list)} direct causal paths to constrain reasoning.")
            formatted_context = "\n- ".join(causal_context_list)
            prompt = f"""
            You are an expert materials scientist AI. Based on the following synthesis conditions and known causal pathways from a scientific knowledge graph, predict the resulting material properties.

            **Synthesis Conditions:**
            {json.dumps(synthesis_inputs, indent=2)}

            **Known Causal Pathways (from Knowledge Graph) :**
            - {formatted_context}

            **Task:**
            Predict the likely outcomes. Provide your answer in a structured JSON format within a ```json block, including a 'confidence' of '1.0' (exact match).
            
            Example JSON output:
            {{
              "predicted_properties": {{
                "carrier_type": "p-type",
                "carrier_concentration": "Increased hole density"
              }},
              "reasoning": "The provided knowledge graph explicitly states a direct causal link from the input conditions to this outcome.",
              "confidence": 1.0
            }}
            """
            print("Querying Gemini with high-confidence constrained prompt...")
            response = self.model.generate_content(prompt)
            return extract_json_from_response(response.text)

        else:
            all_synthesis_params = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
            similar_node, score = self._find_most_similar_node(query_string, all_synthesis_params)
            
            if similar_node and score > 0.5: # Confidence threshold
                analogous_paths = self._find_relevant_paths([similar_node], all_properties)
                if analogous_paths:
                    return self._transfer_learning_query(synthesis_inputs, analogous_paths[0], score, "forward")

        print("No direct path or sufficiently similar node found. Using general knowledge fallback.")
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
        # --- FIX: Convert all values to strings and filter out None before joining ---
        property_keywords = [str(v) for v in desired_properties.values() if v is not None]
        query_string = " and ".join(property_keywords)

        all_synthesis_params = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]

        causal_context_list = self._find_relevant_paths(all_synthesis_params, property_keywords, reverse=True)
        
        if causal_context_list:
            print(f"Found {len(causal_context_list)} direct causal paths for inverse design.")
            formatted_context = "\n- ".join(causal_context_list)
            prompt = f"""
            You are an expert materials scientist AI. Your task is to design a synthesis protocol to achieve specific material properties, guided by a knowledge graph.

            **Desired Material Properties:**
            {json.dumps(desired_properties, indent=2)}

            **Known Causal Pathways (from Knowledge Graph):**
            - {formatted_context}

            **Task:**
            Suggest a set of synthesis conditions. Provide your answer in a structured JSON format within a ```json block, including a 'confidence' of '1.0' (exact match).

            Example JSON output:
            {{
              "suggested_synthesis_conditions": {{
                "method": "Surface oxidation",
                "temperature_c": 200
              }},
              "reasoning": "The knowledge graph contains a direct path from these synthesis conditions to the desired outcome.",
              "confidence": 1.0
            }}
            """
            print("Querying Gemini with high-confidence constrained prompt...")
            response = self.model.generate_content(prompt)
            return extract_json_from_response(response.text)
        
        else:
            all_properties = [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]
            similar_node, score = self._find_most_similar_node(query_string, all_properties)

            if similar_node and score > 0.5: # Confidence threshold
                analogous_paths = self._find_relevant_paths(all_synthesis_params, [similar_node], reverse=True)
                if analogous_paths:
                    return self._transfer_learning_query(desired_properties, analogous_paths[0], score, "inverse", 
                                                       similar_node=similar_node, query_string=query_string)

        print("No direct path or sufficiently similar node found. Using general knowledge fallback.")
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
