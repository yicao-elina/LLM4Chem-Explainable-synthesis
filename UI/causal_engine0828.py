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

    def _direct_path_query(self, original_prompt_data: dict, causal_paths: list, query_type: str):
        """
        NEW: Enhanced query for when direct paths exist. Asks the LLM to provide
        mechanistic explanations and chain-of-thought reasoning.
        """
        print(f"Found {len(causal_paths)} direct causal paths. Requesting mechanistic explanation...")

        # Extract mechanisms for all paths
        all_mechanisms = []
        for path in causal_paths:
            mechanisms = self._get_path_mechanisms(path)
            if mechanisms:
                all_mechanisms.extend(mechanisms)

        if query_type == "forward":
            task_description = "explain the mechanistic pathway from synthesis conditions to material properties"
            input_data_label = "Synthesis Conditions"
            output_format_label = "predicted_properties"
            example_output = '{"carrier_type": "p-type", "band_gap_ev": 1.5, "carrier_concentration": "increased"}'
        else: # inverse
            task_description = "explain the mechanistic reasoning for selecting synthesis conditions to achieve desired properties"
            input_data_label = "Desired Material Properties"
            output_format_label = "suggested_synthesis_conditions"
            example_output = '{"method": "Surface oxidation", "temperature_c": 200, "duration_hours": 2}'

        formatted_paths = "\n- ".join(causal_paths)
        formatted_mechanisms = "\n- ".join(all_mechanisms) if all_mechanisms else "No specific mechanisms provided in knowledge graph"

        prompt = f"""
        You are an expert materials scientist AI. Your knowledge graph contains direct causal pathways relevant to the user's query.
        Your task is to {task_description} using both the specific knowledge from the graph AND your general chemical understanding.

        **{input_data_label} (User's Query):**
        {json.dumps(original_prompt_data, indent=2)}

        **Direct Causal Pathways from Knowledge Graph:**
        - {formatted_paths}

        **Known Mechanisms from Knowledge Graph:**
        - {formatted_mechanisms}

        **Your Task:**
        1. **Mechanistic Analysis**: Explain the chemical/physical mechanisms underlying each step in the causal pathway. Draw upon your general knowledge to fill in details not explicitly stated in the graph.
        
        2. **Chain-of-Thought Reasoning**: Provide a step-by-step logical explanation of how each cause leads to its effect. Include:
           - Electronic structure changes
           - Defect chemistry
           - Thermodynamic considerations
           - Kinetic factors
           - Structure-property relationships

        3. **Quantitative Insights**: Where possible, provide quantitative estimates or ranges based on typical values in materials science.

        4. **Alternative Pathways**: Briefly mention any alternative mechanisms that could lead to similar outcomes.

        **Output Format:**
        Provide your answer in a structured JSON format within a ```json block.
        The JSON MUST include detailed mechanistic reasoning and chain-of-thought analysis.

        Example JSON output:
        {{
          "{output_format_label}": {example_output},
          "mechanistic_explanation": {{
            "primary_mechanism": "Surface oxidation creates oxygen vacancies that act as p-type dopants by removing electrons from the valence band...",
            "electronic_effects": "The removal of electrons creates holes in the valence band, shifting the Fermi level toward the valence band edge...",
            "defect_chemistry": "O2 molecules adsorb on the surface and extract electrons: O2 + 2e- → 2O-. This leaves behind holes...",
            "thermodynamics": "At 200°C, the Gibbs free energy favors oxygen chemisorption without bulk oxidation...",
            "kinetics": "The reaction rate follows Arrhenius behavior with activation energy ~0.5 eV..."
          }},
          "chain_of_thought": [
            "Step 1: At 200°C, oxygen molecules adsorb on the MoS2 surface",
            "Step 2: Oxygen abstracts electrons from Mo d-orbitals, creating Mo5+ states",
            "Step 3: These oxidized states act as acceptors, creating holes",
            "Step 4: Hole concentration increases with oxidation time following sqrt(t) kinetics",
            "Step 5: The Fermi level shifts toward the valence band, establishing p-type behavior"
          ],
          "quantitative_estimates": {{
            "hole_concentration": "~10^12 to 10^13 cm^-2 for monolayer",
            "fermi_level_shift": "~0.1-0.2 eV toward valence band",
            "activation_energy": "~0.5 eV for oxygen chemisorption"
          }},
          "alternative_mechanisms": [
            "Substitutional doping with group V elements could also achieve p-type behavior",
            "Plasma treatment could accelerate the oxidation process"
          ],
          "confidence": 1.0,
          "reasoning": "Direct causal pathway found in knowledge graph, enhanced with mechanistic understanding from general chemistry knowledge."
        }}
        """

        print("Querying Gemini for mechanistic explanation of direct pathway...")
        response = self.model.generate_content(prompt)
        return extract_json_from_response(response.text)

    def _transfer_learning_query(self, original_prompt_data: dict, analogous_context: str, confidence: float, query_type: str, 
                                 similar_node: str = None, query_string: str = None):
        """
        Enhanced transfer learning query that includes mechanistic reasoning.
        """
        print(f"WARNING: No exact path found. Using most similar context with confidence {confidence:.2f} for transfer learning.")

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

        prompt = f"""
        You are an expert materials scientist AI. Your task is to reason from analogous data using mechanistic understanding.
        Your knowledge graph does not contain an exact causal pathway for the user's query, but you have identified similar information.

        **Task:**
        Based on the provided analogous information and your chemical knowledge, {task_description} for the user's target.
        You must provide detailed mechanistic reasoning and chain-of-thought analysis.

        **{input_data_label} (User's Query):**
        {json.dumps(original_prompt_data, indent=2)}

        **Most Similar Known Causal Pathway:**
        - {analogous_context}

        **Known Mechanisms for Similar Pathway:**
        - {formatted_mechanisms}

        **Quantitative Analysis:**
        The embedding distance between the user's query and the most similar known case is {property_embedding_diff:.4f} 
        (where 0 is identical and 2 is opposite).

        **Your Reasoning Process (Mandatory):**
        1. **Mechanistic Comparison**: Compare the mechanisms in the known pathway with what would be expected for the user's case. What fundamental chemistry remains the same? What changes?
        
        2. **Adaptation Strategy**: Based on the embedding distance and mechanistic understanding, explain how to adapt the known pathway: - If distance < 0.4: Minor parameter adjustments with same mechanism
           - If distance 0.4-0.7: Modified mechanism with similar principles  
           - If distance > 0.7: Fundamentally different mechanism required

        3. **Chain-of-Thought Prediction**: Provide step-by-step reasoning for your prediction, including:
           - Electronic structure considerations
           - Defect chemistry modifications
           - Thermodynamic feasibility
           - Kinetic pathway changes

        4. **Uncertainty Quantification**: Explicitly state which aspects are well-supported by analogy and which require extrapolation.

        **Output Format:**
        Provide your answer in a structured JSON format within a ```json block.

        Example JSON output:
        {{
          "{output_format_label}": {example_output},
          "mechanistic_reasoning": {{
            "similarity_analysis": "The known pathway involves p-doping via oxidation. The user seeks n-doping, requiring opposite charge carriers...",
            "adapted_mechanism": "Instead of oxygen creating acceptor states, we need donor states. Rhenium substitution can provide extra electrons...",
            "electronic_structure": "Re has one more valence electron than Mo, creating shallow donor states near the conduction band...",
            "thermodynamic_analysis": "Re-Mo substitution is energetically favorable with formation energy ~1.2 eV..."
          }},
          "chain_of_thought": [
            "Step 1: Identify that n-doping requires electron donors, not acceptors",
            "Step 2: Select Re as it has 7 valence electrons vs Mo's 6",
            "Step 3: Use CVD for controlled substitutional doping",
            "Step 4: Re atoms substitute Mo sites, donating electrons",
            "Step 5: Fermi level shifts toward conduction band, creating n-type behavior"
          ],
          "uncertainty_analysis": {{
            "high_confidence": "Re as n-type dopant is well-established",
            "medium_confidence": "Exact doping concentration achievable",
            "low_confidence": "Precise temperature optimization without experimental data"
          }},
          "confidence": {confidence:.4f},
          "property_embedding_distance": {property_embedding_diff:.4f},
          "analogous_path_used": "{analogous_context}"
        }}
        """

        print("\n" + "="*80)
        print("TRANSFER LEARNING PROMPT WITH MECHANISTIC REASONING:")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")

        print("Querying Gemini with mechanistic transfer learning prompt...")
        response = self.model.generate_content(prompt)
        return extract_json_from_response(response.text)

    def forward_prediction(self, synthesis_inputs: dict):
        """
        Enhanced forward prediction with mechanistic explanations.
        """
        print("\n--- Starting Forward Prediction ---")
        input_keywords = [str(v) for v in synthesis_inputs.values() if v is not None]
        query_string = " and ".join(input_keywords)
        
        all_properties = [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]
        
        causal_context_list = self._find_relevant_paths(input_keywords, all_properties)

        if causal_context_list:
            # NEW: Use enhanced direct path query instead of simple statement
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
        return {
            "error": "Failed to find a direct or analogous path in the knowledge graph.",
            "confidence": 0.0,
            "suggestion": "Consider expanding the knowledge graph or use a general-purpose query."
        }

    def inverse_design(self, desired_properties: dict):
        """
        Enhanced inverse design with mechanistic explanations.
        """
        print("\n--- Starting Inverse Design ---")
        property_keywords = [str(v) for v in desired_properties.values() if v is not None]
        query_string = " and ".join(property_keywords)

        all_synthesis_params = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]

        causal_context_list = self._find_relevant_paths(all_synthesis_params, property_keywords, reverse=True)
        
        if causal_context_list:
            # NEW: Use enhanced direct path query instead of simple statement
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
        # This should find a direct path in the graph and provide mechanistic explanation
        synthesis_params_exact = {
            "temperature": "200°C",
            "method": "Oxidation"
        }
        predicted_props = engine.forward_prediction(synthesis_params_exact)
        print("\nForward Prediction Result (Exact Match with Mechanistic Explanation):")
        print(json.dumps(predicted_props, indent=2))

        # --- Example 2: Forward Prediction (Analogous/Transfer Learning) ---
        synthesis_params_analogous = {
            "temperature": "210°C",
            "method": "Annealing in an oxygen atmosphere"
        }
        predicted_props_analogous = engine.forward_prediction(synthesis_params_analogous)
        print("\nForward Prediction Result (Analogous Match with Mechanistic Reasoning):")
        print(json.dumps(predicted_props_analogous, indent=2))

        # --- Example 3: Inverse Design (Exact Match) ---
        target_properties_exact = {
            "doping": "Controllable p-type doping",
        }
        suggested_synthesis = engine.inverse_design(target_properties_exact)
        print("\nInverse Design Result (Exact Match with Mechanistic Explanation):")
        print(json.dumps(suggested_synthesis, indent=2))
        
        # --- Example 4: Inverse Design (Analogous/Transfer Learning) ---
        target_properties_analogous = {
            "doping": "Achieve tunable hole-based conductivity",
        }
        suggested_synthesis_analogous = engine.inverse_design(target_properties_analogous)
        print("\nInverse Design Result (Analogous Match with Mechanistic Reasoning):")
        print(json.dumps(suggested_synthesis_analogous, indent=2))

    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error initializing or running engine: {e}")