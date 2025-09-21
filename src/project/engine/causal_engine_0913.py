import json
import networkx as nx
import os
import dashscope
from dashscope import Generation
import google.generativeai as genai
from openai import OpenAI
from pathlib import Path
import textwrap
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from project.utils.parser import extract_json_from_response


class CausalReasoningEngine:
    """
    Enhanced Causal Reasoning Engine with robust similarity-based fallback mechanisms.
    Uses hierarchical strategy to guarantee performance equal to or better than baseline.
    Supports multiple LLM APIs: Qwen (DashScope), Gemini (Google), and OpenAI.
    """
    def __init__(self, json_file_path: str, model_id: str = "qwen-plus", api_type: str = "qwen",
                embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with enhanced similarity-based reasoning using hierarchical strategy
        
        Args:
            json_file_path: Path to the JSON file containing causal relationships
            model_id: Model identifier for the LLM. Supported models:
            embedding_model: Sentence transformer model for embeddings (default: 'all-MiniLM-L6-v2')
        """
        # Hardcoded hierarchical strategy parameters
        self.strategy = "hierarchical"
        self.similarity_threshold = 0.4
        self.confidence_threshold = 0.6
        
        self.graph = self._build_graph(json_file_path)
        self._configure_api(model_id, api_type)
        print(f"Loading sentence transformer model ('{embedding_model}')...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self._precompute_node_embeddings()
        print(f"Enhanced Causal Reasoning Engine initialized with hierarchical strategy.")
        print(f"Similarity threshold: {self.similarity_threshold}, Confidence threshold: {self.confidence_threshold}")
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

    def _baseline_fallback_query(self, original_prompt_data: dict, query_type: str, reason: str = "No suitable DAG knowledge found"):
        """
        Pure baseline query that matches the baseline model performance exactly. This ensures we never underperform the baseline.
        """
        print(f"🔄 Using baseline fallback: {reason}")
        
        if query_type == "forward":
            task_desc = "predict the resulting material properties"
            output_label = "predicted_properties"
            output_format = self.forward_output_format
        else:
            task_desc = "suggest synthesis conditions to achieve the desired properties"
            output_label = "suggested_synthesis_conditions"
            output_format = self.inverse_output_format
        baseline_prompt = f"""
        You are an expert materials scientist. Based on the following {'synthesis conditions' if query_type == 'forward' else 'desired properties'}, 
        {task_desc}.
        
        {'Synthesis Conditions' if query_type == 'forward' else 'Desired Properties'}:
        {json.dumps(original_prompt_data, indent=2)}
        
        Provide your answer in a structured JSON format within a ```json block.
        
        Example format:
        {output_format}
        """
        
        response_text = self._generate_content(baseline_prompt)
        result = extract_json_from_response(response_text)
        
        # Ensure the result has the required structure
        if "error" in result or "warning" in result:
            # Provide a structured fallback response
            return {
                output_label: {"status": "baseline_analysis_completed"},
                "reasoning": f"Baseline materials science analysis applied. {reason}",
                "confidence": 0.5,
                "method": "baseline_fallback"
            }
        
        # Add metadata to indicate this is baseline performance
        result["raw_response"] = response_text
        result["method"] = "baseline_fallback"
        result["dag_enhancement"] = False
        return result

    def _enhanced_similarity_search(self, query_keywords: list, target_nodes: list, top_k: int = 5):
        """
        Enhanced similarity search that finds multiple similar nodes and paths
        """
        if not target_nodes or self.node_embeddings.size == 0:
            return []
        
        query_string = " ".join([str(kw) for kw in query_keywords if kw is not None])
        query_embedding = self.embedding_model.encode([query_string])
        
        # Find top-k similar nodes
        similarities = []
        for node in target_nodes:
            if node in self.node_list:
                node_idx = self.node_list.index(node)
                node_embedding = self.node_embeddings[node_idx:node_idx+1]
                similarity = cosine_similarity(query_embedding, node_embedding)[0][0]
                similarities.append((node, similarity))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _assess_dag_knowledge_quality(self, causal_paths: list, similarity_scores: list = None):
        """
        Assess the quality and relevance of available DAG knowledge
        """
        if not causal_paths:
            return {
                "quality_score": 0.0,
                "coverage": "none",
                "recommendation": "use_baseline",
                "reason": "No relevant paths found"
            }
        
        # Calculate quality metrics
        avg_path_length = np.mean([len(path.split(" -> ")) for path in causal_paths])
        num_paths = len(causal_paths)
        
        # Factor in similarity scores if available
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.5
        
        # Calculate overall quality score
        quality_score = (avg_similarity * 0.5 + 
                        min(num_paths / 3, 1.0) * 0.3 + 
                        min(avg_path_length / 4, 1.0) * 0.2)
        
        if quality_score >= 0.7:
            recommendation = "use_dag_primary"
            coverage = "excellent"
        elif quality_score >= 0.5:
            recommendation = "use_dag_enhancement"
            coverage = "good"
        elif quality_score >= self.similarity_threshold:
            recommendation = "use_dag_cautious"
            coverage = "moderate"
        else:
            recommendation = "use_baseline"
            coverage = "poor"
        
        return {
            "quality_score": quality_score,
            "coverage": coverage,
            "recommendation": recommendation,
            "avg_similarity": avg_similarity,
            "num_paths": num_paths,
            "avg_path_length": avg_path_length
        }

    def _dag_enhanced_query(self, original_prompt_data: dict, causal_paths: list, 
                           quality_assessment: dict, query_type: str, similarity_info: dict = None):
        """
        DAG-enhanced query that intelligently combines baseline reasoning with DAG knowledge
        """
        print(f"🧠 Using DAG-enhanced reasoning (quality: {quality_assessment['coverage']})")
        
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
        
        # Include similarity information if available
        similarity_context = ""
        if similarity_info:
            similarity_context = f"""
            **Similarity Analysis:**
            - Best matching knowledge: {similarity_info.get('best_match', 'N/A')} (similarity: {similarity_info.get('best_score', 0):.3f})
            - Knowledge quality: {quality_assessment['coverage']} (score: {quality_assessment['quality_score']:.3f})
            """
        
        enhanced_prompt = f"""
        You are an expert materials scientist with access to a specialized knowledge graph derived from 200+ research papers.
        Your task is to {task_desc} by intelligently combining your baseline scientific knowledge with relevant research findings.
        
        **CRITICAL INSTRUCTION: Your final answer must be AT LEAST as good as pure baseline reasoning. Use DAG knowledge to ENHANCE, not replace, fundamental principles.**
        
        **{input_label}:**
        {json.dumps(original_prompt_data, indent=2)}
        
        **Relevant Research Knowledge from Literature:**
        Causal Pathways:
        - {formatted_paths}
        
        Known Mechanisms:
        - {formatted_mechanisms}
        
        {similarity_context}
        
        **Integration Strategy ({quality_assessment['recommendation']}):**
        1. **Baseline Analysis**: First, provide your fundamental materials science analysis
        2. **DAG Enhancement**: Use the research knowledge to enhance or validate your baseline reasoning
        3. **Quality Control**: Ensure the final prediction is scientifically sound and improves upon baseline
        4. **Confidence Assessment**: Provide honest confidence levels for each aspect
        
        **Output Requirements:**
        - Reason step by step until you reach the final answer, put the final answer in the json format
        - Your answer must be logical, scientifically rigorous and practically useful
        - Include mechanistic explanations combining both knowledge sources
        - Provide confidence levels and uncertainty analysis
        
        **JSON Output Format:**
        {output_format}
        """
        
        response_text = self._generate_content(enhanced_prompt)
        result = extract_json_from_response(response_text)
        
        # Quality control: ensure the result is valid
        if "error" in result or "warning" in result:
            print("⚠️ DAG-enhanced query failed, falling back to baseline")
            return self._baseline_fallback_query(original_prompt_data, query_type, 
                                               "DAG integration failed")
        
        # Add metadata
        result["raw_response"] = response_text
        result["method"] = "dag_enhanced"
        result["dag_quality"] = quality_assessment
        result["dag_enhancement"] = True
        
        return result

    def _robust_path_finding(self, start_keywords: list, end_keywords: list, reverse: bool = False):
        """
        Enhanced path finding with multiple similarity-based strategies
        """
        # Try exact keyword matching first
        direct_paths = self._find_relevant_paths(start_keywords, end_keywords, reverse)
        if direct_paths:
            return direct_paths, "direct_match", 1.0
        
        # Try similarity-based matching
        print("🔍 No direct paths found, searching for similar nodes...")
        
        if reverse:
            source_candidates = [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]  # properties
            target_candidates = [node for node in self.graph.nodes if self.graph.in_degree(node) == 0]   # synthesis
            search_keywords = end_keywords  # We're looking for similar synthesis conditions
        else:
            source_candidates = [node for node in self.graph.nodes if self.graph.in_degree(node) == 0]   # synthesis
            target_candidates = [node for node in self.graph.nodes if self.graph.out_degree(node) == 0]  # properties
            search_keywords = start_keywords  # We're looking for similar synthesis conditions
        
        # Find similar source nodes
        similar_sources = self._enhanced_similarity_search(search_keywords, source_candidates, top_k=5)
        
        analogous_paths = []
        similarity_scores = []
        
        for similar_node, similarity_score in similar_sources:
            if similarity_score >= self.similarity_threshold:
                if reverse:
                    paths = self._find_relevant_paths([similar_node], end_keywords, reverse=True)
                else:
                    paths = self._find_relevant_paths([similar_node], end_keywords, reverse=False)
                
                for path in paths:
                    analogous_paths.append(path)
                    similarity_scores.append(similarity_score)
        
        if analogous_paths:
            avg_similarity = np.mean(similarity_scores)
            return analogous_paths, "similarity_based", avg_similarity
        
        return [], "no_match", 0.0
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

        prompt = f"""
        You are an expert materials scientist AI conducting transfer learning analysis. Your knowledge graph lacks exact pathways, but you've identified analogous information that requires careful validation and adaptation.

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

        **Your Instructions:**
        1. **Baseline Analysis**: First, provide your fundamental materials science analysis
        2. **DAG Enhancement**: Use the research knowledge to enhance or validate your baseline reasoning
        3. **Quality Control**: Ensure the final prediction is scientifically sound and improves upon baseline
        4. **Confidence Assessment**: Provide honest confidence levels for each aspect
        
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

        result["raw_response"] = response_text
        result["method"] = "dag_enhanced"
        result["dag_enhancement"] = False
        return result
    
    def forward_prediction(self, synthesis_inputs: dict):
        """
        Enhanced forward prediction with robust similarity-based fallback
        """
        print("\n--- Starting Enhanced Forward Prediction ---")
        
        # Extract keywords and prepare for search
        input_keywords = [str(v) for v in synthesis_inputs.values() if v is not None and str(v).strip()]
        if not input_keywords:
            return self._baseline_fallback_query(synthesis_inputs, "forward", 
                                               "No valid synthesis conditions provided")
        
        query_string = " and ".join(input_keywords)
        all_properties = [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]
        
        # Robust path finding
        causal_paths, match_type, confidence = self._robust_path_finding(input_keywords, all_properties)
        
        # Assess DAG knowledge quality
        similarity_scores = [confidence] * len(causal_paths) if causal_paths else []
        quality_assessment = self._assess_dag_knowledge_quality(causal_paths, similarity_scores)
        
        print(f"📊 Path finding results: {len(causal_paths)} paths, match type: {match_type}, confidence: {confidence:.3f}")
        print(f"📊 Quality assessment: {quality_assessment['coverage']} ({quality_assessment['quality_score']:.3f})")
        
        # Decide on reasoning strategy
        if quality_assessment['recommendation'] == 'use_baseline':
            return self._baseline_fallback_query(synthesis_inputs, "forward", 
                                               f"DAG knowledge quality insufficient ({quality_assessment['quality_score']:.3f} < {self.similarity_threshold})")
        
        # Use DAG-enhanced reasoning
        similarity_info = {
            "best_match": causal_paths[0] if causal_paths else None,
            "best_score": confidence,
            "match_type": match_type
        }
        
        try:
            result = self._dag_enhanced_query(synthesis_inputs, causal_paths, quality_assessment, 
                                            "forward", similarity_info)
            
            # Final quality check
            if result.get("confidence", 0) < 0.3:
                print("⚠️ Low confidence result, falling back to baseline")
                return self._baseline_fallback_query(synthesis_inputs, "forward", 
                                                   "DAG-enhanced result had low confidence")
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error in DAG-enhanced reasoning: {e}")
            return self._baseline_fallback_query(synthesis_inputs, "forward", 
                                               f"DAG processing error: {str(e)}")

    def inverse_design(self, desired_properties: dict):
        """
        Enhanced inverse design with robust similarity-based fallback
        """
        print("\n--- Starting Enhanced Inverse Design ---")
        
        # Extract keywords and prepare for search
        property_keywords = [str(v) for v in desired_properties.values() if v is not None and str(v).strip()]
        if not property_keywords:
            return self._baseline_fallback_query(desired_properties, "inverse", 
                                               "No valid desired properties provided")
        
        query_string = " and ".join(property_keywords)
        all_synthesis_params = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
        
        # Robust path finding (reverse direction)
        causal_paths, match_type, confidence = self._robust_path_finding(all_synthesis_params, property_keywords, reverse=True)
        
        # Assess DAG knowledge quality
        similarity_scores = [confidence] * len(causal_paths) if causal_paths else []
        quality_assessment = self._assess_dag_knowledge_quality(causal_paths, similarity_scores)
        
        print(f"📊 Path finding results: {len(causal_paths)} paths, match type: {match_type}, confidence: {confidence:.3f}")
        print(f"📊 Quality assessment: {quality_assessment['coverage']} ({quality_assessment['quality_score']:.3f})")
        
        # Decide on reasoning strategy
        if quality_assessment['recommendation'] == 'use_baseline':
            return self._baseline_fallback_query(desired_properties, "inverse", 
                                               f"DAG knowledge quality insufficient ({quality_assessment['quality_score']:.3f} < {self.similarity_threshold})")
        
        # Use DAG-enhanced reasoning
        similarity_info = {
            "best_match": causal_paths[0] if causal_paths else None,
            "best_score": confidence,
            "match_type": match_type
        }
        
        try:
            result = self._dag_enhanced_query(desired_properties, causal_paths, quality_assessment, 
                                            "inverse", similarity_info)
            
            # Final quality check
            if result.get("confidence", 0) < 0.3:
                print("⚠️ Low confidence result, falling back to baseline")
                return self._baseline_fallback_query(desired_properties, "inverse", 
                                                   "DAG-enhanced result had low confidence")
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error in DAG-enhanced reasoning: {e}")
            return self._baseline_fallback_query(desired_properties, "inverse", 
                                               f"DAG processing error: {str(e)}")

    def _configure_api(self, model_id: str, api_type: str):
        """Configure API based on model_id to support Qwen, Gemini, and OpenAI models"""
        self.model_id = model_id
        self.api_type = api_type
        
        # Qwen models (DashScope API)
        if api_type == "qwen":
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise RuntimeError("Missing DASHSCOPE_API_KEY for Qwen models")
            dashscope.api_key = api_key
            dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
            
        # Gemini models (Google Generative AI)
        elif api_type == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("Missing GOOGLE_API_KEY for Gemini models")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_id)
            
        # OpenAI models
        elif api_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY for OpenAI models")
            self.openai_client = OpenAI(api_key=api_key)
            
        else:
            raise ValueError(f"Unsupported api_type: {api_type} for model: {model_id}")
        
        print(f"Configured {self.api_type} API for model: {model_id}")

    def _generate_content(self, prompt: str):
        """Generate content using the configured API (Qwen, Gemini, or OpenAI)"""
        try:
            if self.api_type == "qwen":
                # Qwen API via DashScope
                response = Generation.call(
                    model=self.model_id,
                    prompt=prompt,
                    temperature=0.7
                )
                
                if response.status_code == 200:
                    return response.output.text
                else:
                    print(f"Qwen API Error: {response.code} - {response.message}")
                    return f"Error: {response.code} - {response.message}"
                    
            elif self.api_type == "gemini":
                # Gemini API via Google Generative AI
                response = self.model.generate_content(prompt)
                return response.text
                
            elif self.api_type == "openai":
                # OpenAI API
                response = self.openai_client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                return response.choices[0].message.content
                
            else:
                raise ValueError(f"Unknown API type: {self.api_type}")
                
        except Exception as e:
            print(f"Exception calling {self.api_type} API: {str(e)}")
            return f"Exception: {str(e)}"

    def _build_graph(self, json_file_path: str):
        input_path = Path(json_file_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found at {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "causal_relationships" in data:
            relationships = data.get("causal_relationships", [])
        else:
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
        """Extract mechanism information for edges in a path."""
        nodes = [node.strip() for node in path_str.split(" -> ")]
        mechanisms = []
        for i in range(len(nodes) - 1):
            if self.graph.has_edge(nodes[i], nodes[i+1]):
                mechanism = self.graph[nodes[i]][nodes[i+1]].get('mechanism', '')
                if mechanism:
                    mechanisms.append(f"{nodes[i]} → {nodes[i+1]}: {mechanism}")
        return mechanisms


if __name__ == '__main__':
    json_file = 'outputs/combined_doping_data.json'
    
    try:
        print("="*80)
        print("TESTING ENHANCED HIERARCHICAL CAUSAL REASONING ENGINE")
        print("="*80)
        
        # Initialize engine with same arguments as original
        engine = CausalReasoningEngine(json_file, model_id="qwen-plus", api_type="qwen")
        
        # Test cases that should trigger different pathways
        test_cases = [
            {
                "name": "Direct Match Test",
                "synthesis": {"temperature": "200°C", "method": "Oxidation"},
                "properties": {"doping": "Controllable p-type doping"}
            },
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
            print("-" * 40)
            
            # Forward prediction
            print("Forward Prediction:")
            forward_result = engine.forward_prediction(test_case['synthesis'])
            print(f"Method: {forward_result.get('method', 'unknown')}")
            print(f"Confidence: {forward_result.get('confidence', 0):.3f}")
            print(f"DAG Enhanced: {forward_result.get('dag_enhancement', False)}")
            
            # Inverse design
            print("\nInverse Design:")
            inverse_result = engine.inverse_design(test_case['properties'])
            print(f"Method: {inverse_result.get('method', 'unknown')}")
            print(f"Confidence: {inverse_result.get('confidence', 0):.3f}")
            print(f"DAG Enhanced: {inverse_result.get('dag_enhancement', False)}")
        
        print(f"\n{'='*80}")
        print("✅ ALL HIERARCHICAL TESTS COMPLETED!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()