import json
import networkx as nx
import os
from pathlib import Path
import textwrap
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import dashscope
from dashscope import Generation
import google.generativeai as genai
from openai import OpenAI

from project.utils.parser import extract_json_from_response

class CausalReasoningEngine:
    
    def __init__(self, json_file_path: str, model_id: str = "gemini-1.5-pro-latest", api_type: str = "gemini"):
        self._configure_api(model_id, api_type)
        
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

    def forward_prediction(self, synthesis_inputs: dict):

        # --- FIX: Convert all values to strings and filter out None before joining ---
        input_keywords = [str(v) for v in synthesis_inputs.values() if v is not None]
        query_string = " and ".join(input_keywords)
        
        prompt = f"""
        You are an expert materials scientist AI. Based on the following synthesis conditions predict the resulting material properties.

        **Synthesis Conditions:**
        {query_string}

        **Task:**
        Predict the most likely properties. Directly provide your answer in a structured JSON format within a ```json block.
        
        Example JSON output:
        ```json
        {{
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
            "reasoning": "...",
            "confidence": a float between 0 and 1
        }}
        ```
        """
        response = self._generate_content(prompt)
        return extract_json_from_response(response)

    def inverse_design(self, desired_properties: dict):
        print("\n--- Starting Inverse Design ---")
        # --- FIX: Convert all values to strings and filter out None before joining ---
        property_keywords = [str(v) for v in desired_properties.values() if v is not None]
        query_string = " and ".join(property_keywords)

        
        prompt = f"""
        You are an expert materials scientist AI. Your task is to design a synthesis protocol to achieve specific material properties.

        **Desired Material Properties:**
        {query_string}

        **Task:**
        Predict the most likely synthesis conditions to achieve the desired properties. Directly provide your answer in a structured JSON format within a ```json block.

        Example output format:
        ```json
        {{
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
            "reasoning": "...",
            "confidence": a float between 0 and 1
        }}
        ```
        """
        response = self._generate_content(prompt)
        return extract_json_from_response(response)
        
     

if __name__ == '__main__':
    # Ensure the JSON file and API key are correctly set up
    json_file = 'datas/combined_doping_data.json' # Make sure this file exists
    
    engine = CausalReasoningEngine(json_file)

    # --- Example 1: Forward Prediction (Exact Match) ---
    # This should find a direct path in the graph.
    synthesis_params_exact = {
        "temperature": "200°C",
        "method": "Oxidation"
    }
    predicted_props = engine.forward_prediction(synthesis_params_exact)

    print(json.dumps(predicted_props, indent=2))

    # --- Example 2: Forward Prediction (Analogous/Transfer Learning) ---
    # This query is semantically similar but not identical to the one above.
    # It should trigger the transfer learning fallback.
    # synthesis_params_analogous = {
    #     "temperature": "210°C",
    #     "method": "Annealing in an oxygen atmosphere"
    # }
    # predicted_props_analogous = engine.forward_prediction(synthesis_params_analogous)
    # print("\nForward Prediction Result (Analogous Match):")
    # print(json.dumps(predicted_props_analogous, indent=2))

    # --- Example 3: Inverse Design (Exact Match) ---
    target_properties_exact = {
        "doping": "Controllable p-type doping",
    }
    suggested_synthesis = engine.inverse_design(target_properties_exact)
    print("\nInverse Design Result (Exact Match):")
    print(json.dumps(suggested_synthesis, indent=2))
    
    # --- Example 4: Inverse Design (Analogous/Transfer Learning) ---
    # This query is semantically similar to a property in the graph but uses different wording.
    # target_properties_analogous = {
    #     "doping": "Achieve tunable hole-based conductivity",
    # }
    # suggested_synthesis_analogous = engine.inverse_design(target_properties_analogous)
    # print("\nInverse Design Result (Analogous Match):")
    # print(json.dumps(suggested_synthesis_analogous, indent=2))
