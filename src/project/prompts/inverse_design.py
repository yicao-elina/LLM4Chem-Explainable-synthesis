import json
import re
from typing import Dict, Any, List
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai import ChatOpenAI
from langchain_interface.models import ChatOpenAIWithBatchAPI, BatchedAPIConfig

class CausalReasoningJSONParser(BaseOutputParser[Dict[str, Any]]):
    """
    Parses a JSON object from a string that contains a JSON markdown block.
    This parser is robust against malformed JSON from the LLM.
    """

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Extracts and parses the JSON object from the LLM's response text.

        Args:
            text: The raw text output from the language model.

        Returns:
            A dictionary parsed from the JSON string.

        Raises:
            OutputParserException: If the JSON cannot be parsed.
        """
        # Use regex to find the JSON content within ```json ... ```
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)
        
        if not match:
            raise OutputParserException(f"No JSON markdown block found in the output: {text}")
        
        json_string = match.group(1).strip()
        
        try:
            # Attempt to parse the extracted string as JSON
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            # If parsing fails, raise an exception with detailed information
            raise OutputParserException(
                f"Failed to parse JSON. Error: {e}. Malformed JSON string: \n{json_string}"
            )

    @property
    def _type(self) -> str:
        return "causal_reasoning_json_parser"
# Input variables:
# - desired_properties: A JSON string of the user's target properties.
# - causal_paths: A string listing the relevant paths found in the graph (in reverse).
# - mechanisms: A string listing the known mechanisms for those paths.

inverse_direct_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert materials scientist AI."),
        ("human", """
You are an expert materials scientist AI. Your knowledge graph contains direct causal pathways relevant to the user's query.
Your task is to explain the mechanistic reasoning for selecting synthesis conditions to achieve desired properties using both the specific knowledge from the graph AND your general chemical understanding.

**Desired Material Properties (User's Query):**
{desired_properties}

**Direct Causal Pathways from Knowledge Graph:**
- {causal_paths}

**Known Mechanisms from Knowledge Graph:**
- {mechanisms}

**Your Task:**
1. **Mechanistic Analysis**: Explain the chemical/physical mechanisms underlying each step in the causal pathway that leads to the desired properties.
2. **Chain-of-Thought Reasoning**: Provide a step-by-step logical explanation of how the suggested conditions will produce the desired effects.
3. **Quantitative Insights**: Where possible, suggest specific quantitative parameters (e.g., temperature ranges, durations).
4. **Alternative Pathways**: Briefly mention any alternative methods that could achieve similar outcomes.

**Output Format:**
Provide your answer in a structured JSON format within a ```json block.
The JSON MUST include detailed mechanistic reasoning and chain-of-thought analysis.

Example JSON output:
```json
{{
  "suggested_synthesis_conditions": {{"method": "Surface oxidation", "temperature_c": 200, "duration_hours": 2}},
  "mechanistic_explanation": {{
    "primary_mechanism": "To achieve p-type doping, surface oxidation is an effective method as it creates oxygen vacancies that act as acceptors...",
    "electronic_effects": "These acceptors create holes in the valence band, shifting the Fermi level down...",
    "defect_chemistry": "Controlling temperature and duration manages the density of these acceptor states...",
    "thermodynamics": "At 200°C, the Gibbs free energy favors oxygen chemisorption without bulk degradation..."
  }},
  "chain_of_thought": [
    "Step 1: The goal is p-type doping.",
    "Step 2: The graph indicates oxidation leads to p-type behavior.",
    "Step 3: The mechanism involves creating acceptor states via oxygen on the surface.",
    "Step 4: A moderate temperature of 200°C is effective for this process."
  ],
  "quantitative_estimates": {{
    "temperature_range": "150-250°C",
    "expected_hole_concentration": "~10^12 to 10^13 cm^-2"
  }},
  "alternative_mechanisms": [
    "Substitutional doping with group V elements could also work but is more complex."
  ],
  "confidence": 1.0,
  "reasoning": "Direct causal pathway found in knowledge graph, enhanced with mechanistic understanding."
}}
```
""")]   
)


# Input variables:
# - desired_properties: A JSON string of the user's target properties.
# - analogous_context: The most similar known causal pathway from the graph.
# - mechanisms: Known mechanisms for the analogous pathway.
# - confidence: A float (0.0-1.0) indicating similarity.
# - property_embedding_distance: A float indicating the semantic distance between the desired and known properties.

inverse_transfer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert materials scientist AI."),
        ("human", """
You are an expert materials scientist AI. Your task is to reason from analogous data using mechanistic understanding.
Your knowledge graph does not contain an exact causal pathway for the user's query, but you have identified similar information.

**Task:**
Based on the provided analogous information and your chemical knowledge, suggest synthesis conditions with mechanistic justification to achieve the user's target properties.

**Target Material Properties (User's Query):**
{desired_properties}

**Most Similar Known Causal Pathway:**
- {analogous_context}

**Known Mechanisms for Similar Pathway:**
- {mechanisms}

**Quantitative Analysis:**
The embedding distance between the user's query and the most similar known case is {property_embedding_distance:.4f} (where 0 is identical and 2 is opposite).

**Your Reasoning Process (Mandatory):**
1. **Mechanistic Comparison**: Compare the mechanisms in the known pathway with what would be needed for the user's case.
2. **Adaptation Strategy**: Based on the embedding distance and mechanism, explain how to adapt the known synthesis conditions.
   - If distance < 0.4: Minor parameter adjustments.
   - If distance 0.4-0.7: Modified mechanism with similar principles.
   - If distance > 0.7: Fundamentally different mechanism likely required.
3. **Chain-of-Thought Prediction**: Provide step-by-step reasoning for your suggested synthesis route.
4. **Uncertainty Quantification**: Explicitly state which aspects are well-supported by analogy and which require extrapolation.

**Output Format:**
Provide your answer in a structured JSON format within a ```json block.

Example JSON output:
```json
{{
  "suggested_synthesis_conditions": {{"method": "Chemical Vapor Deposition", "dopant_precursor": "ReO3"}},
  "mechanistic_reasoning": {{
    "similarity_analysis": "The known pathway achieves p-doping via oxidation. The user seeks tunable hole conductivity, which is related but requires precise control...",
    "adapted_mechanism": "Instead of surface oxidation, which can be hard to control, substitutional doping with Re during CVD offers better tunability..."
  }},
  "chain_of_thought": [
    "Step 1: Identify that 'tunable hole conductivity' requires precise control of p-type dopants.",
    "Step 2: The analogous path (oxidation) is less controllable.",
    "Step 3: A better method is substitutional doping during synthesis, like CVD.",
    "Step 4: Suggest CVD with a suitable p-type dopant precursor."
  ],
  "uncertainty_analysis": {{
    "high_confidence": "CVD is a suitable method for controlled doping.",
    "medium_confidence": "The exact precursor flow rates will need experimental optimization."
  }},
  "confidence": {confidence},
  "property_embedding_distance": {property_embedding_distance},
  "analogous_path_used": "{analogous_context}"
}}
```
""")]
)

class InverseDirectChain:
    def __init__(self):
        set_llm_cache(SQLiteCache("checkpoints/._cache.db"))
        self._runnable_config = BatchedAPIConfig(
            max_batch_size=3500
        )
        self._llm = ChatOpenAIWithBatchAPI(
            base_url= "http://localhost:8000/v1",
            api_key = "EMPTY",
            temperature=0,
            model="Qwen/Qwen3-14B",
            max_tokens=None,
            verbose=True,
            top_p=0.98,
        )
        self._inverse_direct_prompt = inverse_direct_prompt
    
    def get_result(self, datas: List[str]) -> List[Dict[str, Any]]:
        """Get all reasoning texts and summarize them."""
        calling_chain = self._inverse_direct_prompt | self._llm | CausalReasoningJSONParser()
        if isinstance(datas, dict):
            inputs = [{"desired_properties": datas["desired_properties"], "causal_paths": datas["causal_paths"], "mechanisms": datas["mechanisms"]}]
            length = 1
        else:
            inputs = [{"desired_properties": data["desired_properties"], "causal_paths": data["causal_paths"], "mechanisms": data["mechanisms"]} for data in datas]
            length = len(datas)
        results = []
        for result in tqdm(calling_chain.batch(inputs=inputs,
            config=self._runnable_config), total=length, desc="Summarizing reasoning"):
            results.append(result)
        return results 

class InverseTransferChain:
    def __init__(self):
        set_llm_cache(SQLiteCache("checkpoints/._cache.db"))
        self._runnable_config = BatchedAPIConfig(
            max_batch_size=3500
        )
        self._llm = ChatOpenAIWithBatchAPI(
            base_url= "http://localhost:8000/v1",
            api_key = "EMPTY",
            temperature=0,
            model="Qwen/Qwen3-14B",
            max_tokens=None,
            verbose=True,
            top_p=0.98,
        )
        self._inverse_transfer_prompt = inverse_transfer_prompt
    def get_result(self, datas: List[str]) -> List[Dict[str, Any]]:
        """Get all reasoning texts and summarize them."""
        calling_chain = self._inverse_transfer_prompt | self._llm | CausalReasoningJSONParser()
        
        if isinstance(datas, dict):
            inputs = [{"desired_properties": datas["desired_properties"], "analogous_context": datas["analogous_context"], "mechanisms": datas["mechanisms"], "confidence": datas["confidence"], "property_embedding_distance": datas["property_embedding_distance"]}]
            length = 1  
        else:
            inputs = [{"desired_properties": data["desired_properties"], "analogous_context": data["analogous_context"], "mechanisms": data["mechanisms"], "confidence": data["confidence"], "property_embedding_distance": data["property_embedding_distance"]} for data in datas]
            length = len(datas)
        results = []
        for result in tqdm(calling_chain.batch(inputs=inputs,
            config=self._runnable_config), total=length, desc="Summarizing reasoning"):
            results.append(result)
        return results 