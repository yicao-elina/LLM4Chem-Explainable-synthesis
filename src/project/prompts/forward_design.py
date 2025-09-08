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
# - synthesis_conditions: A JSON string of the user's input conditions.
# - causal_paths: A string listing the direct paths found in the graph.
# - mechanisms: A string listing the known mechanisms for those paths.

forward_direct_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert materials scientist AI."),
        ("human", """
You are an expert materials scientist AI. Your knowledge graph contains direct causal pathways relevant to the user's query.
Your task is to explain the mechanistic pathway from synthesis conditions to material properties using both the specific knowledge from the graph AND your general chemical understanding.

**Synthesis Conditions (User's Query):**
{synthesis_conditions}

**Direct Causal Pathways from Knowledge Graph:**
- {causal_paths}

**Known Mechanisms from Knowledge Graph:**
- {mechanisms}

**Your Task:**
1. **Mechanistic Analysis**: Explain the chemical/physical mechanisms underlying each step in the causal pathway.
2. **Chain-of-Thought Reasoning**: Provide a step-by-step logical explanation of how each cause leads to its effect.
3. **Quantitative Insights**: Where possible, provide quantitative estimates or ranges.
4. **Alternative Pathways**: Briefly mention any alternative mechanisms that could lead to similar outcomes.

**Output Format:**
Provide your answer in a structured JSON format within a ```json block.
The JSON MUST include detailed mechanistic reasoning and chain-of-thought analysis.

Example JSON output:
{{
  "predicted_properties": {{"carrier_type": "p-type", "band_gap_ev": 1.5, "carrier_concentration": "increased"}},
  "mechanistic_explanation": {{
    "primary_mechanism": "Surface oxidation creates oxygen vacancies that act as p-type dopants...",
    "electronic_effects": "The removal of electrons creates holes in the valence band...",
    "defect_chemistry": "O2 molecules adsorb on the surface and extract electrons...",
    "thermodynamics": "At 200°C, the Gibbs free energy favors oxygen chemisorption..."
  }},
  "chain_of_thought": [
    "Step 1: At 200°C, oxygen molecules adsorb on the MoS2 surface",
    "Step 2: Oxygen abstracts electrons from Mo d-orbitals, creating Mo5+ states",
    "Step 3: These oxidized states act as acceptors, creating holes",
    "Step 4: The Fermi level shifts toward the valence band, establishing p-type behavior"
  ],
  "quantitative_estimates": {{
    "hole_concentration": "~10^12 to 10^13 cm^-2 for monolayer",
    "fermi_level_shift": "~0.1-0.2 eV toward valence band"
  }},
  "alternative_mechanisms": [
    "Substitutional doping with group V elements could also achieve p-type behavior"
  ],
  "confidence": 1.0,
  "reasoning": "Direct causal pathway found in knowledge graph, enhanced with mechanistic understanding."
}}
"""
)]
)


forward_transfer_prompt = ChatPromptTemplate.from_messages(
    [
    ("system", "You are an expert materials scientist AI."),
    ("human", """
You are an expert materials scientist AI. Your task is to reason from analogous data using mechanistic understanding.
Your knowledge graph does not contain an exact causal pathway for the user's query, but you have identified similar information.

**Task:**
Based on the provided analogous information and your chemical knowledge, predict the resulting material properties with mechanistic explanation for the user's target.

**Target Synthesis Conditions (User's Query):**
{synthesis_conditions}

**Most Similar Known Causal Pathway:**
- {analogous_context}

**Known Mechanisms for Similar Pathway:**
- {mechanisms}

**Your Reasoning Process (Mandatory):**
1. **Mechanistic Comparison**: Compare the mechanisms in the known pathway with what would be expected for the user's case. What fundamental chemistry remains the same? What changes?
2. **Adaptation Strategy**: Explain how to adapt the known pathway to predict the outcome for the user's query.
3. **Chain-of-Thought Prediction**: Provide step-by-step reasoning for your prediction.
4. **Uncertainty Quantification**: Explicitly state which aspects are well-supported by analogy and which require extrapolation.

**Output Format:**
Provide your answer in a structured JSON format within a ```json block.

Example JSON output:
{{
  "predicted_properties": {{"carrier_type": "p-type", "band_gap_ev": 1.5}},
  "mechanistic_reasoning": {{
    "similarity_analysis": "The known pathway involves oxidation at 200C. The user's query at 210C is very similar...",
    "adapted_mechanism": "The higher temperature will likely accelerate the oxidation kinetics, leading to a higher defect concentration..."
  }},
  "chain_of_thought": [
    "Step 1: The user's condition is annealing in oxygen, which is a form of oxidation.",
    "Step 2: The temperature is slightly higher, suggesting a faster reaction rate.",
    "Step 3: Faster oxidation leads to more p-type dopants.",
    "Step 4: Therefore, a higher hole concentration is expected compared to the 200C case."
  ],
  "uncertainty_analysis": {{
    "high_confidence": "The material will become p-type.",
    "medium_confidence": "The carrier concentration will increase."
  }},
  "confidence": {confidence},
  "analogous_path_used": "{analogous_context}"
}}
""")]
)


class ForwardDirectChain:
    def __init__(self):
        set_llm_cache(SQLiteCache("checkpoints/._cache.db"))
        self._runnable_config = BatchedAPIConfig(
            max_batch_size=3500
        )
        self._llm = ChatOpenAIWithBatchAPI(
            base_url= "http://localhost:8000/v1",
            api_key = "EMPTY",
            temperature=0,
            model="Qwen/Qwen3-8B",
            max_tokens=None,
            verbose=True,
            top_p=0.98,
        )
        self._forward_direct_prompt = forward_direct_prompt

    def get_result(self, datas: List[str]) -> List[Dict[str, Any]]:
        """Get all reasoning texts and summarize them."""
        calling_chain = self._forward_direct_prompt | self._llm | CausalReasoningJSONParser()
        if isinstance(datas, dict):
            inputs = [{"synthesis_conditions": datas["synthesis_conditions"], "causal_paths": datas["causal_paths"], "mechanisms": datas["mechanisms"]}]
        else:
            inputs = [{"synthesis_conditions": data["synthesis_conditions"], "causal_paths": data["causal_paths"], "mechanisms": data["mechanisms"]} for data in datas]
        results = []
        for result in tqdm(calling_chain.batch(inputs=inputs,
            config=self._runnable_config), total=len(datas), desc="Summarizing reasoning"):
            results.append(result)
        return results 

class ForwardTransferChain:
    def __init__(self):
        set_llm_cache(SQLiteCache("checkpoints/._cache.db"))
        self._runnable_config = BatchedAPIConfig(
            max_batch_size=3500
        )
        self._llm = ChatOpenAIWithBatchAPI(
            base_url= "http://localhost:8000/v1",
            api_key = "EMPTY",
            temperature=0,
            model="Qwen/Qwen3-8B",
            max_tokens=None,
            verbose=True,
            top_p=0.98,
        )
        self._forward_transfer_prompt = forward_transfer_prompt

    def get_result(self, datas: List[str]) -> List[Dict[str, Any]]:
        """Get all reasoning texts and summarize them."""
        calling_chain = self._forward_transfer_prompt | self._llm | CausalReasoningJSONParser()
        if isinstance(datas, dict):
            inputs = [{"synthesis_conditions": datas["synthesis_conditions"], "analogous_context": datas["analogous_context"], "mechanisms": datas["mechanisms"], "confidence": datas["confidence"]}]
        else:   
            inputs = [{"synthesis_conditions": data["synthesis_conditions"], "analogous_context": data["analogous_context"], "mechanisms": data["mechanisms"], "confidence": data["confidence"]} for data in datas]
        results = []
        for result in tqdm(calling_chain.batch(inputs=inputs,
            config=self._runnable_config), total=len(datas), desc="Summarizing reasoning"):
            results.append(result)
        return results 