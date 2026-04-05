"""
Baseline LLM Implementation (Ollama-based)

This variant represents pure LLM reasoning without any knowledge graph:
- No KG integration
- No tier reasoning
- No online search
- Just fundamental materials science knowledge from LLM pre-training
- Tests baseline LLM capability (Tier 3 only)

Author: ARIA Team
Date: 2026-02-01
"""

import json
from pathlib import Path
from typing import Dict, Any
import logging

from ollama_client import get_ollama_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineOllama:
    """
    Baseline LLM variant for ablation study.

    Key characteristics:
    - Pure LLM reasoning (no external knowledge)
    - No knowledge graph constraints
    - No online search
    - Tier 3 reasoning only (parametric fallback)
    - Control condition for measuring KG/search contributions
    """

    def __init__(self, model: str = "qwen2:7b"):
        """
        Initialize Baseline LLM system.

        Args:
            model: Ollama model to use
        """
        self.model = model
        self.ollama = get_ollama_client(model=model)

        logger.info(f"Baseline initialized with model={model} (pure LLM, no KG)")

    def forward_prediction(self, synthesis_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict material properties from synthesis conditions using pure LLM.

        Args:
            synthesis_inputs: Dict with synthesis parameters

        Returns:
            Prediction dict with properties and confidence
        """
        logger.info("Forward prediction (Baseline LLM only)")

        # Convert synthesis inputs to readable format
        input_keywords = [str(v) for v in synthesis_inputs.values() if v is not None]
        query_string = " and ".join(input_keywords)

        prompt = f"""You are an expert materials scientist AI. Based on the following synthesis conditions, predict the resulting material properties.

**Synthesis Conditions:**
{query_string}

**Full Input:**
{json.dumps(synthesis_inputs, indent=2)}

**Task:**
Predict the most likely properties using your knowledge of materials science fundamentals. Directly provide your answer in a structured JSON format.

Respond with ONLY valid JSON in this exact format:
{{
  "predicted_properties": {{
    "carrier_type": "n-type or p-type or null",
    "carrier_concentration": "value with units or null",
    "mobility": "value with units or null",
    "conductivity": "value with units or null",
    "band_gap": "value with units or null",
    "doping_outcome": "description or null",
    "structure_changes": "description or null",
    "phase_transition": "description or null",
    "defect_formation": "description or null",
    "distribution_characteristics": "description or null",
    "thermal": "thermal properties or null",
    "mechanical": "mechanical properties or null",
    "optical": "optical properties or null"
  }},
  "reasoning": "Step-by-step explanation of prediction using fundamental principles",
  "confidence": 0.0-1.0,
  "reasoning_type": "baseline_llm"
}}

IMPORTANT: Base predictions on:
1. Fundamental thermodynamics
2. Crystal structure considerations
3. Electronic structure principles
4. Known material behavior patterns

Do NOT make up specific numerical values unless you are confident based on well-known materials science knowledge.
"""

        try:
            response = self.ollama.generate_json(prompt, temperature=0.0)

            # Ensure metadata
            if 'reasoning_type' not in response:
                response['reasoning_type'] = 'baseline_llm'
            if 'model' not in response:
                response['model'] = self.model

            return response

        except Exception as e:
            logger.error(f"Forward prediction failed: {e}")
            return {
                'predicted_properties': {},
                'reasoning': f'Error: {str(e)}',
                'confidence': 0.0,
                'reasoning_type': 'baseline_llm_error'
            }

    def inverse_design(self, desired_properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design synthesis conditions to achieve desired properties using pure LLM.

        Args:
            desired_properties: Dict with target properties

        Returns:
            Synthesis recommendations dict
        """
        logger.info("Inverse design (Baseline LLM only)")

        # Convert desired properties to readable format
        property_keywords = [str(v) for v in desired_properties.values() if v is not None]
        query_string = " and ".join(property_keywords)

        prompt = f"""You are an expert materials scientist AI. Your task is to design a synthesis protocol to achieve specific material properties.

**Desired Material Properties:**
{query_string}

**Full Property Targets:**
{json.dumps(desired_properties, indent=2)}

**Task:**
Predict the most likely synthesis conditions to achieve the desired properties using your knowledge of materials science fundamentals. Directly provide your answer in a structured JSON format.

Respond with ONLY valid JSON in this exact format:
{{
  "suggested_synthesis_conditions": {{
    "host_material": "material name",
    "dopant": {{
      "element": "dopant element",
      "concentration": "concentration with units",
      "precursor": "precursor compound"
    }},
    "method": "synthesis method (CVD, MBE, etc.)",
    "temperature_c": numeric value or range,
    "pressure_pa": numeric value or null,
    "time_hours": numeric value or range,
    "atmosphere": "atmospheric conditions",
    "electric_field": "field conditions or null",
    "cooling_rate_c_min": numeric value or null,
    "substrate_pretreatment": "treatment details or null",
    "additional_parameters": "other parameters or null"
  }},
  "reasoning": "Step-by-step explanation of why these conditions should work",
  "confidence": 0.0-1.0,
  "reasoning_type": "baseline_llm_inverse"
}}

IMPORTANT: Base recommendations on:
1. Thermodynamic favorability
2. Kinetic accessibility
3. Known synthesis pathways for similar materials
4. Fundamental process-structure-property relationships

Do NOT make up specific parameters unless confident based on well-established knowledge.
"""

        try:
            response = self.ollama.generate_json(prompt, temperature=0.0)

            # Ensure metadata
            if 'reasoning_type' not in response:
                response['reasoning_type'] = 'baseline_llm_inverse'
            if 'model' not in response:
                response['model'] = self.model

            return response

        except Exception as e:
            logger.error(f"Inverse design failed: {e}")
            return {
                'suggested_synthesis_conditions': {},
                'reasoning': f'Error: {str(e)}',
                'confidence': 0.0,
                'reasoning_type': 'baseline_llm_inverse_error'
            }


def main():
    """Test the Baseline implementation."""
    print("="*60)
    print("Testing Baseline LLM (Ollama, No KG)")
    print("="*60)

    try:
        # Initialize
        baseline = BaselineOllama(model="qwen2:7b")

        # Test 1: Forward Prediction
        print("\n[Test 1] Forward Prediction")
        print("-" * 60)

        synthesis_params = {
            "method": "CVD",
            "temperature_c": 750,
            "material": "MoS2",
            "dopant": "Nb",
            "time_hours": 1,
            "atmosphere": "Ar/H2"
        }

        print(f"Input: {json.dumps(synthesis_params, indent=2)}")

        result = baseline.forward_prediction(synthesis_params)

        print(f"\nOutput:")
        print(f"  Predicted properties: {json.dumps(result.get('predicted_properties', {}), indent=2)}")
        print(f"  Confidence: {result.get('confidence', 0.0)}")
        print(f"  Reasoning type: {result.get('reasoning_type', 'unknown')}")
        print(f"  Reasoning: {result.get('reasoning', 'N/A')[:200]}...")

        # Test 2: Inverse Design
        print("\n[Test 2] Inverse Design")
        print("-" * 60)

        desired_props = {
            "carrier_type": "n-type",
            "mobility": "high carrier mobility (>50 cm2/V·s)",
            "material": "2D transition metal dichalcogenide"
        }

        print(f"Desired properties: {json.dumps(desired_props, indent=2)}")

        result = baseline.inverse_design(desired_props)

        print(f"\nOutput:")
        print(f"  Suggested conditions: {json.dumps(result.get('suggested_synthesis_conditions', {}), indent=2)}")
        print(f"  Confidence: {result.get('confidence', 0.0)}")
        print(f"  Reasoning type: {result.get('reasoning_type', 'unknown')}")
        print(f"  Reasoning: {result.get('reasoning', 'N/A')[:200]}...")

        print("\n" + "="*60)
        print("✓ Baseline LLM tests completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
