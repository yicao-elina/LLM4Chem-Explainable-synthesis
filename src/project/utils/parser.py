import json
import re


def extract_json_from_response(text: str):
    """
    Extracts a JSON object from a string that contains a JSON markdown block.
    This version is more robust against malformed JSON from the LLM.
    """
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
