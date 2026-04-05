"""
Extract PSP (Processing → Structure → Property) causal relations from paper abstracts.
Optimized for OpenAlex abstract data using Ollama (7B model) instead of Gemini.

This script processes abstracts fetched from OpenAlex and extracts:
1. Doping experiments (host material, dopant, synthesis conditions, outcomes)
2. Causal relationships (processing → structure → property)

Uses Ollama for local inference (no API costs).
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import time
import subprocess

# ============= Configuration =============
OLLAMA_MODEL = "llama3.1:7b"  # or "mistral:7b", "gemma2:7b"
INPUT_DIR = Path("papers/openalex_metadata/abstracts")
OUTPUT_FILE = Path("outputs/kg_from_abstracts.json")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ============= Prompt Template =============
EXTRACTION_PROMPT = """You are a materials science expert analyzing a research paper abstract about 2D material doping.

Extract the following information in JSON format:

1. **doping_experiments**: Array of experiments described in the abstract
   - host_material: The 2D material being doped (e.g., "MoS2", "graphene")
   - dopant: {element, concentration (if mentioned), precursor}
   - synthesis_conditions: {method, temperature_c, time_hours, atmosphere, etc.}
   - doping_outcome: {site_distribution, structural_changes, distribution_characteristics}
   - property_changes: {electronic, thermal, mechanical, optical}
   - characterization_evidence: List of techniques used

2. **causal_relationships**: Array of causal links in the form Processing → Structure → Property
   - cause_parameter: The processing/synthesis parameter (e.g., "temperature", "cooling rate")
   - effect_on_doping: How it affects doping (e.g., "increases substitutional fraction")
   - affected_property: The final property affected (e.g., "carrier mobility", "band gap")
   - mechanism_quote: Direct quote explaining the mechanism

**Important**:
- Only extract information explicitly mentioned in the abstract
- Use null for missing values
- Focus on causal relationships
- Keep quotes exact from the text

---
ABSTRACT:
{abstract}
---

Respond with ONLY valid JSON in this format:
```json
{{
  "doping_experiments": [...],
  "causal_relationships": [...]
}}
```
"""


def check_ollama_available() -> bool:
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_model_available(model_name: str) -> bool:
    """Check if specific Ollama model is available."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return model_name in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def query_ollama(prompt: str, model: str = OLLAMA_MODEL) -> Optional[str]:
    """
    Query Ollama with a prompt and return the response.

    Args:
        prompt: The prompt to send
        model: The Ollama model to use

    Returns:
        Response text or None if error
    """
    try:
        result = subprocess.run(
            ['ollama', 'run', model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes max per query
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"✗ Ollama error: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print(f"✗ Ollama timeout (>120s)")
        return None
    except Exception as e:
        print(f"✗ Error querying Ollama: {e}")
        return None


def extract_json_from_response(text: str) -> Optional[Dict]:
    """
    Extract JSON from Ollama response.
    Tries multiple strategies to handle various response formats.
    """
    # Strategy 1: Look for JSON code block
    match = re.search(r'```json\s*([\s\S]*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: Look for any code block
    match = re.search(r'```\s*([\s\S]*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try to find JSON object directly
    match = re.search(r'\{[\s\S]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 4: Clean up common issues and try again
    # Remove markdown, explanatory text, etc.
    cleaned = text
    for prefix in ["Here's the JSON:", "The JSON output:", "Response:", "Output:"]:
        cleaned = cleaned.replace(prefix, "")

    cleaned = cleaned.strip()
    if cleaned.startswith('{') and cleaned.endswith('}'):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    print(f"✗ Could not extract valid JSON from response")
    return None


def process_abstract(abstract_file: Path) -> Optional[Dict]:
    """
    Process a single abstract file and extract structured data.

    Args:
        abstract_file: Path to abstract text file

    Returns:
        Dictionary with doping_experiments and causal_relationships, or None
    """
    try:
        content = abstract_file.read_text(encoding='utf-8')

        # Extract abstract section (format: TITLE: ... ABSTRACT: ...)
        abstract_match = re.search(r'ABSTRACT:\s*(.*)', content, re.DOTALL | re.IGNORECASE)
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
        else:
            abstract_text = content

        # Skip if abstract is too short
        if len(abstract_text) < 100:
            print(f"  Skipping {abstract_file.name} (abstract too short)")
            return None

        # Build prompt
        prompt = EXTRACTION_PROMPT.format(abstract=abstract_text[:2000])  # Limit length

        # Query Ollama
        response = query_ollama(prompt)
        if not response:
            return None

        # Extract JSON
        data = extract_json_from_response(response)
        if not data:
            # Save failed response for debugging
            debug_file = OUTPUT_FILE.parent / "failed_extractions" / f"{abstract_file.stem}_response.txt"
            debug_file.parent.mkdir(exist_ok=True)
            debug_file.write_text(response, encoding='utf-8')
            return None

        # Validate structure
        if 'doping_experiments' not in data and 'causal_relationships' not in data:
            print(f"  Warning: No valid data extracted from {abstract_file.name}")
            return None

        return data

    except Exception as e:
        print(f"✗ Error processing {abstract_file.name}: {e}")
        return None


def main():
    """
    Main workflow: process all abstracts and build combined KG.
    """
    print(f"\n{'='*60}")
    print("Extract PSP Relations from OpenAlex Abstracts")
    print(f"{'='*60}\n")

    # Check Ollama availability
    print("Checking Ollama installation...")
    if not check_ollama_available():
        print("✗ ERROR: Ollama is not installed or not running")
        print("  Install: https://ollama.ai/")
        print("  Start: ollama serve")
        return

    print(f"✓ Ollama is available")

    # Check model availability
    print(f"Checking if model '{OLLAMA_MODEL}' is available...")
    if not check_model_available(OLLAMA_MODEL):
        print(f"✗ ERROR: Model '{OLLAMA_MODEL}' not found")
        print(f"  Download with: ollama pull {OLLAMA_MODEL}")
        return

    print(f"✓ Model '{OLLAMA_MODEL}' is ready\n")

    # Load existing data if available
    all_experiments = []
    all_relationships = []
    processed_files = set()

    if OUTPUT_FILE.exists():
        print(f"Loading existing data from {OUTPUT_FILE}...")
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                all_experiments = existing_data.get('doping_experiments', [])
                all_relationships = existing_data.get('causal_relationships', [])

                # Track processed files
                for item in all_experiments:
                    if 'source_file' in item:
                        processed_files.add(item['source_file'])

            print(f"  Loaded {len(all_experiments)} experiments from {len(processed_files)} files\n")
        except json.JSONDecodeError:
            print("  Warning: Could not load existing data, starting fresh\n")

    # Get abstract files
    if not INPUT_DIR.exists():
        print(f"✗ ERROR: Input directory not found: {INPUT_DIR}")
        print(f"  Run fetch_openalex_papers.py first")
        return

    abstract_files = sorted(INPUT_DIR.glob("*.txt"))
    if not abstract_files:
        print(f"✗ ERROR: No abstract files found in {INPUT_DIR}")
        return

    print(f"Found {len(abstract_files)} abstract files")
    print(f"Already processed: {len(processed_files)}")
    print(f"To process: {len(abstract_files) - len(processed_files)}\n")

    # Process each abstract
    new_files_processed = 0
    start_time = time.time()

    for i, abstract_file in enumerate(abstract_files, 1):
        if abstract_file.name in processed_files:
            continue

        print(f"[{i}/{len(abstract_files)}] Processing {abstract_file.name}...")
        new_files_processed += 1

        data = process_abstract(abstract_file)

        if data:
            # Add source file to each entry
            new_exps = data.get('doping_experiments', [])
            for exp in new_exps:
                exp['source_file'] = abstract_file.name

            new_rels = data.get('causal_relationships', [])
            for rel in new_rels:
                rel['source_file'] = abstract_file.name

            all_experiments.extend(new_exps)
            all_relationships.extend(new_rels)

            print(f"  ✓ Extracted {len(new_exps)} experiments, {len(new_rels)} relationships")
        else:
            print(f"  ✗ Failed to extract data")

        # Save progress every 10 files
        if new_files_processed % 10 == 0:
            result = {
                'doping_experiments': all_experiments,
                'causal_relationships': all_relationships
            }
            OUTPUT_FILE.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
            print(f"  Progress saved ({len(all_experiments)} total experiments)")

    # Final save
    result = {
        'metadata': {
            'total_experiments': len(all_experiments),
            'total_relationships': len(all_relationships),
            'source_files': len(set(exp.get('source_file') for exp in all_experiments)),
            'model_used': OLLAMA_MODEL
        },
        'doping_experiments': all_experiments,
        'causal_relationships': all_relationships
    }

    OUTPUT_FILE.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("Extraction Complete!")
    print(f"{'='*60}")
    print(f"Processed {new_files_processed} new files in {elapsed/60:.1f} minutes")
    print(f"Total experiments: {len(all_experiments)}")
    print(f"Total causal relationships: {len(all_relationships)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"\nNext step:")
    print(f"  python build_graph.py")


if __name__ == "__main__":
    main()
