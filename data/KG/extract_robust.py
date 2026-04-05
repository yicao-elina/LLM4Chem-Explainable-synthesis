"""
ROBUST Ollama-based extraction pipeline for 2D materials doping.

Requirements:
- Extract only experimentally implied relations
- Return empty list if no relation exists
- Avoid hallucinated numerical values
- Retry on JSON parse failure
- Validate required keys
- Log failure cases

Output: kg_raw_2d_doping.json
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import subprocess
from datetime import datetime

# ============= Configuration =============
OLLAMA_MODEL = "qwen2:7b"  # Available on your system
INPUT_DIR = Path("papers/openalex_test/abstracts")
OUTPUT_FILE = Path("outputs/kg_raw_2d_doping.json")
LOG_DIR = Path("outputs/extraction_logs")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 3
TIMEOUT_SECONDS = 120

# ============= Extraction Prompt =============
EXTRACTION_PROMPT = """You are a materials science expert. Extract ONLY experimentally-verified causal relationships from this abstract.

CRITICAL RULES:
1. Extract if the abstract describes experiments OR computational/simulation studies (DFT, MD, Monte Carlo, etc.)
   - Mark computational studies clearly in synthesis_conditions.method (e.g., "DFT calculation", "MD simulation")
2. Extract ONLY numerical values explicitly stated in the text (NO guessing or estimating)
3. If no clear causal relationship exists (experimental or computational), return empty arrays
4. Do NOT hallucinate values, methods, or relationships not in the text

Extract two types of data:

**doping_experiments**: Experimental details (if described)
- host_material: The 2D material being doped (e.g., "MoS2", "graphene")
- dopant.element: Dopant element (e.g., "Nb", "N")
- dopant.concentration: ONLY if explicitly stated (e.g., "2 at%", "5×10^18 cm^-3")
- synthesis_conditions.method: Synthesis method (e.g., "CVD", "MBE", "ion implantation")
- synthesis_conditions.temperature_c: ONLY if explicitly stated (numeric value or null)
- doping_outcome.site_distribution.primary_site: "substitutional" | "interstitial" | "surface_adsorption" | "vdw_gap" | null
- property_changes.electronic.carrier_type: "n-type" | "p-type" | null
- characterization_evidence: List of techniques mentioned (e.g., ["STEM", "XPS", "Hall measurement"])

**causal_relationships**: Causal chains (Processing → Structure → Property)
- cause_parameter: The experimental parameter that was varied (e.g., "CVD temperature", "annealing time")
- effect_on_doping: How it affected doping structure (e.g., "increased substitutional fraction")
- affected_property: The final property that changed (e.g., "carrier mobility", "conductivity")
- mechanism_quote: EXACT quote from abstract explaining mechanism (if available)

IMPORTANT:
- Include both experimental AND computational/simulation studies
- For computational studies, method must indicate this (e.g., "DFT", "MD simulation", "Monte Carlo")
- If no clear causal relationships exist, return empty arrays
- Use null for missing numerical values, not estimates
- Only include information explicitly in the abstract

---
ABSTRACT:
{abstract}
---

Respond with ONLY valid JSON:
```json
{{
  "doping_experiments": [...],
  "causal_relationships": [...]
}}
```
"""


# ============= Validation Schema =============
REQUIRED_EXPERIMENT_KEYS = ['host_material', 'dopant', 'synthesis_conditions', 'doping_outcome', 'property_changes', 'characterization_evidence']
REQUIRED_RELATIONSHIP_KEYS = ['cause_parameter', 'effect_on_doping', 'affected_property']


def query_ollama_with_retry(prompt: str, model: str, max_retries: int = MAX_RETRIES) -> Tuple[Optional[str], str]:
    """
    Query Ollama with retry logic.

    Returns:
        (response_text, error_message)
    """
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ['ollama', 'run', model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS
            )

            if result.returncode == 0:
                return result.stdout.strip(), ""
            else:
                error_msg = f"Ollama error (attempt {attempt+1}/{max_retries}): {result.stderr}"
                print(f"  {error_msg}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        except subprocess.TimeoutExpired:
            error_msg = f"Timeout (attempt {attempt+1}/{max_retries})"
            print(f"  {error_msg}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except Exception as e:
            error_msg = f"Exception (attempt {attempt+1}/{max_retries}): {e}"
            print(f"  {error_msg}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return None, f"Failed after {max_retries} attempts"


def extract_json_with_validation(text: str) -> Tuple[Optional[Dict], str]:
    """
    Extract and validate JSON from Ollama response.

    Returns:
        (parsed_json, error_message)
    """
    # Try multiple extraction strategies
    strategies = [
        (r'```json\s*([\s\S]*?)\s*```', 'JSON code block'),
        (r'```\s*([\s\S]*?)\s*```', 'Generic code block'),
        (r'\{[\s\S]*\}', 'Raw JSON object'),
    ]

    for pattern, strategy_name in strategies:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                data = json.loads(json_str)

                # Validate structure
                if not isinstance(data, dict):
                    continue

                if 'doping_experiments' not in data or 'causal_relationships' not in data:
                    return None, f"Missing required top-level keys (strategy: {strategy_name})"

                if not isinstance(data['doping_experiments'], list) or not isinstance(data['causal_relationships'], list):
                    return None, f"Top-level values must be arrays (strategy: {strategy_name})"

                # Validate experiments
                for i, exp in enumerate(data['doping_experiments']):
                    if not isinstance(exp, dict):
                        return None, f"Experiment {i} is not an object"

                    # Check required keys (warn but don't fail)
                    missing_keys = [k for k in REQUIRED_EXPERIMENT_KEYS if k not in exp]
                    if missing_keys:
                        print(f"    Warning: Experiment {i} missing keys: {missing_keys}")

                # Validate relationships
                for i, rel in enumerate(data['causal_relationships']):
                    if not isinstance(rel, dict):
                        return None, f"Relationship {i} is not an object"

                    missing_keys = [k for k in REQUIRED_RELATIONSHIP_KEYS if k not in rel]
                    if missing_keys:
                        return None, f"Relationship {i} missing required keys: {missing_keys}"

                return data, ""

            except json.JSONDecodeError as e:
                continue

    return None, "Could not extract valid JSON from response"


def log_failure(filename: str, abstract: str, response: str, error: str):
    """Log failed extraction for debugging."""
    log_file = LOG_DIR / f"failed_{filename.replace('.txt', '.log')}"

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
        f.write(f"ERROR: {error}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"ABSTRACT:\n{abstract}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"OLLAMA RESPONSE:\n{response}\n")


def process_abstract(abstract_file: Path) -> Tuple[Optional[Dict], str]:
    """
    Process single abstract with robust error handling.

    Returns:
        (extracted_data, error_message)
    """
    try:
        content = abstract_file.read_text(encoding='utf-8')

        # Extract abstract section
        abstract_match = re.search(r'ABSTRACT:\s*(.*)', content, re.DOTALL | re.IGNORECASE)
        if abstract_match:
            abstract_text = abstract_match.group(1).strip()
        else:
            abstract_text = content

        # Skip if too short
        if len(abstract_text) < 100:
            return None, "Abstract too short (<100 chars)"

        # Build prompt
        prompt = EXTRACTION_PROMPT.format(abstract=abstract_text[:2000])

        # Query Ollama with retry
        response, error = query_ollama_with_retry(prompt, OLLAMA_MODEL)
        if not response:
            log_failure(abstract_file.name, abstract_text, "", error)
            return None, error

        # Extract and validate JSON
        data, error = extract_json_with_validation(response)
        if not data:
            log_failure(abstract_file.name, abstract_text, response, error)
            return None, error

        # Check if extraction is non-empty
        exp_count = len(data.get('doping_experiments', []))
        rel_count = len(data.get('causal_relationships', []))

        if exp_count == 0 and rel_count == 0:
            return data, "Empty extraction (no experimental content)"

        return data, ""

    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        log_failure(abstract_file.name, "", "", error_msg)
        return None, error_msg


def main():
    """Main extraction pipeline with robust error handling."""
    print(f"\n{'='*60}")
    print("ROBUST Ollama Extraction Pipeline")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"{'='*60}\n")

    # Check input directory
    if not INPUT_DIR.exists():
        print(f"✗ ERROR: Input directory not found: {INPUT_DIR}")
        print(f"  Run fetch_openalex_test.py first")
        return 1

    abstract_files = sorted(INPUT_DIR.glob("*.txt"))
    if not abstract_files:
        print(f"✗ ERROR: No abstract files found in {INPUT_DIR}")
        return 1

    print(f"Found {len(abstract_files)} abstract files\n")

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

                for item in all_experiments:
                    if 'source_file' in item:
                        processed_files.add(item['source_file'])

                print(f"  Loaded {len(all_experiments)} experiments from {len(processed_files)} files\n")
        except json.JSONDecodeError:
            print("  Warning: Could not load existing data\n")

    # Statistics tracking
    stats = {
        'total': len(abstract_files),
        'processed': 0,
        'success': 0,
        'empty': 0,
        'failed': 0,
        'skipped': len(processed_files)
    }

    # Process each abstract
    start_time = time.time()

    for i, abstract_file in enumerate(abstract_files, 1):
        if abstract_file.name in processed_files:
            print(f"[{i}/{len(abstract_files)}] Skipping {abstract_file.name} (already processed)")
            continue

        print(f"[{i}/{len(abstract_files)}] Processing {abstract_file.name}...")
        stats['processed'] += 1

        data, error = process_abstract(abstract_file)

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

            if new_exps or new_rels:
                print(f"  ✓ Extracted {len(new_exps)} experiments, {len(new_rels)} relationships")
                stats['success'] += 1
            else:
                print(f"  ○ Empty extraction: {error}")
                stats['empty'] += 1
        else:
            print(f"  ✗ Failed: {error}")
            stats['failed'] += 1

        # Save progress every 5 files
        if stats['processed'] % 5 == 0:
            result = {
                'doping_experiments': all_experiments,
                'causal_relationships': all_relationships
            }
            OUTPUT_FILE.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
            print(f"  Progress saved ({len(all_experiments)} total experiments)")

    # Final save with metadata
    result = {
        'metadata': {
            'extraction_date': datetime.now().isoformat(),
            'model_used': OLLAMA_MODEL,
            'total_experiments': len(all_experiments),
            'total_relationships': len(all_relationships),
            'source_files': len(set(exp.get('source_file') for exp in all_experiments)),
            'statistics': stats
        },
        'doping_experiments': all_experiments,
        'causal_relationships': all_relationships
    }

    OUTPUT_FILE.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"\nStatistics:")
    print(f"  Total files: {stats['total']}")
    print(f"  Skipped (already done): {stats['skipped']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Success: {stats['success']}")
    print(f"  Empty (no experimental content): {stats['empty']}")
    print(f"  Failed: {stats['failed']}")
    print(f"\nExtracted:")
    print(f"  Experiments: {len(all_experiments)}")
    print(f"  Causal relationships: {len(all_relationships)}")
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Failed logs: {LOG_DIR}/")

    return 0


if __name__ == "__main__":
    exit(main())
