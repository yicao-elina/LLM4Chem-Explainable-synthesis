import os
import json
import re
from pathlib import Path
import google.generativeai as genai
import time # Added for potential rate limiting

MODEL_ID = "gemini-1.5-pro-latest"

# --- JSON extractor (no changes needed) ---
def extract_json_from_response(text: str):
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        print("✗ JSON Decode Error. Model may have returned malformed JSON.")
        return None

# --- Core (no changes needed) ---
def process_text(file_path: Path, prompt: str, model):
    try:
        # Increased context by reading first ~1M characters if files are huge
        with open(file_path, 'r', encoding='utf-8') as f:
            paper_text = f.read(1_000_000)
    except Exception as e:
        print(f"✗ Error reading {file_path}: {e}")
        return None

    full_prompt = f"{prompt}\n\n--- PAPER TEXT ---\n{paper_text}"
    try:
        response = model.generate_content(full_prompt)
        return extract_json_from_response(response.text)
    except Exception as e:
        print(f"✗ API error for {file_path.name}: {e}")
        return None

# --- Main Pipeline (MODIFIED) ---
def run_pipeline(txt_dir="papers/cleaned/train", prompt_file="prompt.txt", output_file="outputs/combined_doping_data.json"):
    # Setup API
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY. Set it with: export GOOGLE_API_KEY=your_key")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_ID)

    # Load prompt
    prompt = Path(prompt_file).read_text(encoding="utf-8")

    # MODIFICATION 1: Load existing data and track processed files
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_experiments, all_relationships = [], []
    processed_files = set()

    try:
        if output_path.exists():
            print(f"Loading existing data from {output_path}...")
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            all_experiments = existing_data.get("doping_experiments", [])
            all_relationships = existing_data.get("causal_relationships", [])

            # Populate the set of processed files for deduplication
            # This assumes a 'source_file' key exists in each record
            for item in all_experiments:
                if 'source_file' in item:
                    processed_files.add(item['source_file'])
            print(f"Found {len(all_experiments)} existing experiments from {len(processed_files)} files.")

    except (json.JSONDecodeError, FileNotFoundError):
        print("No valid existing data found. Starting fresh.")
        # File might be empty, corrupted, or not found. Start with empty lists.
        all_experiments, all_relationships = [], []
        processed_files = set()

    # Collect text files
    txt_files = sorted(Path(txt_dir).glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in '{txt_dir}'.")
        return
        
    new_files_processed = 0
    for txt in txt_files:
        # MODIFICATION 2: Skip files that are already processed
        if txt.name in processed_files:
            print(f"~ Skipping {txt.name} (already processed)")
            continue

        print(f"\nProcessing new file: {txt.name}...")
        new_files_processed += 1
        data = process_text(txt, prompt, model)
        
        if data:
            # MODIFICATION 3: Add source file information to each new record
            new_exps = data.get("doping_experiments", [])
            for exp in new_exps:
                exp['source_file'] = txt.name
            
            new_rels = data.get("causal_relationships", [])
            for rel in new_rels:
                rel['source_file'] = txt.name

            all_experiments.extend(new_exps)
            all_relationships.extend(new_rels)
            print(f"✓ Extracted {len(new_exps)} experiments and {len(new_rels)} relationships from {txt.name}")
        else:
            print(f"✗ Failed to extract data from {txt.name}")
        
        # Optional: add a small delay to respect API rate limits
        time.sleep(1) 

    # Save combined JSON (this now includes old + new data)
    result = {
        "doping_experiments": all_experiments,
        "causal_relationships": all_relationships,
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n--- Done ---")
    if new_files_processed == 0:
        print("No new files were processed.")
    print(f"Total Experiments: {len(all_experiments)} | Total Relationships: {len(all_relationships)}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    # Run the pipeline on the training set
    run_pipeline()
    ## Uncomment the line below to run on the test set
    # run_pipeline(txt_dir="papers/cleaned/test", prompt_file="prompt.txt", output_file="outputs/test_doping_data.json")