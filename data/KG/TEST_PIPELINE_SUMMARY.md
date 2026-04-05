# Test Pipeline Summary

## What Was Created

### 1. Test Scripts (50 papers)

**`fetch_openalex_test.py`**
- Fetches 50 papers from OpenAlex (instead of 5000)
- Focused queries: MoS2, graphene, site-selective doping
- Date range: 2020-2026 (narrower for testing)
- Output: `papers/openalex_test/`

**`extract_robust.py`**
- **Robust extraction** with all requirements implemented:
  - ✅ Extract only experimentally implied relations
  - ✅ Return empty list if no relation exists
  - ✅ Avoid hallucinated numerical values
  - ✅ Retry on JSON parse failure (3 retries with exponential backoff)
  - ✅ Validate required keys
  - ✅ Log failure cases to `outputs/extraction_logs/`
- Uses `qwen2:7b` (available on your system)
- Output: `outputs/kg_raw_2d_doping.json`

**`run_test_pipeline.sh`**
- Master bash script to run complete pipeline
- Includes prerequisite checks
- Color-coded output
- Progress tracking
- Validation steps

### 2. Key Features of Robust Extraction

```python
# Strict validation in extraction prompt
"CRITICAL RULES:
1. Extract ONLY if the abstract explicitly describes experiments
2. Extract ONLY numerical values explicitly stated in the text
3. If no clear experimental causal relationship exists, return empty arrays
4. Do NOT hallucinate values, methods, or relationships"

# Retry logic
MAX_RETRIES = 3
TIMEOUT_SECONDS = 120

# JSON validation
- Validates structure
- Checks required keys
- Warns on missing optional keys
- Multiple extraction strategies

# Failure logging
- Saves failed abstracts
- Includes Ollama response
- Timestamps for debugging
```

### 3. Output Format

**`kg_raw_2d_doping.json`** contains:

```json
{
  "metadata": {
    "extraction_date": "2026-02-01T...",
    "model_used": "qwen2:7b",
    "total_experiments": 15,
    "total_relationships": 23,
    "source_files": 50,
    "statistics": {
      "total": 50,
      "processed": 50,
      "success": 35,
      "empty": 10,
      "failed": 5
    }
  },
  "doping_experiments": [
    {
      "host_material": "MoS2",
      "dopant": {
        "element": "Nb",
        "concentration": "2 at%",
        "precursor": "NbCl5"
      },
      "synthesis_conditions": {
        "method": "chemical vapor deposition",
        "temperature_c": 750,
        "time_hours": 1
      },
      "doping_outcome": {
        "site_distribution": {
          "primary_site": "substitutional"
        }
      },
      "property_changes": {
        "electronic": {
          "carrier_type": "n-type"
        }
      },
      "characterization_evidence": ["STEM", "XPS"],
      "source_file": "001_MoS2_doping.txt"
    }
  ],
  "causal_relationships": [
    {
      "cause_parameter": "CVD temperature",
      "effect_on_doping": "increased substitutional fraction",
      "affected_property": "carrier concentration",
      "mechanism_quote": "Higher temperature promotes Mo vacancy formation",
      "source_file": "001_MoS2_doping.txt"
    }
  ]
}
```

## Running the Test Pipeline

### Quick Run (Recommended)

```bash
cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD/KG

# Run complete test pipeline
./run_test_pipeline.sh
```

**Expected time:** 10-20 minutes for 50 papers

### Step-by-Step Run

```bash
# Step 1: Fetch 50 papers (~2 minutes)
python3 fetch_openalex_test.py

# Step 2: Extract relations (~10-15 minutes)
python3 extract_robust.py

# Step 3: View results
cat outputs/kg_raw_2d_doping.json | python3 -m json.tool | head -50
```

## Verification

After running, check:

```bash
# 1. Check metadata
python3 -c "import json; kg=json.load(open('outputs/kg_raw_2d_doping.json')); print(json.dumps(kg['metadata'], indent=2))"

# 2. Count extractions
python3 -c "import json; kg=json.load(open('outputs/kg_raw_2d_doping.json')); print(f\"Experiments: {len(kg['doping_experiments'])}, Relations: {len(kg['causal_relationships'])}\")"

# 3. Check for failures
ls -la outputs/extraction_logs/

# 4. View sample relationship
python3 -c "import json; kg=json.load(open('outputs/kg_raw_2d_doping.json')); rel=kg['causal_relationships'][0] if kg['causal_relationships'] else {}; print(f\"{rel.get('cause_parameter')} → {rel.get('effect_on_doping')} → {rel.get('affected_property')}\")"
```

## Expected Results

For 50 test papers, you should see:

- **Success rate**: ~60-80% (some abstracts lack experimental details)
- **Experiments extracted**: 10-30 (depends on content)
- **Causal relationships**: 15-40 (depends on content)
- **Empty extractions**: ~10-20 (computational/theoretical papers)
- **Failed extractions**: <5 (logged for debugging)

## Scaling to 5000 Papers

If test is successful, scale up:

### Option 1: Use existing full-scale scripts

```bash
# 1. Update email in fetch_openalex_papers.py
nano fetch_openalex_papers.py  # Line 17

# 2. Run full pipeline
python3 run_full_pipeline.py
```

### Option 2: Modify test scripts

```python
# In fetch_openalex_test.py:
TARGET_PAPERS = 5000
OUTPUT_DIR = Path("papers/openalex_metadata")

# In extract_robust.py:
INPUT_DIR = Path("papers/openalex_metadata/abstracts")
```

**Expected time for 5000 papers:**
- Fetch: ~1 hour
- Extract: ~3-5 hours (depends on CPU)

## Troubleshooting

### "No experimental content extracted"

This is **normal** if:
- Abstract is purely computational/theoretical
- Abstract doesn't describe synthesis methods
- No clear causal relationships mentioned

The robust extraction returns empty arrays (not an error).

### "JSON parse failure"

Check `outputs/extraction_logs/` for:
- Model response format issues
- Incomplete JSON from model
- Extraction retries exhausted

Fix: Try different model or adjust prompt.

### "Ollama timeout"

Increase timeout:
```python
TIMEOUT_SECONDS = 180  # In extract_robust.py
```

### High failure rate (>20%)

1. Check model: `ollama list`
2. Check logs: `cat outputs/extraction_logs/failed_*.log`
3. Try alternative model: `OLLAMA_MODEL = "mistral:7b"`

## Quality Checks

### Manual validation

```bash
# Pick random sample
python3 << EOF
import json
import random

kg = json.load(open('outputs/kg_raw_2d_doping.json'))
if kg['causal_relationships']:
    sample = random.choice(kg['causal_relationships'])
    print("Random causal relationship:")
    print(f"  Cause: {sample['cause_parameter']}")
    print(f"  Effect: {sample['effect_on_doping']}")
    print(f"  Property: {sample['affected_property']}")
    print(f"  Source: {sample['source_file']}")
EOF
```

Then check the source abstract to verify accuracy.

### Validate against ground truth

Compare extracted relationships with known papers you're familiar with.

## Next Steps After Successful Test

1. **Scale to 5000 papers** (see above)
2. **Integrate with ARIA** reasoning engine
3. **Evaluate impact** on inverse design tasks
4. **Generate publication outputs**:
   ```bash
   python3 generate_publication_outputs.py
   ```

## File Structure After Test

```
KG/
├── fetch_openalex_test.py
├── extract_robust.py
├── run_test_pipeline.sh ✓
│
├── papers/openalex_test/
│   ├── openalex_test_papers.json
│   └── abstracts/
│       ├── 001_MoS2_doping.txt
│       ├── 002_graphene_substitution.txt
│       └── ... (50 total)
│
└── outputs/
    ├── kg_raw_2d_doping.json ✓
    └── extraction_logs/
        ├── failed_005_*.log
        └── failed_023_*.log
```

## Support

- Review logs: `outputs/extraction_logs/`
- Check metadata: `outputs/kg_raw_2d_doping.json` → `metadata` → `statistics`
- Adjust prompts: Edit `EXTRACTION_PROMPT` in `extract_robust.py`
