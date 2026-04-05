# 🚀 Quick Start: Run Test Pipeline

## Single Command

```bash
cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD/KG
./run_test_pipeline.sh
```

**Time:** 10-20 minutes
**Output:** `outputs/kg_raw_2d_doping.json`

---

## What It Does

1. ✅ Checks Ollama is running (uses `qwen2:7b` model)
2. 📥 Fetches 50 papers from OpenAlex
3. 🔬 Extracts causal relations with robust validation
4. 📊 Generates knowledge graph JSON
5. ✅ Validates output and shows statistics

---

## After Running

### Check Results

```bash
# View summary
cat outputs/kg_raw_2d_doping.json | python3 -m json.tool | head -30

# Count extractions
python3 -c "import json; kg=json.load(open('outputs/kg_raw_2d_doping.json')); print(f'Experiments: {len(kg[\"doping_experiments\"])}\nRelationships: {len(kg[\"causal_relationships\"])}')"

# Check quality
cat outputs/extraction_logs/*.log  # If any failures
```

### Expected Output

- **Experiments:** 10-30
- **Causal relationships:** 15-40
- **Success rate:** 60-80%

---

## Scale to 5000 Papers

If test succeeds:

```bash
# Option 1: Use full pipeline
python3 run_full_pipeline.py

# Option 2: Manual steps
python3 fetch_openalex_papers.py  # ~1 hour
python3 extract_from_abstracts.py  # ~3-5 hours
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Ollama not found | `brew install ollama && ollama serve` |
| Model missing | `ollama pull qwen2:7b` |
| High failures | Check `outputs/extraction_logs/` |
| Slow extraction | Normal for local inference |

---

## Files Created

```
outputs/
├── kg_raw_2d_doping.json       ← Main output
└── extraction_logs/            ← Failure logs

papers/openalex_test/
├── openalex_test_papers.json   ← Metadata
└── abstracts/*.txt             ← 50 abstracts
```

---

## Questions?

- Read: `TEST_PIPELINE_SUMMARY.md`
- Detailed guide: `README_OPENALEX_PIPELINE.md`
