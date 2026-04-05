# Quick Start: OpenAlex KG Pipeline

Build a knowledge graph of 5000 papers on 2D material doping (2015-2026) in 4 simple steps.

## Prerequisites (5 minutes)

```bash
# 1. Install Ollama
brew install ollama  # macOS
# or visit https://ollama.ai/

# 2. Start Ollama server (in a separate terminal)
ollama serve

# 3. Download model
ollama pull llama3.1:7b

# 4. Install Python dependencies
pip install requests networkx matplotlib
```

## One-Command Run (4-6 hours)

```bash
cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD/KG

# IMPORTANT: First, edit fetch_openalex_papers.py line 17
# Replace: POLITE_EMAIL = "your_email@jhu.edu"
# With your actual email (for 10x higher API rate limit)

# Run complete pipeline
python run_full_pipeline.py
```

**What it does:**
1. Fetches 5000 papers from OpenAlex (~1 hour)
2. Extracts PSP relations with Ollama (~3 hours)
3. Builds causal graph visualization (~1 minute)
4. Generates publication outputs (~1 minute)

## Step-by-Step Run

If you prefer to run steps individually:

### Step 1: Fetch Papers (1 hour)

```bash
# Edit email first!
nano fetch_openalex_papers.py  # Line 17: POLITE_EMAIL = "your@email.edu"

python fetch_openalex_papers.py
```

**Output:**
- `papers/openalex_metadata/openalex_papers.json` (full metadata)
- `papers/openalex_metadata/abstracts/*.txt` (5000 abstract files)

### Step 2: Extract Relations (3 hours)

```bash
python extract_from_abstracts.py
```

**Output:**
- `outputs/kg_from_abstracts.json` (knowledge graph with experiments + causal relations)

### Step 3: Visualize Graph (1 minute)

```bash
python visualize_kg.py
```

**Output:**
- `outputs/publication/causal_graph_openalex.png` (high-res DAG visualization)

### Step 4: Generate Publication Outputs (1 minute)

```bash
python generate_publication_outputs.py
```

**Output:**
- `outputs/publication/experiments.csv`
- `outputs/publication/causal_relationships.csv`
- `outputs/publication/table_experiments.tex`
- `outputs/publication/table_relationships.tex`
- `outputs/publication/statistics.txt`
- `outputs/publication/host_materials.pdf`
- `outputs/publication/synthesis_methods.pdf`
- `outputs/publication/doping_sites.pdf`

## Verification

Check outputs:

```bash
# View statistics
cat outputs/publication/statistics.txt

# Check KG size
python -c "import json; kg = json.load(open('outputs/kg_from_abstracts.json')); print(f'Experiments: {len(kg[\"doping_experiments\"])}, Relations: {len(kg[\"causal_relationships\"])}')"

# View graph
open outputs/publication/causal_graph_openalex.png
```

## Troubleshooting

### "Ollama not found"
```bash
brew install ollama
ollama serve  # Keep running in separate terminal
```

### "Rate limited by OpenAlex"
- Make sure you set `POLITE_EMAIL` in `fetch_openalex_papers.py`
- The script will auto-retry with backoff

### "JSON extraction failed"
- Check `outputs/failed_extractions/` for debug info
- Progress is saved every 10 files, so you can resume
- Try a different model: edit `extract_from_abstracts.py` line 16
  ```python
  OLLAMA_MODEL = "mistral:7b"  # Alternative model
  ```

### "Pipeline is slow"
- **Fetch step (~1 hour)**: Normal, respects API rate limits
- **Extract step (~3 hours)**: Local CPU inference, expected for 5000 papers
  - To speed up: Process fewer papers (reduce `TARGET_PAPERS` in fetch script)
  - Or use GPU-accelerated Ollama (requires CUDA/Metal setup)

## Next Steps

### Integrate with ARIA

```python
# Load KG into ARIA
import json
kg = json.load(open('outputs/kg_from_abstracts.json'))

# Use in Tier 1: Graph-Constrained Reasoning
causal_relations = kg['causal_relationships']

# Use in Tier 2: Analogical Transfer
doping_experiments = kg['doping_experiments']
```

### Evaluate Impact

```bash
# Compare ARIA with/without KG
python evaluate_aria.py --baseline  # Without KG
python evaluate_aria.py --kg outputs/kg_from_abstracts.json  # With KG
```

### Expand KG

To fetch more papers:

```python
# Edit fetch_openalex_papers.py
TARGET_PAPERS = 10000  # Increase from 5000

# Add more queries to OPENALEX_QUERIES list
```

## Configuration Options

All scripts have configuration sections at the top:

**fetch_openalex_papers.py:**
- `TARGET_PAPERS`: Number to fetch (default: 5000)
- `START_YEAR` / `END_YEAR`: Date range (default: 2015-2026)
- `OPENALEX_QUERIES`: Search queries (add more for coverage)

**extract_from_abstracts.py:**
- `OLLAMA_MODEL`: Model to use (default: llama3.1:7b)
- `INPUT_DIR`: Where to find abstracts
- `EXTRACTION_PROMPT`: Customize extraction instructions

**generate_publication_outputs.py:**
- `INPUT_FILE`: KG JSON to process
- `OUTPUT_DIR`: Where to save outputs

## File Structure

```
KG/
├── fetch_openalex_papers.py       # Step 1: Fetch papers
├── extract_from_abstracts.py      # Step 2: Extract relations
├── visualize_kg.py                # Step 3: Build graph viz
├── generate_publication_outputs.py # Step 4: Generate outputs
├── run_full_pipeline.py           # Run all steps
├── build_graph.py                 # (existing) Graph builder
├── kg_example.json                # (existing) Format reference
│
├── papers/
│   └── openalex_metadata/
│       ├── openalex_papers.json   # Full metadata
│       └── abstracts/*.txt        # Individual abstracts
│
└── outputs/
    ├── kg_from_abstracts.json     # Complete KG
    └── publication/               # All publication outputs
        ├── *.csv                  # Data tables
        ├── *.tex                  # LaTeX tables
        ├── *.pdf/png              # Charts & graphs
        └── statistics.txt         # Summary stats
```

## Support

- **OpenAlex API docs**: https://docs.openalex.org/
- **Ollama docs**: https://ollama.ai/
- **Issues**: File in GitHub repository

## Citation

If using this pipeline, cite:

```bibtex
@software{openalex_kg_pipeline_2026,
  title = {OpenAlex Knowledge Graph Pipeline for 2D Materials Doping},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/...}
}
```
