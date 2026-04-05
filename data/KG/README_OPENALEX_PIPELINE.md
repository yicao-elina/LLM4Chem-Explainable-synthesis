# OpenAlex Knowledge Graph Pipeline

This pipeline retrieves ~5000 papers on 2D material doping (2015-2026) from OpenAlex and constructs a causal knowledge graph for the ARIA reasoning engine.

## Overview

```
OpenAlex API → Metadata + Abstracts → Ollama Extraction → Knowledge Graph JSON → DAG Visualization
```

### Why OpenAlex?

- **Better metadata**: Comprehensive abstracts, keywords, concepts
- **No authentication**: Free API access without keys
- **Better coverage**: 250M+ works across all fields
- **Structured data**: Inverted index abstracts, automatic tagging

### Why use abstracts instead of full papers?

- **Efficiency**: Process 10x more papers in the same time
- **Quality**: Abstracts contain key methodology and findings
- **Cost**: No PDF storage/processing needed
- **Accessibility**: Many papers don't have open access PDFs

## Pipeline Steps

### Step 1: Fetch Papers from OpenAlex

```bash
cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD/KG
python fetch_openalex_papers.py
```

**Before running:**
1. Update `POLITE_EMAIL` in the script with your email (for polite API pool)
2. Optionally adjust `TARGET_PAPERS` (default: 5000)
3. Optionally adjust date range `START_YEAR` and `END_YEAR`

**What it does:**
- Queries OpenAlex with 20+ targeted searches for 2D materials doping
- Fetches title, abstract, keywords, DOI, citations, etc.
- Saves metadata to `papers/openalex_metadata/openalex_papers.json`
- Extracts abstracts to individual text files in `papers/openalex_metadata/abstracts/`
- Deduplicates by OpenAlex ID
- Respects rate limits (100K requests/day, 10 req/sec)

**Output:**
- `papers/openalex_metadata/openalex_papers.json` - Full metadata
- `papers/openalex_metadata/abstracts/*.txt` - Individual abstracts

**Time:** ~30-60 minutes for 5000 papers (depending on network)

### Step 2: Install Ollama (if not already installed)

```bash
# macOS
brew install ollama

# Or download from https://ollama.ai/

# Start Ollama
ollama serve

# In another terminal, pull the model
ollama pull llama3.1:7b
```

**Alternative models:**
- `mistral:7b` - Good balance of speed and quality
- `gemma2:7b` - Google's model, good for technical text
- `llama3.1:8b` - Slightly larger, better reasoning

### Step 3: Extract PSP Relations with Ollama

```bash
python extract_from_abstracts.py
```

**What it does:**
- Processes abstracts from `papers/openalex_metadata/abstracts/`
- Uses Ollama (7B model) to extract:
  - **Doping experiments**: host material, dopant, synthesis conditions, outcomes, properties
  - **Causal relationships**: Processing → Structure → Property chains
- Saves to `outputs/kg_from_abstracts.json`
- Incremental processing (can resume if interrupted)
- Saves failed extractions for debugging

**Output:**
- `outputs/kg_from_abstracts.json` - Combined knowledge graph

**Time:** ~2-5 hours for 5000 abstracts (local inference, depends on CPU/GPU)

### Step 4: Build and Visualize Causal Graph

```bash
python build_graph.py
```

**Before running:**
Update `build_graph.py` to use the new input file:
```python
json_file = 'outputs/kg_from_abstracts.json'
output_image = 'outputs/causal_graph_openalex.png'
```

**What it does:**
- Reads `causal_relationships` from JSON
- Builds Directed Acyclic Graph (DAG) using NetworkX
- Visualizes Processing → Structure → Property chains
- Saves high-resolution PNG (300 DPI)

**Output:**
- `outputs/causal_graph_openalex.png` - Publication-quality visualization

## Data Format

### OpenAlex Metadata (`openalex_papers.json`)

```json
{
  "metadata": {
    "total_papers": 5000,
    "date_fetched": "2026-02-01T...",
    "source": "OpenAlex API"
  },
  "papers": [
    {
      "openalex_id": "https://openalex.org/W...",
      "doi": "10.1021/...",
      "title": "Site-Selective Doping in MoS2...",
      "abstract": "We demonstrate...",
      "keywords": ["MoS2", "doping", "substitutional"],
      "concepts": ["Materials science", "Condensed matter physics"],
      "publication_year": 2023,
      "venue": "Nature Materials",
      "cited_by_count": 42,
      "is_open_access": true,
      "pdf_url": "https://..."
    }
  ]
}
```

### Knowledge Graph (`kg_from_abstracts.json`)

```json
{
  "metadata": {
    "total_experiments": 1523,
    "total_relationships": 892,
    "model_used": "llama3.1:7b"
  },
  "doping_experiments": [
    {
      "experiment_id": "exp_1",
      "host_material": "MoS2",
      "dopant": {
        "element": "Nb",
        "concentration": "2 at%",
        "precursor": "NbCl5"
      },
      "synthesis_conditions": {
        "method": "chemical vapor deposition",
        "temperature_c": 750,
        "time_hours": 1,
        "atmosphere": "Ar/H2"
      },
      "doping_outcome": {
        "site_distribution": {
          "primary_site": "substitutional",
          "site_specificity": "Mo site"
        },
        "structural_changes": {
          "lattice_parameter_change": "+0.2%"
        }
      },
      "property_changes": {
        "electronic": {
          "carrier_type": "n-type",
          "carrier_concentration": {"value": "1e19 cm-3"}
        }
      },
      "characterization_evidence": ["STEM", "XPS", "Hall"],
      "source_file": "0042_Site_selective_doping.txt"
    }
  ],
  "causal_relationships": [
    {
      "cause_parameter": "CVD temperature",
      "effect_on_doping": "increases substitutional fraction",
      "affected_property": "carrier concentration",
      "mechanism_quote": "Higher temperature promotes Mo vacancy formation, enabling Nb substitution",
      "source_file": "0042_Site_selective_doping.txt"
    }
  ]
}
```

## Configuration Options

### `fetch_openalex_papers.py`

```python
TARGET_PAPERS = 5000          # Number of papers to fetch
START_YEAR = 2015             # Start of date range
END_YEAR = 2026               # End of date range
POLITE_EMAIL = "your@email"   # Your email for polite API access
BATCH_SIZE = 200              # Papers per API request (max 200)
```

### `extract_from_abstracts.py`

```python
OLLAMA_MODEL = "llama3.1:7b"  # Model to use
INPUT_DIR = "papers/openalex_metadata/abstracts"
OUTPUT_FILE = "outputs/kg_from_abstracts.json"
```

## Integration with ARIA

The generated knowledge graph can be used with ARIA's three-tier reasoning cascade:

1. **Tier 1 (Graph-Constrained Reasoning)**:
   - Uses `causal_relationships` to constrain predictions
   - Follows Processing → Structure → Property chains
   - Ensures physical plausibility

2. **Tier 2 (Analogical Knowledge Transfer)**:
   - Uses `doping_experiments` as analogical examples
   - Matches by Factual Consistency (FC) and Numerical Compatibility (NC)
   - Transfers knowledge from similar systems

3. **Tier 3 (Parametric Fallback)**:
   - Falls back to LLM's internal knowledge when KG is insufficient
   - Safety circuit breaker

## Evaluation

Compare KG-enriched ARIA performance against baseline:

```bash
# Assuming evaluation scripts exist
python evaluate_aria.py --kg outputs/kg_from_abstracts.json
```

Expected improvements:
- Better constraint satisfaction (physical validity)
- Improved provenance (traceable to papers)
- Better generalization to novel materials

## Troubleshooting

### "Ollama not found"
```bash
# Install Ollama
brew install ollama  # macOS
# or visit https://ollama.ai/

# Start service
ollama serve
```

### "Model not found"
```bash
ollama pull llama3.1:7b
# or
ollama pull mistral:7b
```

### "Rate limited by OpenAlex"
- The script includes automatic retry with backoff
- If persistent, reduce `BATCH_SIZE` or add longer delays
- Check you've set `POLITE_EMAIL` (gives 10x higher rate limit)

### "JSON extraction failed"
- Check `outputs/failed_extractions/` for debug files
- Try a different model (e.g., `mistral:7b`)
- Abstracts with complex formatting may fail
- The script saves progress, so you can resume

### "Too few papers fetched"
- Add more queries to `OPENALEX_QUERIES`
- Broaden search terms
- Extend date range
- Lower minimum abstract length threshold

## Comparison with Existing Pipeline

| Feature | Old Pipeline (1-fetch_papers_multi.py) | New Pipeline (OpenAlex) |
|---------|----------------------------------------|-------------------------|
| Source | Semantic Scholar + arXiv | OpenAlex |
| Target | 200 papers | 5000 papers |
| Content | Full PDFs | Abstracts + metadata |
| Extraction | Gemini (paid) | Ollama (local, free) |
| Time | ~1 hour fetch + ~2 hours extract | ~1 hour fetch + ~3 hours extract |
| Coverage | Only open access PDFs | All papers with abstracts |
| Cost | $20-50 (Gemini API) | $0 (local inference) |

## Next Steps

1. **Evaluate KG quality**:
   - Manual review of extracted experiments
   - Precision/recall vs ground truth
   - Expert validation

2. **Enrich KG**:
   - Add more papers (increase `TARGET_PAPERS`)
   - Add more queries for specific materials
   - Include computational studies (DFT)

3. **Integrate with ARIA**:
   - Load KG into ARIA's reasoning engine
   - Benchmark on inverse design tasks
   - Compare vs baseline LLM

4. **Publication outputs**:
   - Generate CSV tables of experiments
   - LaTeX tables for paper
   - High-res figures of causal graph
   - Documentation of KG enrichment impact

## Citations

If using this pipeline, cite:

- **OpenAlex**: Priem, J., et al. (2022). OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. *arXiv preprint arXiv:2205.01833*.
- **Ollama**: https://ollama.ai/
- **ARIA**: [Your KDD 2026 paper]

## License

This pipeline is part of the 26KDD/ARIA research project. See main repository for license details.
