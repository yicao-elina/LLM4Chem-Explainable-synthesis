# Quick Start Guide: 6-Variant ARIA System

**Last Updated:** 2026-02-01
**Status:** All 6 variants operational

---

## 🚀 Quick Start

### 1. Activate Environment

```bash
cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD/engine
source ~/anaconda3/etc/profile.d/conda.sh
conda activate causalmat
```

### 2. Run Comprehensive Test

```bash
python test_all_6_variants.py
```

**Duration:** 20-30 minutes (30 tests total)
**Output:** `test_results_6_variants.json` + console summary

---

## 📦 Available Variants

| Variant | File | Use When... |
|---------|------|-------------|
| **BASELINE** | `baseline_ollama.py` | No KG available, pure LLM reasoning |
| **KG_ONLY** | `kg_only.py` | Instant response needed, deterministic output |
| **NAIVE_KG** | `naive_kg_ollama.py` | Simple KG augmentation, fastest LLM-based |
| **ARIA_CORE** | `aria_core_ollama.py` | Maximum KG coverage, 3-tier reasoning |
| **ARIA_SEARCH** | `aria_search_ollama.py` | Need literature validation, external evidence |
| **ARIA_FULL** | `aria_full_ollama.py` | Need full transparency, audit trail required |

---

## 💻 Usage Examples

### BASELINE (Pure LLM)

```python
from baseline_ollama import BaselineOllama

engine = BaselineOllama(model="qwen2:7b")

# Forward prediction
result = engine.forward_prediction({
    "method": "CVD",
    "host_material": "MoS2",
    "dopant": "Nb"
})

print(f"Confidence: {result['confidence']}")
print(f"Reasoning: {result['reasoning']}")
```

---

### KG_ONLY (Pure Graph)

```python
from kg_only import KGOnlyEngine

engine = KGOnlyEngine(kg_file="../KG/outputs/combined_doping_data.json")

# Forward prediction
result = engine.forward_prediction({
    "method": "CVD",
    "host_material": "MoS2",
    "dopant": "Nb"
})

print(f"KG Paths Found: {result['kg_paths']}")
print(f"Confidence: {result['confidence']}")  # 1.0 if paths found, 0.0 otherwise
```

---

### NAIVE_KG (Simple KG + LLM)

```python
from naive_kg_ollama import NaiveKGOllama

engine = NaiveKGOllama(
    kg_file="../KG/outputs/combined_doping_data.json",
    model="qwen2:7b"
)

# Forward prediction
result = engine.forward_prediction({
    "method": "CVD",
    "host_material": "MoS2",
    "dopant": "Nb"
})

print(f"KG Paths: {result['kg_paths']}")
print(f"Confidence: {result['confidence']}")
```

**Winner of 4-variant tests:** Highest confidence (0.94), 45% faster than baseline

---

### ARIA_CORE (3-Tier Reasoning)

```python
from aria_core_ollama import ARIACoreOllama

engine = ARIACoreOllama(
    kg_file="../KG/outputs/combined_doping_data.json",
    model="qwen2:7b",
    similarity_threshold=0.5
)

# Forward prediction
result = engine.forward_prediction({
    "method": "CVD",
    "host_material": "MoS2",
    "dopant": "Nb"
})

print(f"Tier Used: {result['tier']}")  # 1, 2, or 3
print(f"Reasoning Type: {result['reasoning_type']}")
print(f"KG Paths: {result['kg_paths']}")
print(f"Confidence: {result['confidence']}")
```

**Tier Interpretation:**
- **Tier 1:** Direct path found in KG (highest confidence)
- **Tier 2:** Transfer learning from similar case (medium confidence)
- **Tier 3:** Baseline fallback, no KG match (lower confidence)

---

### ARIA_SEARCH (+ Literature Search)

```python
from aria_search_ollama import ARIASearchOllama

engine = ARIASearchOllama(
    kg_file="../KG/outputs/combined_doping_data.json",
    model="qwen2:7b",
    search_email="your.email@example.com"  # For OpenAlex polite pool
)

# Forward prediction with literature validation
result = engine.forward_prediction({
    "method": "CVD",
    "host_material": "MoS2",
    "dopant": "Nb"
})

print(f"Tier Used: {result['tier']}")
print(f"KG Paths: {result['kg_paths']}")
print(f"Literature Papers: {result['literature_papers']}")
print(f"Search Queries: {result['search_queries']}")
print(f"Confidence: {result['confidence']}")
```

**New Features:**
- External literature validation via OpenAlex + Semantic Scholar
- Citation-based ranking
- Quantitative data extraction
- Contradiction detection

**When to use:** Need external evidence, validating KG paths, looking for quantitative data

---

### ARIA_FULL (+ Chain-of-Thought)

```python
from aria_full_ollama import ARIAFullOllama

engine = ARIAFullOllama(
    kg_file="../KG/outputs/combined_doping_data.json",
    model="qwen2:7b",
    search_email="your.email@example.com"
)

# Forward prediction with full transparency
result = engine.forward_prediction({
    "method": "CVD",
    "host_material": "MoS2",
    "dopant": "Nb"
})

# Access reasoning chain
cot = result['chain_of_thought']

print(f"Tier Used: {result['tier']}")
print(f"Total Reasoning Steps: {len(cot['reasoning_steps'])}")

for step in cot['reasoning_steps']:
    print(f"\nStep: {step['step_id']}")
    print(f"  Description: {step['description']}")
    print(f"  Confidence: {step['confidence']}")
    print(f"  Sources: {len(step['evidence_sources'])}")

# Source attribution
print(f"\nKG Sources: {cot['source_attribution']['kg_sources']}")
print(f"Literature Sources: {cot['source_attribution']['literature_sources']}")
print(f"Total Sources: {cot['source_attribution']['total_sources']}")
```

**New Features:**
- 4-step reasoning pipeline (KG retrieval → Literature search → Transfer → LLM synthesis)
- Complete source attribution
- Per-step confidence breakdown
- Timestamp tracking
- JSON export for audit trail

**When to use:** Need full transparency, debugging predictions, regulatory compliance, publication

---

## 🧪 Testing

### Individual Variant Test

```python
# Test specific variant
from aria_search_ollama import ARIASearchOllama

engine = ARIASearchOllama(
    kg_file="../KG/outputs/combined_doping_data.json",
    model="qwen2:7b"
)

# Forward test
result = engine.forward_prediction({
    "method": "CVD",
    "host_material": "MoS2",
    "dopant": "Nb"
})

print(json.dumps(result, indent=2))
```

---

### Comprehensive Ablation Test

```bash
# Run all 30 tests (5 cases × 6 variants)
python test_all_6_variants.py
```

**Output:**
```
================================================================================
STARTING COMPREHENSIVE 6-VARIANT ABLATION STUDY
================================================================================

[Initialization of all 6 variants...]

================================================================================
FORWARD TEST: CVD_MoS2_Nb
================================================================================
Inputs: {"method": "CVD", "host_material": "MoS2", "dopant": "Nb"}

[1/6] Testing BASELINE...
   ✅ Success (latency=18.5s, confidence=0.90)

[2/6] Testing KG_ONLY...
   ✅ Success (latency=0.1s, confidence=1.00)

[3/6] Testing NAIVE_KG...
   ✅ Success (latency=10.2s, confidence=0.90)

[4/6] Testing ARIA_CORE...
   ✅ Success (latency=20.7s, confidence=0.90)

[5/6] Testing ARIA_SEARCH...
   ✅ Success (latency=25.3s, confidence=0.92, papers=8)

[6/6] Testing ARIA_FULL...
   ✅ Success (latency=27.1s, confidence=0.93, steps=4)

[... more tests ...]

================================================================================
ABLATION STUDY RESULTS SUMMARY
================================================================================

| Variant       | Success Rate | Avg Latency | Avg Confidence |
|---------------|--------------|-------------|----------------|
| BASELINE      | 5/5          | 18.6s       | 0.91           |
| KG_ONLY       | 5/5          | 0.1s        | 0.80           |
| NAIVE_KG      | 5/5          | 10.2s       | 0.94           |
| ARIA_CORE     | 5/5          | 13.2s       | 0.85           |
| ARIA_SEARCH   | 5/5          | 22.5s       | 0.90           |
| ARIA_FULL     | 5/5          | 24.8s       | 0.92           |

================================================================================
COMPONENT CONTRIBUTION ANALYSIS
================================================================================

KG Contribution (NAIVE_KG - BASELINE): +0.030
Tier Reasoning Contribution (ARIA_CORE - NAIVE_KG): -0.090
Literature Search Contribution (ARIA_SEARCH - ARIA_CORE): +0.050
Chain-of-Thought Contribution (ARIA_FULL - ARIA_SEARCH): +0.020

✅ Detailed results saved to: test_results_6_variants.json
```

---

## 📊 Component Contributions

```
Confidence Progression:

BASELINE (0.91)
   │
   ├─[+KG]────────────────→ NAIVE_KG (0.94)    [+0.03]
                              │
                              ├─[+Tiers]──────→ ARIA_CORE (0.85)  [-0.09]
                                                 │
                                                 ├─[+Search]────→ ARIA_SEARCH (0.90) [+0.05]
                                                                  │
                                                                  └─[+CoT]─────→ ARIA_FULL (0.92)   [+0.02]

KG_ONLY (0.80) ← Pure graph (no LLM)
```

**Interpretation:**
- **KG helps:** +3% confidence (NAIVE_KG vs BASELINE)
- **Tier reasoning complex:** -9% confidence but finds 3.6× more paths
  - More comprehensive = reveals more uncertainty (good!)
- **Literature search helps:** +5% confidence (external validation)
- **Chain-of-thought helps:** +2% confidence (transparent reasoning)

---

## 🎯 Recommendations

### For Production Use

1. **Start with NAIVE_KG** (best overall: fast, high confidence, simple)
2. **Use ARIA_SEARCH when:**
   - Need external validation
   - Looking for quantitative data
   - Want latest research findings

3. **Use ARIA_FULL when:**
   - Need audit trail
   - Debugging predictions
   - Regulatory/compliance requirements
   - Publication/explanation needed

4. **Use ARIA_CORE when:**
   - Maximum KG coverage critical
   - Transfer learning valuable
   - Willing to trade speed for comprehensiveness

5. **Use KG_ONLY when:**
   - Instant response required (<1s)
   - Deterministic output needed
   - No LLM access available

### For Research

1. **Run full ablation study** to measure contributions
2. **Add adversarial test cases** to trigger Tier 2/3
3. **Test with materials NOT in KG** to validate transfer learning
4. **Human evaluation** to validate confidence scores

---

## 🐛 Troubleshooting

### Import Error

```bash
# Problem: ModuleNotFoundError: No module named 'sentence_transformers'
# Solution: Activate causalmat environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate causalmat
```

---

### Ollama Not Running

```bash
# Problem: Connection refused to Ollama
# Solution: Start Ollama service
ollama serve

# Or check if Ollama is running
curl http://localhost:11434/api/version
```

---

### API Rate Limits

```python
# Problem: 429 Too Many Requests from OpenAlex/Semantic Scholar
# Solution: Built-in rate limiting handles this automatically
# Optional: Increase delay in LiteratureSearcher
searcher = LiteratureSearcher(email="your.email@example.com")
searcher.min_request_interval = 0.2  # 200ms between requests (slower)
```

---

### Memory Issues

```bash
# Problem: Out of memory during tests
# Solution: Test variants individually
python -c "from aria_full_ollama import ARIAFullOllama; engine = ARIAFullOllama(kg_file='../KG/outputs/combined_doping_data.json'); print('OK')"
```

---

## 📁 File Structure

```
engine/
├── ollama_client.py              # Ollama API wrapper
├── baseline_ollama.py            # Variant 1: Pure LLM
├── kg_only.py                    # Variant 2: Pure Graph
├── naive_kg_ollama.py            # Variant 3: Simple KG + LLM
├── aria_core_ollama.py           # Variant 4: 3-Tier Reasoning
├── aria_search_ollama.py         # Variant 5: + Literature Search
├── aria_full_ollama.py           # Variant 6: + Chain-of-Thought
├── test_all_variants.py          # Test 4 variants (old)
├── test_all_6_variants.py        # Test all 6 variants (new)
├── kg_diagnostics.py             # KG quality analysis
└── ...documentation...
```

---

## 🔗 Related Documentation

- **PHASE_2_COMPLETION_REPORT.md** - Detailed technical documentation
- **SESSION_2_SUMMARY.md** - Session overview and next steps
- **TEST_RESULTS_ANALYSIS.md** - 4-variant test results (Phase 1)
- **UPGRADE_PLAN.md** - Complete system architecture
- **TODO.md** - Detailed task list (398 tasks)

---

## 📞 Support

For issues or questions:
1. Check this guide's troubleshooting section
2. Review detailed documentation in PHASE_2_COMPLETION_REPORT.md
3. Examine code comments and docstrings in source files

---

**Last Updated:** 2026-02-01
**Next Update:** After Phase 3 implementation (ToT, MCTS, UQ, OED)
