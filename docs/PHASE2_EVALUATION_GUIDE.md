# ARIA Phase II Evaluation Framework Guide

**Status:** Phase II-A ✅ Complete | Phase II-B ✅ Complete
**Date:** 2026-02-02
**Author:** ARIA Team

---

## 📋 Overview

Phase II introduces **2D materials-specific evaluation metrics** using an **LLM-as-a-Judge** approach. This framework goes beyond generic semantic similarity to assess domain-specific correctness for processing-structure-property (PSP) reasoning.

### Phase I vs Phase II

| Aspect | Phase I (Reproduction) | Phase II (New Framework) |
|--------|----------------------|--------------------------|
| **Metrics** | Generic: Scientific Accuracy, Functional Equivalence, Reasoning Quality, Completeness, Interpretability | Domain-Specific: Processing Feasibility, Structure Emergence, Property Consistency, Causal PSP Reasoning |
| **Evaluation** | Rule-based + semantic similarity | LLM-as-a-Judge with detailed rubrics |
| **Systems** | 4 variants (Baseline, Naive KG, Online KG, ARIA) | 6 variants (adds KG_ONLY, ARIA_CORE) |
| **Scoring** | 0-1 normalized scores | 0-100 point system with weighted components |

---

## 🎯 Phase II Metrics

### 1. Processing Feasibility (40 points)

**What it measures:** Thermodynamic & kinetic viability of predicted processing conditions

**Criteria:**
- Temperature ranges appropriate for material phase stability
- Pressure conditions compatible with synthesis method
- Time scales realistic for defect formation/annealing
- Atmosphere correct for doping/oxidation control
- Equipment feasibility in typical labs
- Safety considerations (no hazardous conditions)

**Scoring:**
- 35-40: Fully viable, realistic conditions
- 25-34: Minor feasibility issues (e.g., slightly high temp)
- 15-24: Significant issues (wrong atmosphere, unrealistic pressure)
- 0-14: Fundamentally impossible (violates thermodynamics)

---

### 2. Structure Emergence (30 points)

**What it measures:** Accuracy of predicted structural outcomes from processing

**Criteria:**
- Defect type prediction (vacancy, substitution, interstitial)
- Defect density/concentration estimates
- Lattice strain effects
- Stacking order (for heterostructures)
- Phase purity considerations
- Crystal structure/symmetry preservation

**Scoring:**
- 25-30: Correct defect type, realistic density, accurate structure
- 18-24: Correct type, minor inaccuracies in density/strain
- 10-17: Partially correct (right family, wrong specific type)
- 0-9: Incorrect defect prediction or major structural errors

---

### 3. Property Consistency (20 points)

**What it measures:** Coherence between predicted properties and structure/processing

**Criteria:**
- Electronic properties (band gap, carrier type, conductivity) match structure
- Mechanical properties consistent with defect density
- Optical properties aligned with electronic structure
- Magnetic properties match dopant configuration
- No violations of conservation laws or symmetry

**Scoring:**
- 17-20: All properties consistent with structure and processing
- 12-16: Minor inconsistencies (band gap off by <0.5 eV)
- 6-11: Significant inconsistencies (wrong carrier type)
- 0-5: Major violations (metallic for insulating structure)

---

### 4. Causal PSP Reasoning (10 points)

**What it measures:** Quality of Processing→Structure→Property causal chain

**Criteria:**
- Clear P→S→P chain with explicit connections
- Mechanistic explanations (how processing creates structure)
- Physical justifications grounded in theory
- Uncertainty acknowledged and quantified
- References to principles or literature

**Scoring:**
- 8-10: Clear chain, mechanisms, uncertainty quantified
- 5-7: Partial chain, some mechanisms
- 2-4: Mentions connections but lacks detail
- 0-1: No causal reasoning or PSP chain

---

## 🔧 Using the LLM Judge

### Installation

The judge is already integrated into the ARIA evaluation framework:

```bash
# Already installed with ARIA
cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD
source ~/anaconda3/etc/profile.d/conda.sh
conda activate causalmat
```

### Basic Usage

```python
from src.evaluation.ollama_judge import OllamaJudge

# Initialize judge
judge = OllamaJudge(model="qwen2:7b", temperature=0.0)

# Evaluate a single prediction
result = judge.evaluate_all_metrics(
    query={
        'task': 'forward_prediction',
        'processing': {
            'method': 'CVD',
            'temperature': '800°C',
            'atmosphere': 'H2/Ar'
        }
    },
    prediction={
        'structure': {'defects': 'sulfur vacancies'},
        'properties': {'carrier_type': 'n-type'},
        'reasoning': 'H2 creates reducing conditions...'
    },
    ground_truth={
        'structure': {'defects': 'sulfur vacancies'},
        'properties': {'carrier_type': 'n-type'}
    }
)

# Access scores
print(f"Overall: {result['overall_score']}/100")
print(f"Processing Feasibility: {result['metric_scores']['processing_feasibility']['score']}/40")
print(f"Structure Emergence: {result['metric_scores']['structure_emergence']['score']}/30")
print(f"Property Consistency: {result['metric_scores']['property_consistency']['score']}/20")
print(f"PSP Reasoning: {result['metric_scores']['causal_psp_reasoning']['score']}/10")
```

### Batch Evaluation

```python
# Evaluate multiple test cases
test_cases = [
    {
        'id': 'test_1',
        'query': {...},
        'prediction': {...},
        'ground_truth': {...}
    },
    # ... more test cases
]

results = judge.batch_evaluate(test_cases)

# Generate report
from src.evaluation.ollama_judge import create_judge_report
create_judge_report(results, 'results/phase2/judge_report.txt')
```

---

## 📊 Running Phase II Evaluation

### Option 1: Quick Test (Example)

```bash
# Test the judge on example data
cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD
source ~/anaconda3/etc/profile.d/conda.sh
conda activate causalmat

python src/evaluation/ollama_judge.py
```

### Option 2: Full 6-Variant Evaluation (Coming Soon)

Phase II-C will provide a complete evaluation script that:
- Evaluates all 6 variants (BASELINE, KG_ONLY, NAIVE_KG, ARIA_CORE, ARIA_SEARCH, ARIA_FULL)
- Uses LLM judge for all 4 new metrics
- Generates NEW_TABLE_2 results
- Calculates component contributions (KG value, Tier value, Search value, CoT value)

---

## 📁 File Structure

```
docs/evaluation/
├── NEW_TABLE_1_metrics.tex          # Metric definitions (LaTeX)
└── NEW_TABLE_2_results.tex          # Results schema (LaTeX template)

src/evaluation/
├── ollama_judge.py                  # LLM-as-a-Judge implementation
├── metrics.py                       # Phase I metrics (legacy)
└── run_phase1_evaluation.py         # Phase I runner (legacy)

results/phase2/
├── judge_reports/                   # Detailed judge evaluations
├── tables/                          # Generated LaTeX tables
└── logs/                            # Evaluation logs
```

---

## 🔬 Metric Design Philosophy

### Inspired by Chemical Reactions Paper

The reference paper (arXiv-2512.13668v1) evaluates chemical synthesis predictions using:
- **Substance Accuracy** → We adapt to **Structure Emergence**
- **Action Coverage** → We adapt to **Processing Feasibility**
- **Reaction Feasibility** → We adapt to **Property Consistency**
- **Judge Reasoning** → We adapt to **Causal PSP Reasoning**

### Why These Metrics for 2D Materials?

1. **Processing Feasibility** (40%): Most critical - impossible conditions invalidate everything
2. **Structure Emergence** (30%): Core challenge - defect engineering is central to 2D materials
3. **Property Consistency** (20%): Validation - properties must follow from structure
4. **PSP Reasoning** (10%): Interpretability - ensures predictions are explainable

### Comparison to Phase I Metrics

| Phase I (Generic) | Phase II (Domain-Specific) | Rationale |
|-------------------|---------------------------|-----------|
| Scientific Accuracy | Processing Feasibility | Focuses on thermodynamic/kinetic correctness, not just terminology |
| Functional Equivalence | Structure Emergence | Evaluates specific structural predictions (defects, strain) |
| Completeness | Property Consistency | Checks property-structure coherence, not just field coverage |
| Reasoning Quality | Causal PSP Reasoning | Requires mechanistic P→S→P chains, not generic explanations |
| Interpretability | *(Integrated)* | Transparency now embedded in reasoning metric |

---

## 🎓 Judge Design Decisions

### Why Ollama (Not GPT)?

- **Local inference:** No API costs, no rate limits
- **Reproducibility:** Deterministic with temperature=0.0
- **Privacy:** Research data stays local
- **Customization:** Can fine-tune judge models if needed

### Why Structured Rubrics?

- **Consistency:** Same criteria applied to all predictions
- **Transparency:** Clear explanation for every score
- **Debuggability:** Failure modes explicitly identified
- **Comparability:** Scores normalized to 0-100 scale

### Temperature = 0.0

We use deterministic generation for evaluation to ensure:
- Reproducible scores across runs
- No stochastic variation in judgments
- Fair comparison between systems

---

## 🚀 Next Steps (Phase II-C & II-D)

### Phase II-C: Extend Evaluation Scripts

**Tasks:**
- [ ] Create `run_phase2_evaluation.py` for 6-variant assessment
- [ ] Integrate LLM judge into evaluation pipeline
- [ ] Generate NEW_TABLE_2 with actual results
- [ ] Calculate component contributions (ablation analysis)
- [ ] Add backward compatibility with Phase I

### Phase II-D: Sensitivity Testing

**Tasks:**
- [ ] Design perturbation framework
  - Physical inconsistency (invalid parameters)
  - Broken PSP chains (missing causal links)
  - Fluent-but-invalid (plausible nonsense)
  - Paraphrase robustness (synonym queries)
- [ ] Implement perturbation generators
- [ ] Run sensitivity analysis
- [ ] Generate sensitivity report

---

## 📖 Usage Examples

### Example 1: Evaluate CVD MoS2 Prediction

```python
from src.evaluation.ollama_judge import OllamaJudge

judge = OllamaJudge()

query = {
    'task': 'forward_prediction',
    'processing': {
        'method': 'Chemical Vapor Deposition',
        'temperature': '800°C',
        'time': '30 min',
        'atmosphere': 'H2/Ar (1:10)',
        'precursor': 'MoO3 + S powder'
    }
}

prediction = {
    'structure': {
        'material': 'MoS2 monolayer',
        'defects': 'Sulfur vacancies (Vs)',
        'defect_density': '~10^12 cm^-2',
        'strain': 'Minimal (<0.5%)'
    },
    'properties': {
        'band_gap': '1.8 eV (direct gap)',
        'carrier_type': 'n-type',
        'mobility': '~50 cm^2/Vs',
        'PL_peak': '~680 nm'
    },
    'reasoning': 'CVD at 800°C provides sufficient energy for MoO3 reduction. '
                 'H2 atmosphere creates reducing conditions, leading to sulfur '
                 'vacancies that act as electron donors (n-type doping). '
                 'Direct band gap of 1.8 eV is characteristic of monolayer MoS2.'
}

ground_truth = {
    'structure': {
        'material': 'MoS2',
        'defects': 'sulfur vacancies',
        'defect_density': '10^12-10^13 cm^-2'
    },
    'properties': {
        'band_gap': '1.8 eV',
        'carrier_type': 'n-type'
    }
}

result = judge.evaluate_all_metrics(query, prediction, ground_truth)
```

Expected output:
```
Processing Feasibility: 38/40 (realistic conditions, appropriate atmosphere)
Structure Emergence: 28/30 (correct defect type and density)
Property Consistency: 18/20 (properties match structure)
Causal PSP Reasoning: 9/10 (clear P→S→P chain with mechanisms)
Overall: 93/100
```

### Example 2: Batch Evaluate Test Set

```python
import json
from pathlib import Path
from src.evaluation.ollama_judge import OllamaJudge, create_judge_report

# Load test set
with open('data/KG/outputs/combined_doping_data.json') as f:
    kg_data = json.load(f)

# Prepare test cases (simplified)
test_cases = []
for i, rel in enumerate(kg_data['causal_relationships'][:5]):
    test_cases.append({
        'id': f'test_{i}',
        'query': {'processing': rel['cause_parameter']},
        'prediction': {...},  # Get from model
        'ground_truth': rel
    })

# Evaluate
judge = OllamaJudge()
results = judge.batch_evaluate(test_cases)

# Save report
create_judge_report(results, 'results/phase2/judge_reports/batch_eval.txt')
```

---

## 📚 References

1. **Chemical Reactions Paper:** `references/arXiv-2512.13668v1/`
   - Source of LLM-as-a-Judge methodology
   - Inspiration for metric design

2. **Phase I Documentation:** `docs/EVALUATION_FRAMEWORK_PLAN.md`
   - Original evaluation plan
   - Phase I metrics and reproduction targets

3. **ARIA Variants:** See `README.md` for full system descriptions

---

## ✅ Deliverables Checklist

### Phase II-A: Design Metrics ✅
- [x] NEW_TABLE_1_metrics.tex - Metric definitions
- [x] NEW_TABLE_2_results.tex - Results schema template
- [x] Domain-specific rubrics for 2D materials

### Phase II-B: Implement Judge ✅
- [x] ollama_judge.py - LLM judge implementation
- [x] Judge prompt templates with detailed rubrics
- [x] Structured JSON output format
- [x] Batch evaluation support
- [x] Report generation

### Phase II-C: Extend Evaluation Scripts ⏳
- [ ] run_phase2_evaluation.py - Full 6-variant evaluation
- [ ] Integration with existing variants
- [ ] NEW_TABLE_2 generation with actual data
- [ ] Component contribution analysis

### Phase II-D: Sensitivity Testing ⏳
- [ ] Perturbation framework design
- [ ] Sensitivity analysis implementation
- [ ] Sensitivity report generation

---

**Last Updated:** 2026-02-02
**Status:** Phase II-A & II-B Complete | Phase II-C & II-D Pending
