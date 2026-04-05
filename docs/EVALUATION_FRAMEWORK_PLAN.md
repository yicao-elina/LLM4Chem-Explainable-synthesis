# ARIA Evaluation Framework: Implementation Plan

**Date:** 2026-02-02
**Status:** Phase I In Progress
**Goal:** Reproduce existing metrics, then design 2D materials-specific evaluation

---

## 📋 Task Overview

### Primary Objectives
1. **Phase I:** Reproduce existing evaluation results (NO metric changes)
2. **Phase II:** Design new 2D materials-specific evaluation framework

### Reference Documents
- **Reference Paper:** `references/arXiv-2512.13668v1/` (chemical reactions)
  - `evaluation.tex` - Evaluation methodology
  - `table1.tex` - Metric definitions
  - `table2.tex` - Results schema
- **Target to Reproduce:** `results/previous_results/main_table.tex`
- **Current Code:** `src/evaluation/evaluation.py`
- **Data:**
  - In-Domain: `data/KG/outputs/combined_doping_data.json` (777 nodes, 421 edges)
  - Out-of-Domain: `data/KG/outputs/test_doping_data.json`

---

## 🎯 Phase I: Reproduction (Current Task)

### Goal
Reproduce `results/previous_results/main_table.tex` numerically and structurally using current 6 variants.

### Target Metrics (From main_table.tex)

1. **Scientific Accuracy** (0-1 scale)
   - How scientifically correct is the prediction?
   - Measures factual correctness of materials science knowledge

2. **Functional Equivalence** (0-1 scale)
   - Does the predicted outcome achieve the same function as ground truth?
   - Semantic similarity of functional outcomes

3. **Reasoning Quality** (0-1 scale)
   - Quality of the reasoning chain provided
   - Coherence, logic, scientific justification

4. **Completeness** (0-1 scale)
   - How complete is the answer?
   - Coverage of all relevant aspects (processing, structure, properties)

5. **Interpretability** (0-1 scale)
   - How easy is it to understand the output?
   - Clarity, structure, transparency

### Systems to Evaluate

**Previous (4 systems):**
1. Baseline LLM
2. Naive KG+LLM
3. Online KG+LLM
4. ARIA

**Current (6 systems - need mapping):**
1. **BASELINE** → Baseline LLM
2. **KG_ONLY** → (Pure graph, may not have direct equivalent)
3. **NAIVE_KG** → Naive KG+LLM
4. **ARIA_CORE** → ARIA (partial)
5. **ARIA_SEARCH** → Online KG+LLM (with search)
6. **ARIA_FULL** → ARIA (full with CoT)

**Mapping Strategy:**
- Use **BASELINE** for "Baseline LLM"
- Use **NAIVE_KG** for "Naive KG+LLM"
- Use **ARIA_SEARCH** for "Online KG+LLM" (closest match with literature search)
- Use **ARIA_FULL** for "ARIA" (most complete system with CoT)
- *Optional:* Include ARIA_CORE and KG_ONLY as additional ablation points

### Tasks × Domains

**Tasks:**
1. Forward Prediction (synthesis → properties)
2. Inverse Design (properties → synthesis)

**Domains:**
1. In-Domain (ID): Materials/protocols in KG
2. Out-of-Domain (OOD): Novel materials/protocols not in KG

**Total:** 2 tasks × 2 domains × 6 systems = 24 configurations

---

## 🔍 Analysis of Current evaluation.py

### What It Does

**Current Metrics:**
- `dag_confidence`: Confidence score from model
- `dag_similarity`: Semantic similarity to ground truth (cosine similarity)
- `baseline_similarity`: Baseline model similarity
- `dag_improvement`: Difference between DAG and baseline

**Current Approach:**
- Uses `SentenceTransformer` embeddings for semantic similarity
- Evaluates on perturbed test cases (0.0 to 0.9 perturbation)
- Compares DAG-LLM vs Baseline Gemini

### Gaps Identified

**Missing Metrics:**
1. ❌ **Scientific Accuracy** - Not explicitly measured
2. ❌ **Functional Equivalence** - Uses generic semantic similarity (not functional)
3. ❌ **Reasoning Quality** - Not evaluated
4. ❌ **Completeness** - Not measured
5. ❌ **Interpretability** - Not assessed

**Missing Features:**
- ❌ No In-Domain vs Out-of-Domain split
- ❌ No multi-system comparison (only DAG vs Baseline)
- ❌ No structured metric definitions
- ❌ Uses Gemini API (we need Ollama)

### Required Changes

**1. Implement 5 Missing Metrics**

Each metric needs:
- Clear definition
- Scoring function (0-1)
- Ground truth comparison method

**Approach Options:**

**Option A: Rule-Based Metrics**
- Scientific Accuracy: Check for scientific terms, units, concepts
- Functional Equivalence: Semantic similarity of functional outcomes
- Reasoning Quality: Check for causal keywords, mechanism explanations
- Completeness: Coverage of expected fields (processing/structure/properties)
- Interpretability: Structure analysis, readability metrics

**Option B: LLM-as-a-Judge (More Robust)**
- Use Ollama model to score each metric independently
- Provide clear rubric for each metric
- Get structured JSON output

**Recommendation:** Start with Option A for Phase I (faster, reproducible), then implement Option B for Phase II.

---

## 📊 Phase I Implementation Plan

### Step 1: Define Metric Functions

Create `src/evaluation/metrics.py`:

```python
def scientific_accuracy(predicted: dict, ground_truth: dict,
                       embedding_model) -> float:
    """
    Measure scientific correctness of prediction.
    - Extract scientific concepts, values, units
    - Compare to ground truth
    - Return 0-1 score
    """
    pass

def functional_equivalence(predicted: dict, ground_truth: dict,
                          embedding_model) -> float:
    """
    Measure functional similarity.
    - Focus on outcome/effect rather than process
    - Semantic similarity of functional descriptions
    """
    pass

def reasoning_quality(reasoning: str) -> float:
    """
    Measure quality of reasoning chain.
    - Presence of causal language
    - Mechanism explanations
    - Logical coherence
    """
    pass

def completeness(predicted: dict, expected_fields: list) -> float:
    """
    Measure completeness of answer.
    - Coverage of expected fields
    - Depth of each field
    """
    pass

def interpretability(output: dict) -> float:
    """
    Measure how easy to understand.
    - Structure clarity
    - Presence of explanations
    - Readability
    """
    pass
```

### Step 2: Implement Domain Split

Load both KG files:
- In-Domain: `data/KG/outputs/combined_doping_data.json`
- Out-of-Domain: `data/KG/outputs/test_doping_data.json`

Create test cases for each domain separately.

### Step 3: Update Evaluation Script

Modify `src/evaluation/evaluation.py`:

```python
def evaluate_system_comprehensive(
    system: Any,  # One of 6 variants
    test_cases: List[Dict],
    domain: str,  # 'in-domain' or 'out-of-domain'
    task: str,  # 'forward' or 'inverse'
    embedding_model: SentenceTransformer
) -> pd.DataFrame:
    """
    Evaluate system on all 5 metrics.

    Returns DataFrame with:
    - system_name
    - domain
    - task
    - scientific_accuracy
    - functional_equivalence
    - reasoning_quality
    - completeness
    - interpretability
    - overall (average)
    """
    pass
```

### Step 4: Run Evaluation on All Systems

```python
systems = {
    'Baseline LLM': BaselineOllama(),
    'Naive KG+LLM': NaiveKGOllama(),
    'Online KG+LLM': ARIASearchOllama(),
    'ARIA': ARIAFullOllama()
}

domains = ['in-domain', 'out-of-domain']
tasks = ['forward', 'inverse']

results = []
for system_name, system in systems.items():
    for domain in domains:
        for task in tasks:
            test_cases = load_test_cases(domain, task)
            result = evaluate_system_comprehensive(
                system, test_cases, domain, task, embedding_model
            )
            results.append(result)

results_df = pd.concat(results)
```

### Step 5: Generate LaTeX Table

```python
def generate_main_table_latex(results_df: pd.DataFrame) -> str:
    """
    Generate LaTeX table matching main_table.tex format.

    Structure:
    - Forward Prediction section
      - Each system with In-Domain and Out-of-Domain rows
      - Domain gap row
    - Performance Comparison section
    - Inverse Design section
      - Same structure as Forward
    """
    pass
```

### Step 6: Validate Reproduction

Compare generated table with `results/previous_results/main_table.tex`:
- Check metric values are in same range (0-1)
- Check domain gap patterns match
- Check performance comparison trends match

---

## 🎓 Phase II: New Evaluation Framework

### Phase II-A: Design 2D Materials-Specific Metrics

**Inspired by reference paper's chemical reactions metrics, design:**

1. **Processing Feasibility Score** (40 points)
   - Thermodynamic viability (temperature, pressure ranges)
   - Kinetic feasibility (time scales, reaction rates)
   - Equipment accessibility
   - Safety considerations

2. **Structure Emergence Score** (30 points)
   - Defect type predictions
   - Strain analysis
   - Stacking order
   - Crystal structure correctness

3. **Property Consistency Score** (20 points)
   - Predicted properties match expected from structure
   - Electronic properties (band gap, conductivity type)
   - Mechanical properties
   - Optical properties

4. **Causal PSP Reasoning Score** (10 points)
   - Processing→Structure→Property chain present
   - Mechanism explanations provided
   - Physical justifications given

**Additional Metrics:**

5. **Physical Constraint Awareness**
   - Obeys conservation laws
   - Respects symmetry requirements
   - Acknowledges limitations

6. **Novelty Handling**
   - Ability to reason about novel materials
   - Transfer learning quality
   - Uncertainty quantification

### Phase II-B: LLM-as-a-Judge Implementation

**Judge Design (Using Ollama):**

```python
class OllamaJudge:
    """
    LLM-based evaluator for 2D materials processing.
    """

    def __init__(self, model: str = "qwen2:7b"):
        self.ollama = get_ollama_client(model=model)

    def score_prediction(
        self,
        query: dict,
        prediction: dict,
        ground_truth: dict,
        metric: str
    ) -> dict:
        """
        Score prediction on specific metric.

        Returns:
        {
            "score": float (0-100),
            "justification": str,
            "failure_modes": List[str],
            "strengths": List[str]
        }
        """
        prompt = self._create_judge_prompt(
            query, prediction, ground_truth, metric
        )

        response = self.ollama.generate_json(prompt)
        return response
```

**Judge Prompt Template:**

```
You are an expert materials scientist evaluating AI-generated predictions.

**Query:**
{query_json}

**Ground Truth:**
{ground_truth_json}

**Model Prediction:**
{prediction_json}

**Metric:** {metric_name}
**Rubric:** {metric_rubric}

**Task:**
1. Compare the prediction to ground truth
2. Score the prediction on {metric_name} (0-100)
3. Provide justification
4. Identify failure modes (if any)
5. Identify strengths

**Output Format:**
```json
{
  "score": <float 0-100>,
  "justification": "<detailed explanation>",
  "failure_modes": ["<mode 1>", "<mode 2>"],
  "strengths": ["<strength 1>", "<strength 2>"]
}
```
```

### Phase II-C: New Result Tables

**New Table 1: Metric Definitions**

```latex
\begin{table}[h]
\caption{Evaluation Metrics for 2D Materials Processing Reasoning}
\label{tab:metrics_2d}
\begin{tabular}{lcp{8cm}}
\toprule
\textbf{Metric} & \textbf{Weight} & \textbf{Definition} \\
\midrule
Processing Feasibility & 40\% & Thermodynamic/kinetic viability... \\
Structure Emergence & 30\% & Defect prediction accuracy... \\
Property Consistency & 20\% & Electronic/mechanical properties... \\
Causal PSP Reasoning & 10\% & Processing→Structure→Property chain... \\
\bottomrule
\end{tabular}
\end{table}
```

**New Table 2: Results Schema**

```latex
\begin{table*}[ht]
\caption{6-Variant Ablation Study Results}
\label{tab:ablation_6variants}
\begin{tabular}{lcccccc}
\toprule
\textbf{System} & \textbf{Domain} & \textbf{Processing} & \textbf{Structure} & \textbf{Property} & \textbf{PSP} & \textbf{Overall} \\
& & \textbf{Feasibility} & \textbf{Emergence} & \textbf{Consistency} & \textbf{Reasoning} & \\
\midrule
\multicolumn{7}{c}{\textbf{Forward Prediction}} \\
\midrule
BASELINE & ID & ... & ... & ... & ... & ... \\
BASELINE & OOD & ... & ... & ... & ... & ... \\
KG_ONLY & ID & ... & ... & ... & ... & ... \\
KG_ONLY & OOD & ... & ... & ... & ... & ... \\
NAIVE_KG & ID & ... & ... & ... & ... & ... \\
NAIVE_KG & OOD & ... & ... & ... & ... & ... \\
ARIA_CORE & ID & ... & ... & ... & ... & ... \\
ARIA_CORE & OOD & ... & ... & ... & ... & ... \\
ARIA_SEARCH & ID & ... & ... & ... & ... & ... \\
ARIA_SEARCH & OOD & ... & ... & ... & ... & ... \\
ARIA_FULL & ID & ... & ... & ... & ... & ... \\
ARIA_FULL & OOD & ... & ... & ... & ... & ... \\
\midrule
\multicolumn{7}{c}{\textbf{Component Contributions}} \\
\midrule
KG Value (NAIVE - BASE) & ... \\
Tier Value (CORE - NAIVE) & ... \\
Search Value (SEARCH - CORE) & ... \\
CoT Value (FULL - SEARCH) & ... \\
\bottomrule
\end{tabular}
\end{table*}
```

### Phase II-D: Sensitivity Testing

**Perturbation Types:**

1. **Physical Inconsistency**
   - Invalid temperature ranges
   - Impossible pressure values
   - Contradictory conditions

2. **Broken PSP Chains**
   - Processing without expected structure outcome
   - Structure without property implications
   - Missing causal links

3. **Fluent but Invalid**
   - Grammatically correct but scientifically wrong
   - Plausible-sounding nonsense
   - Mixing incompatible concepts

4. **Paraphrase Robustness**
   - Same query, different wording
   - Synonyms for materials/methods
   - Unit conversions

**Expected Behavior:**
- Good systems should detect physical inconsistencies
- Should maintain reasoning even with broken chains (fill gaps)
- Should score lower on fluent-but-invalid
- Should be robust to paraphrasing

---

## 📈 Success Criteria

### Phase I Success
✅ Reproduced `main_table.tex` with <10% numerical deviation
✅ All 5 metrics implemented and validated
✅ In-Domain vs Out-of-Domain split working
✅ Evaluation runs on all 6 variants

### Phase II Success
✅ New metrics designed specifically for 2D materials
✅ Ollama judge implemented and validated
✅ New tables generated with 6-variant ablation results
✅ Sensitivity testing protocol defined and executed
✅ Results show clear component contributions

---

## 🗓️ Timeline

### Week 1 (Current)
- ✅ Task planning complete
- 🔄 Phase I: Metric implementation
- 🔄 Phase I: Evaluation script update
- 🔄 Phase I: Run and reproduce

### Week 2
- Phase II-A: Design new metrics
- Phase II-B: Implement Ollama judge
- Phase II-C: Create new tables

### Week 3
- Phase II-D: Sensitivity testing
- Documentation and paper writing

---

## 📝 Deliverables Checklist

### Phase I
- [ ] `src/evaluation/metrics.py` - 5 metric implementations
- [ ] `src/evaluation/evaluation_v2.py` - Updated evaluation script
- [ ] `results/phase2/reproduced_main_table.tex` - Reproduced table
- [ ] `results/phase2/reproduction_validation_report.md` - Validation notes

### Phase II
- [ ] `docs/evaluation/NEW_TABLE_1_metrics.tex` - New metric definitions
- [ ] `docs/evaluation/NEW_TABLE_2_results.tex` - New results schema
- [ ] `src/evaluation/ollama_judge.py` - LLM judge implementation
- [ ] `src/evaluation/sensitivity_testing.py` - Perturbation framework
- [ ] `results/phase2/6_variant_evaluation_results.csv` - Full results
- [ ] `results/phase2/sensitivity_analysis.pdf` - Sensitivity report

---

**Status:** Phase I In Progress
**Next Action:** Implement metrics.py with 5 core metrics
**Updated:** 2026-02-02
