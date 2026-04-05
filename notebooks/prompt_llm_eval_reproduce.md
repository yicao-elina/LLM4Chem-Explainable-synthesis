
## 📌 Task: Evaluation Framework Adaptation & Reproduction for 2D Materials Processing Reasoning

---

## 🔍 Background & Source Files

This task integrates **three distinct evaluation layers** that must all be handled correctly.

### 1️⃣ Reference Evaluation Framework (Chemical Reactions)

* Evaluation methodology:
  `@references/arXiv-2512.13668v1/evaluation.tex`
* Reference metric definitions (Table 1):
  `@references/arXiv-2512.13668v1/table1.tex`
* Reference result table (Table 2):
  `@references/arXiv-2512.13668v1/table2.tex`

These define the **design philosophy** and **judge logic**, but are **not directly reusable** for 2D materials.

---

### 2️⃣ Our Existing Results (Must Be Reproduced First)

* Previous result table (ground truth reproduction target):
  `@results/previous_results/main_table.tex`

This table defines:

* The **systems being compared**
* The **metric layout**
* The **in-domain vs out-of-domain analysis**
* The **aggregation logic**

⚠️ **This table must be faithfully reproduced using our current LLM variants before any new metric redesign is performed.**

---

### 3️⃣ Our Current Evaluation Code & Data

* Existing evaluation scripts:
  `@src/evaluation`
* In-Domain ground truth:
  `@KG/outputs/combined_doping_data.json`
* Out-of-Domain ground truth:
  `@KG/outputs/test_doping_data.json`

---

## 🎯 Overall Objectives (Strict Order)

### Phase I — Reproduce Existing Results (No Metric Changes)

1. Audit `@src/evaluation`
2. Ensure all metrics used in
   `@results/previous_results/main_table.tex`
   are:

   * Clearly defined
   * Correctly implemented
3. Re-run evaluation using **current LLM variants**
4. Reproduce `main_table.tex` numerically and structurally

📌 **No metric redefinition is allowed in Phase I.**
This phase validates correctness and reproducibility.

---

### Phase II — Design New Evaluation Tables (Reference-Guided)

Using the **reference paper’s evaluation philosophy**, design **new Table 1 and Table 2** tailored to **2D materials processing reasoning**.

---

## 🧠 Domain Definitions (Must Be Used Exactly)

### In-Domain (ID)

* Materials with **established literature precedent**
* Fully covered by the Knowledge Graph
* Complete Processing–Structure–Property (PSP) chains exist
* Ground truth:
  `@KG/outputs/combined_doping_data.json`

### Out-of-Domain (OOD)

* **Novel compositions or processing routes**
* Post-date KG construction
* Partial or missing PSP chains
* Ground truth:
  `@KG/outputs/test_doping_data.json`

Strict ID/OOD separation is mandatory.

---

## 🧩 Required Deliverables

---

### 1️⃣ Phase I Output — Reproduction Report

* Confirm whether `@src/evaluation`:

  * Correctly defines each metric
  * Properly handles ID/OOD splits
* If discrepancies exist:

  * Fix implementation bugs
  * Do **not** redefine metrics
* Reproduce:

  * `@results/previous_results/main_table.tex`

📄 Output:

* Reproduced LaTeX table
* Brief validation notes

---

### 2️⃣ Phase II-A — New Metric Design (Reference Table 1 → 2D Materials)

* Analyze reference `table1.tex`
* Redesign metrics to explicitly evaluate:

  * Processing feasibility (thermodynamics, kinetics)
  * Structure emergence (defects, strain, stacking)
  * Property consistency
  * Causal PSP reasoning
  * Physical constraint awareness
* If a reference metric is unsuitable:

  * Replace it with a stronger 2D-material-specific metric

📄 Output:

* New **Table 1** LaTeX template
  (2D materials processing metrics)

---

### 3️⃣ Phase II-B — New Result Table Design (Reference Table 2 → 2D Materials)

* Analyze reference `table2.tex`
* Design a new **Table 2** that:

  * Uses redesigned metrics
  * Supports:

    * System × Domain × Metric
    * Domain gap analysis
    * Multi-model comparison

📄 Output:

* New **Table 2** LaTeX template

---

### 4️⃣ LLM-as-a-Judge Design (Ollama)

* Implement a judge using **Ollama**
* Follow the **reasoning-judge design** in the reference paper
* The judge must:

  * Score each metric independently
  * Produce structured JSON output
  * Provide justification + failure mode

📄 Output:

* Judge prompt template
* Output schema
* Explanation of design choices

---

### 5️⃣ Evaluation Script Extension

* Extend `@src/evaluation` to:

  * Support new metrics
  * Preserve backward compatibility with Phase I
* Metrics must be:

  * Modular
  * Explicitly mapped to judge prompts

📄 Output:

* Updated Python evaluation scripts

---

### 6️⃣ Sensitivity & Robustness Testing

Design perturbations to probe:

* Physical inconsistency
* Broken PSP chains
* Fluent but causally invalid reasoning
* Paraphrase robustness

Define:

* Perturbation types
* Expected metric behavior
* Sensitivity reporting

📄 Output:

* Sensitivity testing protocol

---

## ⚙️ Technical Constraints

* All LLM calls via **Ollama**
* Local inference only
* Deterministic settings where possible
* Metric definitions must be operationalizable

---

## ✅ Final Checklist

The agent must deliver:

1. ✅ Reproduced `main_table.tex`
2. ✅ New Table 1 (2D materials metrics)
3. ✅ New Table 2 (evaluation results schema)
4. ✅ Audited & extended evaluation scripts
5. ✅ Ollama judge design
6. ✅ Sensitivity testing methodology

