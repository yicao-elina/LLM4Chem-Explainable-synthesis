## 📌 Task: Tailor Evaluation Framework from Chemical Reactions to 2D Materials Processing

### 🔍 Background & References

We are adapting the evaluation framework from the following reference paper:

* **Evaluation methodology**:
  `@references/arXiv-2512.13668v1/evaluation.tex`
* **Original evaluation metrics (Table 1)**:
  `@references/arXiv-2512.13668v1/table1.tex`
* **Original evaluation results (Table 2)**:
  `@references/arXiv-2512.13668v1/table2.tex`

The original framework is designed for **chemical reaction reasoning**.
Our goal is to **systematically adapt it to 2D materials processing and synthesis reasoning**.

---

## 🎯 High-Level Objective

Design a **new evaluation framework** where a **reasoning LLM (called via Ollama)** acts as a **judge** for 2D materials processing tasks, with:

1. **Updated, domain-specific evaluation metrics**
2. **A reproducible judging script**
3. **A sensitivity-testing protocol via controlled perturbations**

---

## 🧩 Required Deliverables

### 1️⃣ Metric Redesign (Table 1 Adaptation)

* Analyze the original metrics in `table1.tex`
* Modify and/or replace them so they are **well-aligned with 2D materials processing**, such as:

  * Growth / synthesis pathways (CVD, MBE, exfoliation, intercalation, etc.)
  * Processing–structure–property–performance reasoning
  * Defect formation, strain, stacking order, interfaces
  * Physical plausibility and thermodynamic consistency
* Clearly define **each updated metric**, including:

  * Metric name
  * What it measures
  * Why it is necessary for 2D materials reasoning
  * Expected failure modes

📄 **Output**:
A **new LaTeX table template** (updated Table 1) in `.tex` format, ready to drop into a paper.

---

### 2️⃣ Judge Implementation (Table 2 Evaluation Script)

* Write a **Python evaluation script** that:

  * Iterates over all metrics defined in the updated Table 1
  * Uses **Ollama** to call the **reasoning LLM described in the reference paper**
  * Treats the LLM as a **structured judge**, not a generator
* The script should:

  * Provide **explicit judging instructions** per metric
  * Enforce **structured outputs** (e.g., JSON with scores + justification)
  * Aggregate results in a format compatible with **Table 2**

📄 **Output**:

* Python evaluation script
* Example LLM judge prompt
* Example structured LLM output

---

### 3️⃣ Updated Result Table Template (Table 2)

* Redesign `table2.tex` so it:

  * Reflects the **new metrics**
  * Supports **per-metric scores**, optional confidence, and reasoning trace
  * Is extensible to multiple models / ablations

📄 **Output**:
Updated **Table 2 LaTeX template** in `.tex` format.

---

### 4️⃣ Judge Design Documentation

Provide a **clear methodological description** (paper-ready) covering:

* How the LLM judge is designed
* Why LLM-based judging is appropriate for 2D materials reasoning
* How prompts, constraints, and output schemas enforce consistency
* Known limitations and mitigation strategies (e.g., self-consistency, calibration)

📄 **Output**:
A concise **“Evaluation Method” section** suitable for inclusion in a methods appendix.

---

### 5️⃣ Sensitivity & Robustness Testing via Perturbations

Design a protocol to **intentionally perturb inputs** in order to probe judge sensitivity, including:

* Physically invalid processing steps
* Subtle thermodynamic inconsistencies
* Missing intermediate reasoning steps
* Overconfident but incorrect conclusions
* Equivalent processes phrased differently (paraphrase robustness)

Define:

* Perturbation types
* Expected metric responses
* How deviations are quantified

📄 **Output**:

* Perturbation taxonomy
* How perturbations are injected
* How judge sensitivity is measured and reported

---

## ⚙️ Technical Constraints

* Use **Ollama** for all LLM calls
* Assume **local inference**
* Evaluation must be:

  * Deterministic where possible
  * Reproducible
  * Modular (metrics are easy to add/remove)

---

## ✅ Final Output Summary

The final response should include:

1. Updated **Table 1 metrics** (`.tex`)
2. Updated **Table 2 template** (`.tex`)
3. **Python evaluation script** using Ollama
4. **Judge prompt + output schema**
5. **Evaluation methodology text**
6. **Sensitivity testing design**

