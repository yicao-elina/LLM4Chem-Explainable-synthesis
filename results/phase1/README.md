# Phase I Evaluation Results

**Status:** ✅ Complete
**Date:** 2026-02-03
**Evaluations:** 160 (4 systems × 2 tasks × 2 domains × 10 tests)

---

## 📁 Available Files

### Clean Copies (No OneDrive Attributes - Recommended)

**Located in:** `/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD/results/phase1/`

1. **`phase1_raw_results_clean.csv`** (22KB)
   - All 160 individual evaluations
   - Columns: system, domain, task, test_id, scientific_accuracy, functional_equivalence, reasoning_quality, completeness, interpretability, overall

2. **`tables/phase1_summary_clean.csv`** (1.1KB)
   - Grouped results by system/domain/task
   - Average metrics for each configuration

3. **`tables/phase1_domain_gaps_clean.csv`** (940B)
   - In-Domain minus Out-of-Domain gaps
   - Shows generalization performance

4. **`tables/reproduced_main_table_clean.tex`** (1.6KB)
   - LaTeX table for publication
   - Matches format of target main_table.tex

### Original Files with Symlinks

- `phase1_raw_results_latest.csv` → `phase1_raw_results_20260203_013847.csv`
- `tables/phase1_summary_latest.csv` → `phase1_summary_20260203_013847.csv`
- `tables/phase1_domain_gaps_latest.csv` → `phase1_domain_gaps_20260203_013847.csv`
- `tables/reproduced_main_table_latest.tex` → `reproduced_main_table_20260203_013847.tex`

---

## 🚀 Quick Access

### Option 1: Interactive Viewer (Recommended)

```bash
cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD
./scripts/view_phase1_results.sh
```

Features:
- View all results interactively
- Export to Desktop
- Formatted display

### Option 2: Direct File Access

```bash
# View summary table
cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD/results/phase1
column -t -s, tables/phase1_summary_clean.csv

# View domain gaps
column -t -s, tables/phase1_domain_gaps_clean.csv

# View raw results (first 20)
head -20 phase1_raw_results_clean.csv | column -t -s,

# View LaTeX table
cat tables/reproduced_main_table_clean.tex
```

### Option 3: Open in Excel/Numbers

```bash
# Export to Desktop
cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD/results/phase1
cp phase1_raw_results_clean.csv ~/Desktop/
cp tables/phase1_summary_clean.csv ~/Desktop/
cp tables/phase1_domain_gaps_clean.csv ~/Desktop/

# Then open from Desktop with your preferred app
```

### Option 4: Python/Pandas

```python
import pandas as pd

# Load results
raw_results = pd.read_csv('results/phase1/phase1_raw_results_clean.csv')
summary = pd.read_csv('results/phase1/tables/phase1_summary_clean.csv')
gaps = pd.read_csv('results/phase1/tables/phase1_domain_gaps_clean.csv')

# View
print(summary)
print(gaps)
```

---

## 📊 Summary Statistics

### Best Performers (In-Domain)

**Forward Prediction:**
- 🥇 ARIA & Online KG+LLM (tied at 0.62 overall)
- ARIA: High completeness (0.93), interpretability (0.92)
- Online KG+LLM: High scientific accuracy (0.61), functional equivalence (0.55)

**Inverse Design:**
- 🥇 ARIA (0.53 overall)
- High interpretability (0.92), completeness (0.80)

### Generalization (Domain Gap)

**Best (Lowest Gap):**
- Naive KG+LLM: 0% gap (but low absolute performance)
- ARIA: 2-4% gap (good balance)

**Surprise:**
- Baseline LLM: -4% gap (actually better on OOD!)

### Critical Issue Found

**Naive KG+LLM:**
- Reasoning Quality = 0.00 across ALL tests
- Likely missing reasoning output in predictions
- Needs investigation of variant implementation

---

## 📈 Data Structure

### Raw Results (phase1_raw_results_clean.csv)

```
system,domain,task,test_id,scientific_accuracy,functional_equivalence,reasoning_quality,completeness,interpretability,overall
ARIA,in-domain,forward,in-domain_0,0.58,0.44,0.21,0.93,0.92,0.62
...
```

160 rows (10 tests × 4 systems × 2 tasks × 2 domains)

### Summary Table (phase1_summary_clean.csv)

```
system,domain,task,scientific_accuracy,functional_equivalence,reasoning_quality,completeness,interpretability,overall
ARIA,in-domain,forward,0.57,0.42,0.33,0.93,0.92,0.64
ARIA,in-domain,inverse,0.40,0.29,0.24,0.80,0.92,0.53
ARIA,out-of-domain,forward,0.41,0.41,0.31,0.93,0.92,0.60
ARIA,out-of-domain,inverse,0.27,0.26,0.27,0.80,0.92,0.51
...
```

16 rows (4 systems × 2 tasks × 2 domains)

### Domain Gaps (phase1_domain_gaps_clean.csv)

```
system,task,scientific_accuracy_gap,functional_equivalence_gap,reasoning_quality_gap,completeness_gap,interpretability_gap,overall_gap
ARIA,forward,16.0,1.0,-8.0,0.0,0.0,4.0
ARIA,inverse,13.0,3.0,-3.0,0.0,0.0,2.0
...
```

8 rows (4 systems × 2 tasks)

---

## 🔧 Troubleshooting

### "File cannot be opened" or "Permission denied"

**Cause:** OneDrive extended attributes

**Solution:** Use the `*_clean.csv` versions which have attributes removed

### "Symlink broken"

**Cause:** Incorrect symlink paths (fixed)

**Solution:** Symlinks now corrected to point to actual files in same directory

### "Excel shows garbled data"

**Cause:** CSV encoding or OneDrive sync issues

**Solution:**
1. Use `*_clean.csv` versions
2. Export to Desktop first
3. Open from Desktop (not from OneDrive folder)

---

## 📝 File Locations

```
results/phase1/
├── phase1_raw_results_clean.csv ✅ USE THIS
├── phase1_raw_results_latest.csv (symlink)
├── phase1_raw_results_20260203_013847.csv (timestamped original)
│
└── tables/
    ├── phase1_summary_clean.csv ✅ USE THIS
    ├── phase1_summary_latest.csv (symlink, now fixed)
    ├── phase1_summary_20260203_013847.csv (timestamped original)
    │
    ├── phase1_domain_gaps_clean.csv ✅ USE THIS
    ├── phase1_domain_gaps_latest.csv (symlink, now fixed)
    ├── phase1_domain_gaps_20260203_013847.csv (timestamped original)
    │
    ├── reproduced_main_table_clean.tex ✅ USE THIS
    ├── reproduced_main_table_latest.tex (symlink, now fixed)
    └── reproduced_main_table_20260203_013847.tex (timestamped original)
```

---

## ✅ What's Fixed

1. ✅ Symlinks corrected (were pointing to wrong paths)
2. ✅ Clean copies created without OneDrive attributes
3. ✅ Interactive viewer script added
4. ✅ All files verified readable
5. ✅ This README created for easy reference

---

**Last Updated:** 2026-02-03
**Next Steps:** Use clean files or viewer script to analyze results
