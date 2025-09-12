# Tests

This directory contains preliminary **test cases** and **scripts** for evaluating the LLM engine against gold-standard results. It aims to serve as demo for LLM Hackathon for Materials and Chemistry 2025, and how you may use the engine along with `run_eval.py` to evaluate the performance of the engine against known test cases of your choice.

---

## 📂 Contents
- **CASE\_xxx YAML files** → describe case studies and their corresponding aliases  
- **aliases.yaml** → maps shorthand names to full forms  
- **run_eval.py** → main test runner that loads cases, executes the engine, and compares outputs  
- **results/benchmark_results.csv** → generated file containing evaluation metrics

---

## ▶️ How to run the tests

1. Activate Python environment from the project root after cloning the repo:
   
   `conda activate causalmat`   # or whichever environment you use

2. Run the evaluation script:

   `python tests/run_eval.py`

3. Check the output in:

   `results/benchmark_results.csv`

📊 Example output

1. The `preds/output_message_CASE002.txt` file demonstrate the output message of running `run_eval.py` against CASE002. 

2. The CSV file includes metrics such as:

Case	Edge F1	Path F1	Inverse@3

CASE001	0.0	0.0	False

(numbers above are placeholders — real values depend on the engine and gold standards used)

🛠️ Adding new tests and modifying gold standards

- Place new case definitions under `tests/cases` as YAML files.
- Modify/ add new gold standards file (`CASE{number}_edges.json`, `CASE{number}_paths.json`, `CASE{number}_receipe.json`).
- (optional) Update aliases.yaml if new shorthand mappings are required. We plan to update normalization method using LLM moving forward.
- Run `run_eval.py` again to include them in evaluation.

📝 Notes

- Commit only YAML definitions and scripts, not large CSV result files for current version.
- Ensure the engine and dependencies are installed before running tests.
- The tests are designed to check:
  - Edge-level F1 (correctness of causal links)
  - Path-level accuracy (causal reasoning chains)
  - Inverse design hit rate (whether target properties can be designed)
