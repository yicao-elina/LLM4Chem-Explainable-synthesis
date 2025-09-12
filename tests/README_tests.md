# Tests

This directory contains **test cases** and **scripts** for evaluating the LLM engine against gold-standard results.

---

## 📂 Contents
- **CASE\_xxx YAML files** → describe case studies and their corresponding aliases  
- **aliases.yaml** → maps shorthand names to full forms  
- **run_eval.py** → main test runner that loads cases, executes the engine, and compares outputs  
- **results/benchmark_results.csv** → generated file containing evaluation metrics

---

## ▶️ How to Run the Tests

1. Activate your Python environment from the project root:
   
   conda activate causalmat   # or whichever environment you use

2. Run the evaluation script:

   python tests/run_eval.py

3. Check the output in:

   results/benchmark_results.csv

📊 Example Output

The CSV file includes metrics such as:

Case	Edge F1	Path F1	Inverse@3
CASE001	0.0	0.0	False

(numbers above are placeholders — real values depend on the engine and gold standards used)

🛠️ Adding New Tests

- Place new case definitions under tests/ as YAML files.
- Update aliases.yaml if new shorthand mappings are required.
- Run run_eval.py again to include them in evaluation.

📝 Notes

- Commit only YAML definitions and scripts, not large CSV result files.
- Ensure the engine and dependencies are installed before running tests.
- The tests are designed to check:
  - Edge-level F1 (correctness of causal links)
  - Path-level accuracy (causal reasoning chains)
  - Inverse design hit rate (whether target properties can be designed)
