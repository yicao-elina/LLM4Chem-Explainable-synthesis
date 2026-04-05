#!/bin/bash

# Phase II-D Sensitivity Testing Runner Script
# Tests robustness to perturbations

set -e  # Exit on error

# Configuration
PROJECT_ROOT="/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD"
OUTPUT_DIR="${PROJECT_ROOT}/results/phase2/sensitivity"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         ARIA Phase II-D: Sensitivity & Robustness Testing     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Activate conda environment
echo -e "${YELLOW}[1/4] Activating conda environment...${NC}"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate causalmat
echo -e "${GREEN}✓ Environment activated: causalmat${NC}"
echo ""

# Change to project directory
cd "${PROJECT_ROOT}"
echo -e "${YELLOW}[2/4] Working directory: ${PROJECT_ROOT}${NC}"
echo ""

# Create output directory
echo -e "${YELLOW}[3/4] Creating output directory...${NC}"
mkdir -p "${OUTPUT_DIR}/reports"
mkdir -p "${OUTPUT_DIR}/logs"
echo -e "${GREEN}✓ Output directory: ${OUTPUT_DIR}${NC}"
echo ""

# Run sensitivity testing
echo -e "${YELLOW}[4/4] Running sensitivity testing...${NC}"
echo -e "${BLUE}Testing 4 perturbation types:${NC}"
echo -e "  1. Physical Inconsistency"
echo -e "  2. Broken PSP Chains"
echo -e "  3. Fluent-but-Invalid"
echo -e "  4. Paraphrase Robustness"
echo ""

LOG_FILE="${OUTPUT_DIR}/logs/sensitivity_${TIMESTAMP}.log"

# Create Python script to run sensitivity testing
cat > /tmp/run_sensitivity.py << 'EOPYTHON'
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from src.evaluation.sensitivity_testing import (
    SensitivityTester,
    generate_sensitivity_report
)
import json

# Load test data
with open('data/KG/outputs/combined_doping_data.json') as f:
    kg_data = json.load(f)

# Prepare test cases
relationships = kg_data.get('causal_relationships', [])[:5]  # First 5 for testing
test_cases = []
for i, rel in enumerate(relationships):
    test_cases.append({
        'id': f'test_{i}',
        'synthesis_conditions': {
            'method': rel.get('method', 'CVD'),
            'temperature': rel.get('temperature', '800°C'),
            'atmosphere': rel.get('atmosphere', 'H2/Ar')
        },
        'ground_truth_properties': {
            'band_gap': rel.get('band_gap', '1.8 eV'),
            'carrier_type': rel.get('carrier_type', 'n-type')
        },
        'ground_truth_synthesis': {
            'method': rel.get('method', 'CVD'),
            'temperature': rel.get('temperature', '800°C')
        }
    })

print(f"Loaded {len(test_cases)} test cases")

# Initialize tester
tester = SensitivityTester(judge_model="qwen2:7b")

# Generate perturbations
print("Generating perturbation suite...")
perturbed = tester.generate_perturbation_suite(test_cases)
print(f"Generated {len(perturbed)} perturbed cases")

# Evaluate
print("\nEvaluating perturbations with LLM judge...")
results_df = tester.evaluate_perturbation_suite(perturbed)

# Save results
results_df.to_csv('results/phase2/sensitivity/sensitivity_results.csv', index=False)
print(f"Results saved to: results/phase2/sensitivity/sensitivity_results.csv")

# Analyze
print("\nAnalyzing sensitivity...")
analysis = tester.analyze_sensitivity(results_df)

# Generate report
generate_sensitivity_report(
    results_df,
    analysis,
    'results/phase2/sensitivity/reports/sensitivity_report.txt'
)

print("\nSensitivity testing complete!")
EOPYTHON

python /tmp/run_sensitivity.py 2>&1 | tee "${LOG_FILE}"

# Check if completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Sensitivity testing completed successfully!${NC}"
else
    echo ""
    echo -e "${RED}✗ Sensitivity testing failed. Check log: ${LOG_FILE}${NC}"
    exit 1
fi

# Cleanup
rm /tmp/run_sensitivity.py

# Summary
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   SENSITIVITY TESTING COMPLETE                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Output Files:${NC}"
echo -e "  📊 Results:     ${OUTPUT_DIR}/sensitivity_results.csv"
echo -e "  📄 Report:      ${OUTPUT_DIR}/reports/sensitivity_report.txt"
echo -e "  📝 Log:         ${LOG_FILE}"
echo ""

# Display results summary
if [ -f "${OUTPUT_DIR}/sensitivity_results.csv" ]; then
    echo -e "${YELLOW}Results Summary:${NC}"
    echo ""
    head -20 "${OUTPUT_DIR}/sensitivity_results.csv" | column -t -s,
    echo ""
fi

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Phase II-D Complete!                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
