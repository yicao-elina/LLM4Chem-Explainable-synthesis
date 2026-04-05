#!/bin/bash

# Phase I Evaluation Runner Script
# Reproduces main_table.tex results using 4 ARIA variants
# Outputs: CSV results, LaTeX table, and raw model predictions

set -e  # Exit on error

# Configuration
PROJECT_ROOT="/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD"
OUTPUT_DIR="${PROJECT_ROOT}/results/phase1"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         ARIA Phase I Evaluation - Metric Reproduction         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Activate conda environment
echo -e "${YELLOW}[1/5] Activating conda environment...${NC}"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate causalmat
echo -e "${GREEN}✓ Environment activated: causalmat${NC}"
echo ""

# Change to project directory
cd "${PROJECT_ROOT}"
echo -e "${YELLOW}[2/5] Working directory: ${PROJECT_ROOT}${NC}"
echo ""

# Create output directory
echo -e "${YELLOW}[3/5] Creating output directory...${NC}"
mkdir -p "${OUTPUT_DIR}/raw_outputs"
mkdir -p "${OUTPUT_DIR}/tables"
mkdir -p "${OUTPUT_DIR}/logs"
echo -e "${GREEN}✓ Output directory: ${OUTPUT_DIR}${NC}"
echo ""

# Run evaluation
echo -e "${YELLOW}[4/5] Running Phase I evaluation...${NC}"
echo -e "${BLUE}This will evaluate 4 systems × 2 tasks × 2 domains × 10 test cases = 160 evaluations${NC}"
echo ""

LOG_FILE="${OUTPUT_DIR}/logs/evaluation_${TIMESTAMP}.log"

python src/evaluation/run_phase1_evaluation.py 2>&1 | tee "${LOG_FILE}"

# Check if evaluation completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Evaluation completed successfully!${NC}"
else
    echo ""
    echo -e "${RED}✗ Evaluation failed. Check log: ${LOG_FILE}${NC}"
    exit 1
fi

# Copy results to timestamped directory
echo ""
echo -e "${YELLOW}[5/5] Organizing results...${NC}"

# Move generated files from results/phase2 to results/phase1
if [ -d "results/phase2" ]; then
    cp results/phase2/phase1_raw_results.csv "${OUTPUT_DIR}/phase1_raw_results_${TIMESTAMP}.csv" 2>/dev/null || true
    cp results/phase2/phase1_summary_table.csv "${OUTPUT_DIR}/tables/phase1_summary_${TIMESTAMP}.csv" 2>/dev/null || true
    cp results/phase2/phase1_domain_gaps.csv "${OUTPUT_DIR}/tables/phase1_domain_gaps_${TIMESTAMP}.csv" 2>/dev/null || true
    cp results/phase2/reproduced_main_table.tex "${OUTPUT_DIR}/tables/reproduced_main_table_${TIMESTAMP}.tex" 2>/dev/null || true

    # Create symlinks to latest
    ln -sf "phase1_raw_results_${TIMESTAMP}.csv" "${OUTPUT_DIR}/phase1_raw_results_latest.csv"
    ln -sf "tables/phase1_summary_${TIMESTAMP}.csv" "${OUTPUT_DIR}/tables/phase1_summary_latest.csv"
    ln -sf "tables/phase1_domain_gaps_${TIMESTAMP}.csv" "${OUTPUT_DIR}/tables/phase1_domain_gaps_latest.csv"
    ln -sf "tables/reproduced_main_table_${TIMESTAMP}.tex" "${OUTPUT_DIR}/tables/reproduced_main_table_latest.tex"
fi

echo -e "${GREEN}✓ Results organized${NC}"
echo ""

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                      EVALUATION COMPLETE                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Output Files:${NC}"
echo -e "  📊 Raw Results:     ${OUTPUT_DIR}/phase1_raw_results_latest.csv"
echo -e "  📋 Summary Table:   ${OUTPUT_DIR}/tables/phase1_summary_latest.csv"
echo -e "  📉 Domain Gaps:     ${OUTPUT_DIR}/tables/phase1_domain_gaps_latest.csv"
echo -e "  📄 LaTeX Table:     ${OUTPUT_DIR}/tables/reproduced_main_table_latest.tex"
echo -e "  📝 Log File:        ${LOG_FILE}"
echo ""

# Display summary statistics
if [ -f "${OUTPUT_DIR}/tables/phase1_summary_latest.csv" ]; then
    echo -e "${YELLOW}Summary Statistics:${NC}"
    echo ""
    head -20 "${OUTPUT_DIR}/tables/phase1_summary_latest.csv" | column -t -s,
    echo ""
fi

# Display comparison with target
if [ -f "results/previous_results/main_table.tex" ]; then
    echo -e "${YELLOW}Comparison Target:${NC}"
    echo -e "  Original: results/previous_results/main_table.tex"
    echo -e "  Reproduced: ${OUTPUT_DIR}/tables/reproduced_main_table_latest.tex"
    echo ""
    echo -e "${BLUE}To compare, run:${NC}"
    echo -e "  diff results/previous_results/main_table.tex ${OUTPUT_DIR}/tables/reproduced_main_table_latest.tex"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Phase I Reproduction Done!                  ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
