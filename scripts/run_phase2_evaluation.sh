#!/bin/bash

# Phase II-C Evaluation Runner Script
# Full 6-variant evaluation with LLM-as-a-Judge

set -e  # Exit on error

# Configuration
PROJECT_ROOT="/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD"
OUTPUT_DIR="${PROJECT_ROOT}/results/phase2"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      ARIA Phase II-C: 6-Variant LLM-Judge Evaluation          ║${NC}"
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
mkdir -p "${OUTPUT_DIR}/judge_reports"
mkdir -p "${OUTPUT_DIR}/tables"
mkdir -p "${OUTPUT_DIR}/logs"
echo -e "${GREEN}✓ Output directory: ${OUTPUT_DIR}${NC}"
echo ""

# Warn about runtime
echo -e "${YELLOW}[4/5] Preparing to run evaluation...${NC}"
echo -e "${RED}⚠️  WARNING: This evaluation uses LLM-as-a-Judge${NC}"
echo -e "${RED}   Expected runtime: 2-4 hours${NC}"
echo -e "${RED}   (4 metrics × 6 systems × 2 tasks × 2 domains × 10 tests = 960 LLM calls)${NC}"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user"
    exit 1
fi
echo ""

# Run evaluation
echo -e "${YELLOW}[5/5] Running Phase II-C evaluation...${NC}"
echo -e "${BLUE}Evaluating 6 variants with domain-specific metrics${NC}"
echo ""

LOG_FILE="${OUTPUT_DIR}/logs/phase2_evaluation_${TIMESTAMP}.log"

python src/evaluation/run_phase2_evaluation.py 2>&1 | tee "${LOG_FILE}"

# Check if evaluation completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Evaluation completed successfully!${NC}"
else
    echo ""
    echo -e "${RED}✗ Evaluation failed. Check log: ${LOG_FILE}${NC}"
    exit 1
fi

# Organize results
echo ""
echo -e "${YELLOW}Organizing results...${NC}"

# Create symlinks to latest
if [ -f "${OUTPUT_DIR}/phase2_raw_results.csv" ]; then
    ln -sf phase2_raw_results.csv "${OUTPUT_DIR}/phase2_raw_results_latest.csv"
    ln -sf phase2_summary_table.csv "${OUTPUT_DIR}/phase2_summary_latest.csv"
    ln -sf phase2_component_contributions.csv "${OUTPUT_DIR}/phase2_contributions_latest.csv"
    ln -sf phase2_domain_gaps.csv "${OUTPUT_DIR}/phase2_gaps_latest.csv"
    ln -sf phase2_results_table.tex "${OUTPUT_DIR}/tables/phase2_results_latest.tex"
fi

echo -e "${GREEN}✓ Results organized${NC}"
echo ""

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                      EVALUATION COMPLETE                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Output Files:${NC}"
echo -e "  📊 Raw Results:         ${OUTPUT_DIR}/phase2_raw_results_latest.csv"
echo -e "  📋 Summary Table:       ${OUTPUT_DIR}/phase2_summary_latest.csv"
echo -e "  🧩 Component Contrib:   ${OUTPUT_DIR}/phase2_contributions_latest.csv"
echo -e "  📉 Domain Gaps:         ${OUTPUT_DIR}/phase2_gaps_latest.csv"
echo -e "  📄 LaTeX Table:         ${OUTPUT_DIR}/tables/phase2_results_latest.tex"
echo -e "  📝 Log File:            ${LOG_FILE}"
echo ""

# Display summary if available
if [ -f "${OUTPUT_DIR}/phase2_summary_latest.csv" ]; then
    echo -e "${YELLOW}Summary Statistics (first 20 rows):${NC}"
    echo ""
    head -20 "${OUTPUT_DIR}/phase2_summary_latest.csv" | column -t -s,
    echo ""
fi

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Phase II-C Complete!                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
