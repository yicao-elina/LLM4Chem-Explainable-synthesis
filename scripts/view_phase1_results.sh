#!/bin/bash

# Phase I Results Viewer
# Easy access to all Phase I evaluation results

PROJECT_ROOT="/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD"
PHASE1_DIR="${PROJECT_ROOT}/results/phase1"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              ARIA Phase I Evaluation Results                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if results exist
if [ ! -d "$PHASE1_DIR" ]; then
    echo -e "${RED}Error: Phase I results directory not found!${NC}"
    exit 1
fi

cd "$PHASE1_DIR"

echo -e "${YELLOW}Available Files:${NC}"
echo ""
echo "1. phase1_raw_results_clean.csv (22KB - All 160 evaluations)"
echo "2. tables/phase1_summary_clean.csv (1.1KB - Grouped results)"
echo "3. tables/phase1_domain_gaps_clean.csv (940B - ID-OOD gaps)"
echo "4. tables/reproduced_main_table_clean.tex (1.6KB - LaTeX table)"
echo ""

# Function to view file
view_file() {
    local file=$1
    local description=$2

    echo -e "${GREEN}=== $description ===${NC}"
    echo ""

    if [[ $file == *.tex ]]; then
        cat "$file"
    else
        # For CSV, show formatted with column
        head -20 "$file" | column -t -s,
    fi

    echo ""
    echo -e "${YELLOW}Full file: $file${NC}"
    echo ""
}

# Main menu
while true; do
    echo -e "${BLUE}Select option:${NC}"
    echo "1) View Summary Table"
    echo "2) View Domain Gaps"
    echo "3) View Raw Results (first 20 rows)"
    echo "4) View LaTeX Table"
    echo "5) Show File Paths"
    echo "6) Export to Desktop"
    echo "q) Quit"
    echo ""
    read -p "Choice: " choice

    case $choice in
        1)
            view_file "tables/phase1_summary_clean.csv" "Summary Table (Grouped by System/Domain/Task)"
            ;;
        2)
            view_file "tables/phase1_domain_gaps_clean.csv" "Domain Gaps (In-Domain - Out-of-Domain)"
            ;;
        3)
            view_file "phase1_raw_results_clean.csv" "Raw Results (First 20 rows)"
            ;;
        4)
            view_file "tables/reproduced_main_table_clean.tex" "LaTeX Table for Publication"
            ;;
        5)
            echo -e "${GREEN}=== File Paths ===${NC}"
            echo ""
            echo "Raw Results:"
            echo "  $PHASE1_DIR/phase1_raw_results_clean.csv"
            echo ""
            echo "Summary Table:"
            echo "  $PHASE1_DIR/tables/phase1_summary_clean.csv"
            echo ""
            echo "Domain Gaps:"
            echo "  $PHASE1_DIR/tables/phase1_domain_gaps_clean.csv"
            echo ""
            echo "LaTeX Table:"
            echo "  $PHASE1_DIR/tables/reproduced_main_table_clean.tex"
            echo ""
            ;;
        6)
            echo -e "${YELLOW}Exporting to Desktop...${NC}"
            cp phase1_raw_results_clean.csv ~/Desktop/
            cp tables/phase1_summary_clean.csv ~/Desktop/
            cp tables/phase1_domain_gaps_clean.csv ~/Desktop/
            cp tables/reproduced_main_table_clean.tex ~/Desktop/
            echo -e "${GREEN}✓ Files exported to Desktop${NC}"
            echo ""
            ;;
        q|Q)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice"
            ;;
    esac
done
