#!/bin/bash

# ==============================================================================
# TEST PIPELINE: Fetch 50 papers → Extract with Ollama → Build KG → Visualize
# ==============================================================================

set -e  # Exit on error

echo "=========================================="
echo "2D Materials Doping KG Test Pipeline"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to KG directory
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"
echo ""

# ==============================================================================
# Step 0: Check Prerequisites
# ==============================================================================
echo "=========================================="
echo "Step 0: Checking Prerequisites"
echo "=========================================="

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}✗ Ollama not found${NC}"
    echo "  Install: brew install ollama"
    exit 1
fi
echo -e "${GREEN}✓ Ollama installed${NC}"

# Check if Ollama is running
if ! ollama list &> /dev/null; then
    echo -e "${YELLOW}⚠ Ollama not running. Starting...${NC}"
    ollama serve &
    OLLAMA_PID=$!
    sleep 3
else
    echo -e "${GREEN}✓ Ollama is running${NC}"
fi

# Check for qwen2:7b model
if ollama list | grep -q "qwen2:7b"; then
    echo -e "${GREEN}✓ Model qwen2:7b available${NC}"
else
    echo -e "${YELLOW}⚠ Model qwen2:7b not found. Downloading...${NC}"
    ollama pull qwen2:7b
fi

# Check Python packages
echo "Checking Python packages..."
python3 -c "import requests, networkx, matplotlib" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Required Python packages installed${NC}"
else
    echo -e "${RED}✗ Missing Python packages${NC}"
    echo "  Install: pip install requests networkx matplotlib"
    exit 1
fi

echo ""

# ==============================================================================
# Step 1: Fetch 50 Test Papers from OpenAlex
# ==============================================================================
echo "=========================================="
echo "Step 1: Fetching 50 Test Papers"
echo "=========================================="

START_TIME=$SECONDS

python3 fetch_openalex_test.py

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Fetch failed${NC}"
    exit 1
fi

FETCH_TIME=$((SECONDS - START_TIME))
echo -e "${GREEN}✓ Fetch completed in ${FETCH_TIME}s${NC}"
echo ""

# ==============================================================================
# Step 2: Extract PSP Relations with Ollama (Robust)
# ==============================================================================
echo "=========================================="
echo "Step 2: Extracting Causal Relations"
echo "=========================================="

START_TIME=$SECONDS

python3 extract_robust.py

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Extraction failed${NC}"
    exit 1
fi

EXTRACT_TIME=$((SECONDS - START_TIME))
echo -e "${GREEN}✓ Extraction completed in ${EXTRACT_TIME}s${NC}"
echo ""

# ==============================================================================
# Step 3: Validate Output
# ==============================================================================
echo "=========================================="
echo "Step 3: Validating KG Output"
echo "=========================================="

if [ ! -f "outputs/kg_raw_2d_doping.json" ]; then
    echo -e "${RED}✗ Output file not found: outputs/kg_raw_2d_doping.json${NC}"
    exit 1
fi

# Check JSON validity and extract stats
python3 << EOF
import json
import sys

try:
    with open('outputs/kg_raw_2d_doping.json', 'r') as f:
        kg = json.load(f)

    exp_count = len(kg.get('doping_experiments', []))
    rel_count = len(kg.get('causal_relationships', []))

    print(f"✓ Valid JSON")
    print(f"  Experiments: {exp_count}")
    print(f"  Causal relationships: {rel_count}")

    if exp_count == 0 and rel_count == 0:
        print("⚠ Warning: No data extracted (abstracts may lack experimental content)")
        sys.exit(0)

    # Show sample
    if rel_count > 0:
        print("\nSample causal relationship:")
        rel = kg['causal_relationships'][0]
        print(f"  {rel.get('cause_parameter', 'N/A')} → {rel.get('effect_on_doping', 'N/A')} → {rel.get('affected_property', 'N/A')}")

except json.JSONDecodeError as e:
    print(f"✗ Invalid JSON: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Validation failed${NC}"
    exit 1
fi

echo ""

# ==============================================================================
# Step 4: Generate Statistics
# ==============================================================================
echo "=========================================="
echo "Step 4: Generating Statistics"
echo "=========================================="

python3 << EOF
import json
from collections import Counter

with open('outputs/kg_raw_2d_doping.json', 'r') as f:
    kg = json.load(f)

experiments = kg.get('doping_experiments', [])
relationships = kg.get('causal_relationships', [])

print(f"Total experiments: {len(experiments)}")
print(f"Total causal relationships: {len(relationships)}")

if experiments:
    materials = Counter(exp.get('host_material', 'Unknown') for exp in experiments)
    print("\nTop 5 host materials:")
    for mat, count in materials.most_common(5):
        print(f"  {mat}: {count}")

if relationships:
    causes = Counter(rel.get('cause_parameter', 'Unknown') for rel in relationships)
    print("\nTop 5 causal parameters:")
    for cause, count in causes.most_common(5):
        print(f"  {cause}: {count}")

# Check metadata
if 'metadata' in kg and 'statistics' in kg['metadata']:
    stats = kg['metadata']['statistics']
    print(f"\nExtraction statistics:")
    print(f"  Success rate: {stats['success']}/{stats['processed']} ({stats['success']/max(stats['processed'],1)*100:.1f}%)")
    print(f"  Empty (no experimental content): {stats['empty']}")
    print(f"  Failed: {stats['failed']}")
EOF

echo ""

# ==============================================================================
# Final Summary
# ==============================================================================
echo "=========================================="
echo "TEST PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  ✓ papers/openalex_test/openalex_test_papers.json (metadata)"
echo "  ✓ papers/openalex_test/abstracts/*.txt (50 abstracts)"
echo "  ✓ outputs/kg_raw_2d_doping.json (knowledge graph)"
echo "  ✓ outputs/extraction_logs/ (failure logs)"
echo ""
echo "Next steps:"
echo "  1. Review: cat outputs/kg_raw_2d_doping.json | jq '.metadata'"
echo "  2. Check failures: ls outputs/extraction_logs/"
echo "  3. If successful, scale up to 5000 papers:"
echo "     - Edit fetch_openalex_papers.py (TARGET_PAPERS = 5000)"
echo "     - Edit extract_robust.py (INPUT_DIR = 'papers/openalex_metadata/abstracts')"
echo "     - Run: python run_full_pipeline.py"
echo ""
echo "Pipeline timing:"
echo "  Fetch: ${FETCH_TIME}s"
echo "  Extract: ${EXTRACT_TIME}s"
echo ""
