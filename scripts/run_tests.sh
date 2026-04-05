#!/bin/bash
# ARIA Test Runner Script

set -e  # Exit on error

echo "============================================"
echo "ARIA Comprehensive Testing"
echo "============================================"
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "causalmat" ]]; then
    echo "❌ Error: causalmat conda environment not activated"
    echo "Please run: conda activate causalmat"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "⚠️  Warning: Ollama service not detected"
    echo "Please start Ollama: ollama serve"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Navigate to project root
cd "$(dirname "$0")/.."

echo "📁 Working directory: $(pwd)"
echo ""

# Default to 6-variant test
TEST_SCRIPT="tests.test_all_6_variants"

# Parse command line arguments
if [[ "$1" == "4" ]] || [[ "$1" == "phase1" ]]; then
    TEST_SCRIPT="tests.test_all_variants"
    echo "Running Phase 1 tests (4 variants)..."
elif [[ "$1" == "6" ]] || [[ "$1" == "phase2" ]] || [[ -z "$1" ]]; then
    echo "Running Phase 2 tests (6 variants)..."
else
    echo "Usage: $0 [4|6|phase1|phase2]"
    echo "  4, phase1: Run 4-variant tests"
    echo "  6, phase2: Run 6-variant tests (default)"
    exit 1
fi

echo ""
echo "🚀 Starting tests..."
echo "⏱️  Estimated time: 15-30 minutes"
echo ""

# Run tests
python -m $TEST_SCRIPT

echo ""
echo "============================================"
echo "✅ Testing Complete!"
echo "============================================"
echo ""
echo "📊 Results saved to: results/"
echo ""
