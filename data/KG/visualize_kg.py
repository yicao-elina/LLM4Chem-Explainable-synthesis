"""
Wrapper script to visualize the knowledge graph using build_graph.py.

This script calls the existing build_graph.py with the correct parameters
for the OpenAlex-derived knowledge graph.
"""

from pathlib import Path
import sys

# Import the existing build_graph function
from build_graph import build_causal_graph

# ============= Configuration =============
INPUT_KG = "outputs/kg_from_abstracts.json"
OUTPUT_IMAGE = "outputs/publication/causal_graph_openalex.png"

def main():
    """Visualize the causal graph from OpenAlex KG."""
    input_path = Path(INPUT_KG)
    output_path = Path(OUTPUT_IMAGE)

    if not input_path.exists():
        print(f"✗ ERROR: Knowledge graph file not found: {input_path}")
        print(f"  Run extract_from_abstracts.py first")
        return 1

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Visualizing Causal Knowledge Graph")
    print(f"{'='*60}\n")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}\n")

    # Call the existing build_causal_graph function
    try:
        build_causal_graph(str(input_path), str(output_path))
        print(f"\n✓ Visualization complete!")
        print(f"  View: {output_path}")
        return 0
    except Exception as e:
        print(f"\n✗ ERROR during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
