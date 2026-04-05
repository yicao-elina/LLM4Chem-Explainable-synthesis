import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap

# --- Robustness Improvement: Optional Color Library ---
# Make the jhu_colors library optional for better portability.
try:
    import jhu_colors
    JHU_COLORS_AVAILABLE = True
except ImportError:
    JHU_COLORS_AVAILABLE = False
    print("Warning: 'jhu_colors' package not found. Using default matplotlib colors.")


def build_causal_graph(json_file_path: str, output_image_path: str):
    """
    Loads causal relationships from a JSON file, builds a Directed Acyclic Graph (DAG),
    and saves a high-quality visualization.

    Args:
        json_file_path (str): The path to the input JSON file.
        output_image_path (str): The path to save the output graph image.
    """
    # --- 1. Load and Parse Data ---
    input_path = Path(json_file_path)
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    # --- Robustness Improvement: Handle potential JSON errors ---
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_path}. Please check if the file is valid.")
        return

    relationships = data.get("causal_relationships", [])
    if not relationships:
        print("No causal relationships found in the JSON file.")
        return

    # --- 2. Build the Graph using NetworkX ---
    G = nx.DiGraph()

    for rel in relationships:
        # --- Bug Fix: Handle NoneType values from JSON ---
        # Use `or` to provide a default string if .get() returns None or an empty string.
        cause_text = rel.get("cause_parameter") or "Unknown Cause"
        effect_text = rel.get("effect_on_doping") or "Unknown Effect"
        affected_property = rel.get("affected_property")

        # --- Improvement: Create more specific node labels ---
        # Combine the effect with the property it affects for a more informative graph.
        cause = textwrap.fill(cause_text.strip(), width=25)
        
        if affected_property and affected_property.strip():
            effect_label = f"{effect_text.strip()}\n({affected_property.strip()})"
        else:
            effect_label = effect_text.strip()
        
        effect = textwrap.fill(effect_label, width=25)
        
        # Add edge if both cause and effect are meaningful
        if 'unknown' not in cause.lower() and 'n/a' not in cause.lower() and \
           'unknown' not in effect.lower() and 'n/a' not in effect.lower():
            # Add mechanism as an edge attribute for richer data representation
            mechanism = rel.get("mechanism_quote", "")
            G.add_edge(cause, effect, mechanism=mechanism)


    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Check for cycles, which would invalidate the DAG assumption
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            print(f"Warning: Cycles detected in the graph, it is not a DAG. Found {len(cycles)} cycles.")
        else:
            print("Confirmed: The graph is a Directed Acyclic Graph (DAG).")
    except Exception as e:
        print(f"Could not check for cycles: {e}")


    # --- 3. Visualize the Graph (Publication Quality) ---
    plt.figure(figsize=(24, 24)) # Increased size for better readability

    # Use a layout that spreads nodes out to minimize overlap
    # 'spring_layout' is good for this. k adjusts the optimal distance between nodes.
    # For very large graphs, you might need to increase 'k' and 'iterations'.
    pos = nx.spring_layout(G, k=1.2, iterations=70, seed=42)

    # --- Node and Edge Styling with Fallback ---
    if JHU_COLORS_AVAILABLE:
        node_color = jhu_colors.get_jhu_color('Spirit Blue')
        edge_color = jhu_colors.get_jhu_color('Heritage Blue')
    else:
        node_color = '#1f78b4'  # Default blue
        edge_color = '#333333'  # Dark grey

    node_sizes = [len(node) * 100 for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_sizes, alpha=0.85)

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_color,
        arrowstyle='->',
        arrowsize=25,
        width=1.5,
        alpha=0.6,
        node_size=[s + 500 for s in node_sizes] # Ensure arrows don't overlap nodes
    )

    # Label styling
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', font_weight='bold')

    # --- 4. Final Touches and Saving ---
    plt.title("Causal Graph of Doping in 2D Materials", fontsize=28, fontweight='bold')
    plt.axis('off') # Hide the axes
    plt.tight_layout()

    output_path = Path(output_image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')

    print(f"Graph visualization saved to {output_path}")
    plt.show()


if __name__ == '__main__':
    # Ensure the JSON file is in the same directory or provide the correct path
    json_file = 'outputs/combined_doping_data.json'
    output_image = 'outputs/causal_graph.png'
    build_causal_graph(json_file, output_image)
