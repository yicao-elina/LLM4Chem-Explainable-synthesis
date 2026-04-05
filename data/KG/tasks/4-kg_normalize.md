Purpose: Convert raw relations → causal KG.

Using @26KDD/KG/build_graph.py as reference:

1) Write a wrapper script that:
   - Takes kg_raw_2d_doping.json
   - Normalizes entity text (temperature, method, phase, property)
   - Filters low-quality relations
2) Generate a final KG JSON that EXACTLY matches kg_example.json schema.

Output:
- kg_example_2d_doping_enriched.json

Validation:
- Run build_graph.py successfully
- Report:
  - Number of nodes
  - Number of edges
  - Number of complete PSP chains
