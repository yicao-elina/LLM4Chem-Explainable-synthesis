Goal:
Extend the existing ARIA engine with a persistent, on-demand KG enrichment mechanism,
while preserving the current online-search-based reasoning pipeline.

Context:
- Current online search (Gemini grounding tool) is used only for reasoning-time validation.
- The Knowledge Graph (KG) is sparse and static.
- We want dynamic KG growth using literature search + Ollama extraction.

Task:
Implement a dual-loop system:

LOOP A — Persistent KG Enrichment (NEW)
1) When no direct causal path is found OR confidence < threshold:
   - Query OpenAlex API for papers relevant to the user query (keywords from synthesis / properties).
   - Retrieve title + abstract (2015–2026).
2) Use Ollama (7B) to extract structured PSP relations:
   {
     "cause_parameter",
     "effect_on_structure",
     "affected_property",
     "mechanism_quote"
   }
3) Normalize entities and deduplicate against existing KG.
4) Append relations to KG JSON (same schema as kg_example.json).
5) Rebuild NetworkX graph and embeddings in memory.
6) Log newly added nodes/edges with timestamps.

LOOP B — Ephemeral Online Reasoning (EXISTING)
- Keep the current Gemini grounding-based online validation unchanged.
- Do NOT persist grounding results into the KG.

Implementation requirements:
- Add a KGEnricher module (separate file).
- Do not modify core ARIA reasoning logic.
- Add a controller that decides when to trigger enrichment.
- Ensure reproducibility and logging.

Outputs:
- enriched_kg.json
- kg_growth_log.csv (timestamp, nodes_added, edges_added)
- ablation-ready KG snapshots
- Minimal tests demonstrating:
  (a) query before enrichment
  (b) KG growth
  (c) improved path discovery after enrichment
