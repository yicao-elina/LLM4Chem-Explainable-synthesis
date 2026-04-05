Implement KG diagnostics comparing:
- Original KG
- Enriched KG

Metrics:
- # processing nodes
- # structure nodes
- # property nodes
- # edges (P→S, S→P)
- # complete PSP chains
- Average path length
- Mechanism coverage (% edges with mechanism_quote)

Output:
1) CSV table: kg_statistics.csv
2) LaTeX table: kg_statistics.tex
3) PDF figure:
   - Bar chart: original vs enriched KG size
   - Histogram: PSP chain lengths
