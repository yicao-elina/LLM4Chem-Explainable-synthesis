# ARIA Engine Upgrade Plan: State-of-the-Art Causal Reasoning

## Executive Summary

Transform ARIA from Gemini-based system to open-source Ollama with advanced causal reasoning capabilities, rigorous ablation studies, and dynamic KG enrichment.

**Current State:**
- 5 variants (baseline, native_kg, aria, online, CoT) using Gemini API
- Static KG with 3-tier reasoning
- Basic online search integration
- Limited causal inference capabilities

**Target State:**
- Ollama-based open-source models (reasoning-capable)
- 6 rigorous ablation variants
- Advanced causal reasoning (ToT, MCTS, uncertainty quantification)
- Dynamic KG enrichment with literature search
- Optimal experimental design for directed search

---

## Phase 1: Ollama Migration & Ablation Rigor (Week 1-2)

### 1.1 Replace Gemini with Ollama

**Objective:** Migrate all 4 variants to use local Ollama models

**Models to Use:**
- **Primary:** `deepseek-r1:7b` or `qwen2.5:14b` (best reasoning capability)
- **Embedding:** Keep `all-MiniLM-L6-v2` or upgrade to `nomic-embed-text`
- **Fallback:** `llama3.1:8b` for comparison

**Implementation:**
```python
# New: engine/ollama_client.py
class OllamaClient:
    """Unified Ollama client for all variants"""
    def __init__(self, model="deepseek-r1:7b"):
        self.model = model
        self.base_url = "http://localhost:11434"

    def generate(self, prompt, temperature=0.0, max_tokens=2048):
        # Structured output support
        # JSON mode enforcement
        # Retry logic
        pass

    def embed(self, text):
        # Embedding generation
        pass
```

**Files to Update:**
- `baseline.py` → `baseline_ollama.py`
- `aria.py` → `aria_ollama.py`
- `online.py` → `online_ollama.py`
- `CoT.py` → `CoT_ollama.py`

**Migration Checklist:**
- [ ] Create `ollama_client.py` wrapper
- [ ] Update API calls (genai → ollama)
- [ ] Replace LangChain chains with Ollama prompts
- [ ] Test JSON output formatting
- [ ] Benchmark performance vs Gemini
- [ ] Update all imports and configurations

### 1.2 Fix Ablation Study Rigor

**Current Issues:**
1. `naive_kg.py` is empty (not implemented)
2. Missing KG-only variant (no LLM enhancement)
3. Online search not properly isolated
4. No proper control for each component

**New Rigorous Ablation Design:**

```
┌────────────────────────────────────────────────────────────────┐
│ Variant 1: BASELINE (Pure LLM)                                │
│ - Ollama only, no KG, no search                               │
│ - Tier 3 reasoning exclusively                                │
│ - Control for model capability                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ Variant 2: KG_ONLY (Graph-based without LLM enhancement)      │
│ - Direct graph traversal + template filling                   │
│ - No LLM reasoning, only mechanistic lookup                   │
│ - Tests KG contribution in isolation                          │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ Variant 3: NAIVE_KG (LLM + KG without reasoning tiers)        │
│ - Concatenate all KG paths to LLM prompt                      │
│ - No tier separation, no transfer learning                    │
│ - Tests simple KG augmentation                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ Variant 4: ARIA_CORE (Tier 1/2/3 without search)              │
│ - Full 3-tier reasoning (Direct/Transfer/Fallback)            │
│ - Analogy-based transfer learning                             │
│ - No online search component                                  │
│ - Tests hierarchical reasoning contribution                   │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ Variant 5: ARIA_SEARCH (ARIA + Online Search)                 │
│ - 3-tier reasoning + literature validation                    │
│ - Dynamic search for KG validation                            │
│ - Citation analysis                                            │
│ - Tests search component contribution                         │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ Variant 6: ARIA_FULL (Complete System)                        │
│ - 3-tier reasoning                                             │
│ - Online search validation                                     │
│ - Chain-of-thought transparency                                │
│ - Dynamic KG enrichment (NEW)                                  │
│ - Advanced causal reasoning (NEW)                              │
└────────────────────────────────────────────────────────────────┘
```

**Ablation Matrix:**

| Variant | LLM | KG | Tier 1 | Tier 2 | Tier 3 | Search | CoT | Dynamic KG | Causal++ |
|---------|-----|----|----|----|----|-----|-----|---------|----------|
| BASELINE | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| KG_ONLY | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| NAIVE_KG | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| ARIA_CORE | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| ARIA_SEARCH | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| ARIA_FULL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Component Contribution Analysis:**
- KG contribution: Compare NAIVE_KG vs BASELINE
- Tier reasoning: Compare ARIA_CORE vs NAIVE_KG
- Search contribution: Compare ARIA_SEARCH vs ARIA_CORE
- CoT contribution: Compare ARIA_FULL vs ARIA_SEARCH
- Dynamic KG: Compare ARIA_FULL (with/without enrichment)
- Advanced causal: Compare ARIA_FULL (with/without ToT/MCTS)

---

## Phase 2: Advanced Causal Reasoning (Week 3-4)

### 2.1 Tree of Thought (ToT) Implementation

**Concept:** Explore multiple reasoning paths simultaneously, evaluate quality, backtrack if needed

**Architecture:**
```python
class TreeOfThoughtReasoner:
    """
    Multi-path causal reasoning with evaluation and pruning.

    For each query:
    1. Generate K candidate reasoning paths (K=3-5)
    2. Evaluate each path's coherence and plausibility
    3. Prune low-quality paths
    4. Combine insights from top paths
    """

    def __init__(self, branching_factor=3, depth=3):
        self.branching_factor = branching_factor
        self.max_depth = depth

    def generate_reasoning_tree(self, query, kg_context):
        """
        Generate tree of reasoning paths.

        Tree Structure:
        Root (Query)
          ├─ Path 1: Direct mechanism A → Outcome A
          ├─ Path 2: Alternative mechanism B → Outcome B
          └─ Path 3: Multi-hop mechanism C → Outcome C

        Each path scored by:
        - Mechanistic plausibility (from KG evidence)
        - Logical coherence (LLM evaluation)
        - Empirical support (literature citations)
        """
        pass

    def evaluate_path(self, path):
        """
        Score reasoning path quality.

        Metrics:
        - Causal consistency: Do intermediate steps logically connect?
        - KG alignment: How well does path match KG evidence?
        - Uncertainty: Are there competing explanations?
        """
        pass

    def prune_and_combine(self, paths):
        """
        Keep top-k paths, synthesize final answer.

        Strategy:
        - If paths agree: High confidence
        - If paths diverge: Report uncertainty + alternatives
        - If contradiction: Identify confounding variables
        """
        pass
```

**Integration:**
```python
# In ARIA_FULL variant
def forward_prediction_with_tot(self, synthesis_inputs):
    # Step 1: Standard ARIA reasoning (baseline path)
    baseline_path = self._tier_reasoning(synthesis_inputs)

    # Step 2: Generate alternative ToT paths
    tot_paths = self.tot_reasoner.generate_reasoning_tree(
        query=synthesis_inputs,
        kg_context=self.kg
    )

    # Step 3: Evaluate and combine
    final_reasoning = self.tot_reasoner.prune_and_combine(
        paths=[baseline_path] + tot_paths
    )

    return final_reasoning
```

### 2.2 MCTS-Style Tree Splitting Along Covariates

**Motivation:** When predicted outcomes are negative/implausible, explore finer-grained covariates

**Concept:** Monte Carlo Tree Search adapted for causal discovery

```python
class MCTSCausalExplorer:
    """
    MCTS for identifying confounding variables and refining causal paths.

    Problem: Coarse inputs → Negative/unexpected outcomes
    Solution: Systematically split inputs into finer granularity

    Example:
    Input: "CVD at 750°C" → Negative outcome
    Split: Temperature (750°C) + Pressure (?) + Gas composition (?)
           → Identify that pressure is missing confounder
    """

    def __init__(self, kg, max_iterations=100):
        self.kg = kg
        self.max_iterations = max_iterations

    def explore_confounders(self, synthesis_inputs, observed_outcome):
        """
        MCTS exploration of covariate space.

        MCTS Phases:
        1. Selection: Choose most promising covariate to split
        2. Expansion: Add new granular covariates
        3. Simulation: Predict outcome with refined inputs
        4. Backpropagation: Update covariate importance scores

        Returns:
        - Identified confounders
        - Refined causal path
        - Uncertainty quantification
        """
        tree = MCTSNode(inputs=synthesis_inputs)

        for iteration in range(self.max_iterations):
            # 1. Selection
            node = self._select_promising_node(tree)

            # 2. Expansion
            if node.is_expandable():
                child = self._expand_covariate(node)
                node = child

            # 3. Simulation
            outcome = self._simulate_outcome(node.state)
            reward = self._compute_reward(outcome, observed_outcome)

            # 4. Backpropagation
            self._backpropagate(node, reward)

        # Return best path
        return self._extract_best_path(tree)

    def _select_promising_node(self, node):
        """UCB1 selection: balance exploration vs exploitation"""
        pass

    def _expand_covariate(self, node):
        """
        Systematically split coarse inputs.

        Covariate Hierarchy:
        - Temperature → [Temperature, Ramp rate, Dwell time]
        - Pressure → [Pressure, Gas composition, Flow rate]
        - Dopant → [Dopant, Concentration, Precursor, Deposition rate]
        """
        pass

    def _compute_reward(self, predicted, observed):
        """
        Reward = similarity(predicted, observed)
        Higher reward → Better covariate set
        """
        pass
```

**Integration with ARIA:**
```python
def inverse_design_with_mcts(self, desired_properties):
    # Standard ARIA inverse design
    initial_synthesis = self._tier_reasoning(desired_properties)

    # Forward prediction to check plausibility
    predicted = self.forward_prediction(initial_synthesis)

    # If poor match, explore confounders with MCTS
    if self._similarity(predicted, desired_properties) < 0.6:
        refined_synthesis = self.mcts_explorer.explore_confounders(
            synthesis_inputs=initial_synthesis,
            observed_outcome=desired_properties
        )
        return refined_synthesis

    return initial_synthesis
```

### 2.3 Uncertainty Reasoning Over Added Information

**Objective:** Quantify whether new covariates/information improve predictions

**Method:** Bayesian Information Gain

```python
class UncertaintyQuantifier:
    """
    Assess value of additional covariates using information theory.

    Question: Does adding covariate X reduce prediction uncertainty?

    Metrics:
    - Mutual Information: I(Y; X) where Y=outcome, X=covariate
    - Conditional Entropy: H(Y|X) - how much uncertainty remains?
    - Expected Information Gain: E[IG] before acquiring X
    """

    def compute_information_gain(self, covariate, outcome_distribution):
        """
        IG(Y|X) = H(Y) - H(Y|X)

        High IG → Covariate X is informative
        Low IG → Covariate X is redundant
        """
        pass

    def rank_covariates_by_value(self, candidate_covariates, current_state):
        """
        For each candidate covariate:
        1. Estimate I(Y; X) using KG data
        2. Compute expected reduction in entropy
        3. Rank by information gain

        Output: Prioritized list for directed search
        """
        pass

    def stopping_criterion(self, current_uncertainty, threshold=0.1):
        """
        When to stop adding covariates?

        Stop if:
        - Uncertainty < threshold (confident enough)
        - Diminishing returns (IG < 0.05)
        - No more covariates available
        """
        pass
```

### 2.4 Optimal Experimental Design for Directed Search

**Concept:** Use experimental design theory to guide online search

**Implementation:**
```python
class OptimalSearchDesigner:
    """
    Apply OED principles to decide what to search for next.

    Classical OED Goals:
    - D-optimality: Maximize determinant of Fisher information
    - A-optimality: Minimize trace of covariance matrix
    - E-optimality: Maximize minimum eigenvalue

    Adapted for Causal Search:
    - Maximize information about causal mechanisms
    - Minimize overlap with existing KG knowledge
    - Prioritize high-impact covariates
    """

    def design_search_query(self, current_kg, target_relationship):
        """
        Generate optimal search query to fill KG gaps.

        Strategy:
        1. Identify missing links in causal graph
        2. Compute Fisher information matrix for each missing link
        3. Select query that maximizes information gain
        4. Generate literature search query

        Example:
        Current KG: Temperature → Mobility (missing mechanism)
        Optimal Query: "temperature effect on carrier scattering in MoS2"
        """
        pass

    def compute_fisher_information(self, parameter_space):
        """
        Fisher Information: How much data tells us about parameter?

        High FI → Small experiments reveal parameter
        Low FI → Need many experiments to learn parameter
        """
        pass

    def adaptive_search_strategy(self, search_results):
        """
        Update search strategy based on results.

        If search finds useful info:
        - Extract causal relationships
        - Update KG
        - Re-plan next search

        If search is uninformative:
        - Try alternative query
        - Broaden/narrow scope
        - Change search terms
        """
        pass
```

---

## Phase 3: Dynamic KG Enrichment (Week 5-6)

### 3.1 Persistent KG with On-Demand Growth

**Architecture:**

```python
class DynamicKGManager:
    """
    Manage persistent KG with incremental enrichment.

    Features:
    - Save/load KG state
    - Incremental updates (no full reload)
    - Version control (track changes)
    - Confidence decay (old knowledge degrades)
    """

    def __init__(self, kg_path="outputs/dynamic_kg.json"):
        self.kg_path = Path(kg_path)
        self.kg = self._load_or_initialize_kg()
        self.enrichment_log = []

    def _load_or_initialize_kg(self):
        """Load existing KG or create new one."""
        if self.kg_path.exists():
            return self._load_kg_with_metadata()
        else:
            return self._create_empty_kg()

    def enrich_kg_on_demand(self, query, trigger_condition="low_confidence"):
        """
        Dynamically enrich KG when needed.

        Trigger Conditions:
        - Low confidence prediction (< 0.5)
        - Missing causal path
        - Conflicting evidence
        - User request

        Process:
        1. Identify knowledge gap
        2. Generate literature search query
        3. Execute search (OpenAlex/Semantic Scholar)
        4. Extract causal relations (Ollama)
        5. Validate & integrate into KG
        6. Save updated KG
        """
        # 1. Identify gap
        gap = self._identify_knowledge_gap(query)

        # 2. Search literature
        papers = self.literature_searcher.search(gap)

        # 3. Extract relations
        new_relations = self.relation_extractor.extract(papers)

        # 4. Validate
        validated = self._validate_relations(new_relations)

        # 5. Integrate
        self._integrate_into_kg(validated)

        # 6. Save
        self.save_kg()

        return validated

    def _identify_knowledge_gap(self, query):
        """
        Analyze where KG is insufficient.

        Gap Types:
        - Missing nodes (materials, properties)
        - Missing edges (causal relationships)
        - Low-confidence edges (weak evidence)
        - Contradictory edges (conflicting mechanisms)
        """
        pass

    def save_kg(self):
        """Save KG with metadata (timestamp, version, etc.)"""
        metadata = {
            "last_updated": datetime.now().isoformat(),
            "version": self.kg_version,
            "enrichment_count": len(self.enrichment_log),
            "total_nodes": self.kg.number_of_nodes(),
            "total_edges": self.kg.number_of_edges()
        }

        kg_data = {
            "metadata": metadata,
            "causal_relationships": self._export_edges(),
            "doping_experiments": self._export_nodes(),
            "enrichment_log": self.enrichment_log
        }

        with open(self.kg_path, 'w') as f:
            json.dump(kg_data, f, indent=2)
```

### 3.2 Literature Search Integration

```python
class LiteratureSearcher:
    """
    Search scientific literature to enrich KG.

    APIs:
    - OpenAlex (primary)
    - Semantic Scholar (fallback)
    - arXiv (preprints)
    """

    def __init__(self):
        self.openalex = OpenAlexClient()
        self.semantic_scholar = SemanticScholarClient()

    def search(self, knowledge_gap):
        """
        Generate targeted search query and fetch papers.

        Input: KnowledgeGap object
        - missing_cause: "annealing temperature"
        - missing_effect: "substitutional fraction"
        - material_context: "MoS2"

        Output: List of Paper objects
        - title, abstract, doi, year
        - relevance_score
        """
        query = self._construct_search_query(knowledge_gap)
        papers = self.openalex.search(query, limit=20)
        ranked = self._rank_by_relevance(papers, knowledge_gap)
        return ranked[:10]

    def _construct_search_query(self, gap):
        """
        Build targeted search query.

        Template:
        "{cause} effect on {effect} in {material}"

        Example:
        "annealing temperature effect on substitutional doping in MoS2"
        """
        pass
```

### 3.3 Ollama-Based Relation Extraction

```python
class OllamaRelationExtractor:
    """
    Extract causal relations from paper abstracts using Ollama.

    Reuses: extract_robust.py logic
    Enhanced: Targeted extraction for specific gaps
    """

    def __init__(self, model="qwen2:7b"):
        self.ollama = OllamaClient(model)
        self.extraction_prompt = self._load_extraction_prompt()

    def extract(self, papers, target_gap=None):
        """
        Extract causal relations from papers.

        If target_gap specified:
        - Focus extraction on that specific relationship
        - Filter out irrelevant relations
        - Higher precision, lower recall

        Returns: List[CausalRelation]
        """
        relations = []
        for paper in papers:
            prompt = self._build_prompt(paper.abstract, target_gap)
            response = self.ollama.generate(prompt)
            extracted = self._parse_response(response)
            relations.extend(extracted)

        return relations

    def _build_prompt(self, abstract, target_gap):
        """
        Targeted extraction prompt.

        If gap = "temperature → substitutional fraction":
        - Prompt specifically for this relationship
        - Ignore other causal paths
        - Extract mechanism quote
        """
        if target_gap:
            return f"""Extract ONLY causal relationships about how {target_gap.cause}
            affects {target_gap.effect} in {target_gap.material}.

            Abstract: {abstract}

            Output JSON:
            {{
              "cause_parameter": "{target_gap.cause}",
              "effect_on_doping": "...",
              "affected_property": "{target_gap.effect}",
              "mechanism_quote": "..."
            }}
            """
        else:
            # Use general extraction prompt
            return self.extraction_prompt.format(abstract=abstract)
```

### 3.4 Integration with ARIA Reasoning

```python
class ARIAWithDynamicKG:
    """
    Full ARIA system with on-demand KG enrichment.
    """

    def __init__(self):
        self.kg_manager = DynamicKGManager()
        self.aria_core = ARIACore(kg=self.kg_manager.kg)
        self.enrichment_enabled = True

    def forward_prediction(self, synthesis_inputs):
        # Attempt standard ARIA reasoning
        result = self.aria_core.forward_prediction(synthesis_inputs)

        # Check if confidence is low
        if result.confidence < 0.5 and self.enrichment_enabled:
            print("Low confidence. Enriching KG...")

            # Enrich KG on-demand
            new_relations = self.kg_manager.enrich_kg_on_demand(
                query=synthesis_inputs,
                trigger_condition="low_confidence"
            )

            if new_relations:
                # Retry with enriched KG
                self.aria_core.reload_kg(self.kg_manager.kg)
                result = self.aria_core.forward_prediction(synthesis_inputs)
                result.metadata["kg_enriched"] = True
                result.metadata["new_relations_added"] = len(new_relations)

        return result
```

---

## Phase 4: Implementation Priorities

### Sprint 1 (Week 1): Foundation
- [ ] Create `ollama_client.py` wrapper
- [ ] Migrate `baseline.py` → `baseline_ollama.py`
- [ ] Implement `KG_ONLY.py` variant
- [ ] Implement `NAIVE_KG.py` variant
- [ ] Test all 3 variants on sample data

### Sprint 2 (Week 2): Core ARIA Migration
- [ ] Migrate `aria.py` → `aria_core_ollama.py`
- [ ] Migrate `online.py` → `aria_search_ollama.py`
- [ ] Migrate `CoT.py` → `aria_full_ollama.py`
- [ ] Create unified evaluation framework for 6 variants

### Sprint 3 (Week 3): Advanced Causal Reasoning
- [ ] Implement `TreeOfThoughtReasoner`
- [ ] Implement `MCTSCausalExplorer`
- [ ] Implement `UncertaintyQuantifier`
- [ ] Implement `OptimalSearchDesigner`
- [ ] Integrate into `aria_full_ollama.py`

### Sprint 4 (Week 4): Testing & Refinement
- [ ] Test ToT on challenging cases
- [ ] Test MCTS on confounded examples
- [ ] Benchmark uncertainty quantification
- [ ] Optimize search query generation

### Sprint 5 (Week 5): Dynamic KG
- [ ] Implement `DynamicKGManager`
- [ ] Implement `LiteratureSearcher` (OpenAlex + Semantic Scholar)
- [ ] Implement `OllamaRelationExtractor`
- [ ] Integration testing

### Sprint 6 (Week 6): Full System Integration
- [ ] Create `ARIAWithDynamicKG` class
- [ ] End-to-end testing
- [ ] Performance benchmarking
- [ ] Documentation

---

## Evaluation Framework

### Rigorous Ablation Metrics

```python
class RigorousEvaluation:
    """
    Comprehensive evaluation of all 6 variants.
    """

    def evaluate_all_variants(self, test_cases):
        """
        Test all variants on same test cases.

        Metrics:
        1. Accuracy (similarity to ground truth)
        2. Confidence calibration
        3. Robustness to perturbation
        4. Reasoning quality (human eval)
        5. Computational cost
        6. KG utilization (% of KG used)
        """
        pass

    def component_contribution_analysis(self):
        """
        Quantify contribution of each component.

        Comparisons:
        - KG vs no-KG: NAIVE_KG - BASELINE
        - Tiers vs naive: ARIA_CORE - NAIVE_KG
        - Search vs no-search: ARIA_SEARCH - ARIA_CORE
        - CoT vs no-CoT: ARIA_FULL - ARIA_SEARCH
        - ToT vs standard: ARIA_FULL(ToT) - ARIA_FULL(no-ToT)
        - MCTS vs no-MCTS: Similar
        - Dynamic KG vs static: Similar
        """
        pass

    def failure_analysis(self):
        """
        Categorize failure modes for each variant.

        Categories:
        - Missing KG knowledge
        - Incorrect reasoning
        - Poor transfer learning
        - Search failure
        - Extraction errors
        """
        pass
```

---

## Success Criteria

### Technical
- [ ] All 6 variants working with Ollama
- [ ] Ablation study shows statistically significant improvements for each component
- [ ] ToT finds alternative mechanisms in ≥30% of cases
- [ ] MCTS identifies confounders in ≥20% of negative predictions
- [ ] Dynamic KG enrichment adds ≥10 high-quality relations per 100 queries
- [ ] End-to-end latency < 30s per query (including enrichment)

### Scientific
- [ ] Tree of Thought reveals 2-3 plausible mechanisms per query
- [ ] MCTS correctly identifies missing covariates in controlled experiments
- [ ] Uncertainty quantification correlates with prediction error (ρ > 0.6)
- [ ] Optimal search reduces search iterations by ≥40% vs random search
- [ ] Dynamic KG improves coverage from ~50% to ~80% over time

### Practical
- [ ] System runs on single GPU (or CPU-only for smaller models)
- [ ] All components documented with examples
- [ ] Evaluation results reproducible
- [ ] Code follows PEP8 and best practices

---

## File Structure (After Upgrade)

```
engine/
├── ollama_client.py                 # NEW: Unified Ollama interface
├── baseline_ollama.py               # Variant 1
├── kg_only.py                       # NEW: Variant 2
├── naive_kg_ollama.py               # NEW: Variant 3
├── aria_core_ollama.py              # Variant 4
├── aria_search_ollama.py            # Variant 5
├── aria_full_ollama.py              # Variant 6 (with all enhancements)
│
├── causal_reasoning/                # NEW: Advanced reasoning components
│   ├── tree_of_thought.py
│   ├── mcts_explorer.py
│   ├── uncertainty_quantifier.py
│   └── optimal_search.py
│
├── kg_management/                   # NEW: Dynamic KG components
│   ├── dynamic_kg_manager.py
│   ├── literature_searcher.py
│   ├── relation_extractor.py
│   └── kg_validator.py
│
├── evaluation/                      # Enhanced evaluation
│   ├── rigorous_ablation.py
│   ├── component_analysis.py
│   └── failure_analysis.py
│
└── prompts/                         # Ollama-optimized prompts
    ├── baseline_prompts.py
    ├── kg_prompts.py
    ├── tier_prompts.py
    ├── tot_prompts.py
    └── extraction_prompts.py
```

---

## Next Steps

1. **Review and approve** this plan
2. **Prioritize** which components to implement first
3. **Set timeline** for each sprint
4. **Assign resources** (if team project)
5. **Begin implementation** with Sprint 1

This upgrade will transform ARIA into a cutting-edge causal reasoning system with:
- Open-source foundation (Ollama)
- Rigorous scientific methodology (6-variant ablation)
- State-of-the-art reasoning (ToT, MCTS, UQ, OED)
- Dynamic knowledge growth (literature-driven KG enrichment)

Ready to begin implementation!
