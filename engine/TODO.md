# ARIA Engine Upgrade - Implementation Checklist

## 🎯 Overall Progress: 0/98 tasks complete

---

## Phase 1: Ollama Migration & Ablation Rigor (Weeks 1-2)

### Sprint 1.1: Ollama Client Foundation (5 tasks)
- [ ] 1.1.1 Create `ollama_client.py` with base OllamaClient class
- [ ] 1.1.2 Implement `generate()` method with JSON mode enforcement
- [ ] 1.1.3 Implement `embed()` method for embeddings
- [ ] 1.1.4 Add retry logic with exponential backoff
- [ ] 1.1.5 Write unit tests for ollama_client

### Sprint 1.2: Baseline Migration (8 tasks)
- [ ] 1.2.1 Copy `baseline.py` → `baseline_ollama.py`
- [ ] 1.2.2 Replace `google.generativeai` imports with `ollama_client`
- [ ] 1.2.3 Update `forward_prediction()` to use Ollama
- [ ] 1.2.4 Update `inverse_design()` to use Ollama
- [ ] 1.2.5 Test JSON output parsing
- [ ] 1.2.6 Benchmark performance (latency, quality)
- [ ] 1.2.7 Compare outputs with original Gemini version
- [ ] 1.2.8 Document API changes

### Sprint 1.3: KG-Only Variant (10 tasks)
- [ ] 1.3.1 Create `kg_only.py` from scratch
- [ ] 1.3.2 Implement graph loading (`_build_graph()`)
- [ ] 1.3.3 Implement direct path traversal (`_find_exact_paths()`)
- [ ] 1.3.4 Implement template-based response generation (no LLM)
- [ ] 1.3.5 Add forward prediction with templates
- [ ] 1.3.6 Add inverse design with templates
- [ ] 1.3.7 Handle missing paths gracefully
- [ ] 1.3.8 Test on sample data
- [ ] 1.3.9 Measure KG coverage (% queries with paths)
- [ ] 1.3.10 Document limitations

### Sprint 1.4: Naive KG Variant (12 tasks)
- [ ] 1.4.1 Create `naive_kg_ollama.py`
- [ ] 1.4.2 Load KG and build graph
- [ ] 1.4.3 Implement simple path extraction (all paths)
- [ ] 1.4.4 Concatenate all KG paths into prompt
- [ ] 1.4.5 NO tier separation - flat structure
- [ ] 1.4.6 NO transfer learning - exact match only
- [ ] 1.4.7 Implement forward prediction
- [ ] 1.4.8 Implement inverse design
- [ ] 1.4.9 Test on cases with/without KG coverage
- [ ] 1.4.10 Compare vs baseline (measure KG contribution)
- [ ] 1.4.11 Analyze failure modes
- [ ] 1.4.12 Document design decisions

### Sprint 1.5: ARIA Core Migration (15 tasks)
- [ ] 1.5.1 Copy `aria.py` → `aria_core_ollama.py`
- [ ] 1.5.2 Replace Gemini client with OllamaClient
- [ ] 1.5.3 Update LangChain chains → Ollama prompts
- [ ] 1.5.4 Migrate `ForwardDirectChain` logic
- [ ] 1.5.5 Migrate `ForwardTransferChain` logic
- [ ] 1.5.6 Migrate `InverseDirectChain` logic
- [ ] 1.5.7 Migrate `InverseTransferChain` logic
- [ ] 1.5.8 Implement tier decision logic (1→2→3)
- [ ] 1.5.9 Implement similarity-based transfer (>0.5 threshold)
- [ ] 1.5.10 Test Tier 1 (direct paths)
- [ ] 1.5.11 Test Tier 2 (transfer learning)
- [ ] 1.5.12 Test Tier 3 (fallback)
- [ ] 1.5.13 Verify confidence scoring
- [ ] 1.5.14 Compare vs naive_kg (measure tier contribution)
- [ ] 1.5.15 Document tier switching logic

---

## Phase 2: Search & CoT Migration (Weeks 2-3)

### Sprint 2.1: ARIA Search Migration (18 tasks)
- [ ] 2.1.1 Copy `online.py` → `aria_search_ollama.py`
- [ ] 2.1.2 Replace Gemini API calls
- [ ] 2.1.3 Implement literature search interface
- [ ] 2.1.4 Connect to OpenAlex API
- [ ] 2.1.5 Connect to Semantic Scholar API
- [ ] 2.1.6 Implement search query generation
- [ ] 2.1.7 Implement citation extraction
- [ ] 2.1.8 Implement grounding analysis (map claims to sources)
- [ ] 2.1.9 Update `_direct_path_query()` with search validation
- [ ] 2.1.10 Update `_transfer_learning_query()` with search
- [ ] 2.1.11 Update `_baseline_fallback_query()` with search
- [ ] 2.1.12 Test search integration
- [ ] 2.1.13 Measure search accuracy (% validated claims)
- [ ] 2.1.14 Analyze search failures
- [ ] 2.1.15 Optimize search query templates
- [ ] 2.1.16 Add caching for search results
- [ ] 2.1.17 Compare vs aria_core (measure search contribution)
- [ ] 2.1.18 Document search pipeline

### Sprint 2.2: ARIA Full (CoT) Migration (20 tasks)
- [ ] 2.2.1 Copy `CoT.py` → `aria_full_ollama.py`
- [ ] 2.2.2 Replace Gemini client
- [ ] 2.2.3 Migrate `AdvancedKGRetriever` class
- [ ] 2.2.4 Migrate `ChainOfThoughtReasoner` class
- [ ] 2.2.5 Migrate `CausalReasoningEngine` class
- [ ] 2.2.6 Update Step 1: Knowledge Retrieval
- [ ] 2.2.7 Update Step 2: Baseline Analysis
- [ ] 2.2.8 Update Step 3: KG Enhancement
- [ ] 2.2.9 Update Step 4: Multi-hop Reasoning
- [ ] 2.2.10 Update Step 5: Knowledge Synthesis
- [ ] 2.2.11 Update Step 6: Validation
- [ ] 2.2.12 Test full 6-step pipeline
- [ ] 2.2.13 Verify transparency (all steps logged)
- [ ] 2.2.14 Verify source attribution
- [ ] 2.2.15 Test multi-hop path finding (up to 3 hops)
- [ ] 2.2.16 Measure reasoning quality
- [ ] 2.2.17 Compare vs aria_search (measure CoT contribution)
- [ ] 2.2.18 Optimize prompt templates for Ollama
- [ ] 2.2.19 Add confidence calibration
- [ ] 2.2.20 Document CoT pipeline

---

## Phase 3: Advanced Causal Reasoning (Weeks 3-4)

### Sprint 3.1: Tree of Thought (15 tasks)
- [ ] 3.1.1 Create `causal_reasoning/tree_of_thought.py`
- [ ] 3.1.2 Implement `TreeOfThoughtReasoner` class
- [ ] 3.1.3 Implement `generate_reasoning_tree()` method
- [ ] 3.1.4 Implement branching strategy (K=3-5 paths)
- [ ] 3.1.5 Implement path evaluation metrics
- [ ] 3.1.6 Implement pruning logic (keep top-k)
- [ ] 3.1.7 Implement path combination/synthesis
- [ ] 3.1.8 Add uncertainty detection (diverging paths)
- [ ] 3.1.9 Add contradiction detection
- [ ] 3.1.10 Test on simple causal scenarios
- [ ] 3.1.11 Test on complex multi-mechanism scenarios
- [ ] 3.1.12 Integrate with aria_full_ollama
- [ ] 3.1.13 Benchmark: % cases with >1 plausible path
- [ ] 3.1.14 Measure improvement over single-path reasoning
- [ ] 3.1.15 Document ToT algorithm

### Sprint 3.2: MCTS Causal Explorer (18 tasks)
- [ ] 3.2.1 Create `causal_reasoning/mcts_explorer.py`
- [ ] 3.2.2 Implement `MCTSNode` class (tree structure)
- [ ] 3.2.3 Implement `MCTSCausalExplorer` class
- [ ] 3.2.4 Implement Selection phase (UCB1 formula)
- [ ] 3.2.5 Implement Expansion phase (covariate splitting)
- [ ] 3.2.6 Implement Simulation phase (outcome prediction)
- [ ] 3.2.7 Implement Backpropagation phase (reward update)
- [ ] 3.2.8 Define covariate hierarchy (temperature→[temp, ramp, dwell])
- [ ] 3.2.9 Implement reward function (similarity to target)
- [ ] 3.2.10 Implement stopping criteria (max iterations)
- [ ] 3.2.11 Extract best path from tree
- [ ] 3.2.12 Test on confounded examples
- [ ] 3.2.13 Test covariate identification accuracy
- [ ] 3.2.14 Integrate with aria_full_ollama (inverse design)
- [ ] 3.2.15 Benchmark: % of confounders correctly identified
- [ ] 3.2.16 Optimize hyperparameters (exploration constant)
- [ ] 3.2.17 Add visualization (tree structure)
- [ ] 3.2.18 Document MCTS algorithm

### Sprint 3.3: Uncertainty Quantifier (12 tasks)
- [ ] 3.3.1 Create `causal_reasoning/uncertainty_quantifier.py`
- [ ] 3.3.2 Implement `UncertaintyQuantifier` class
- [ ] 3.3.3 Implement information gain calculation (IG = H(Y) - H(Y|X))
- [ ] 3.3.4 Implement entropy estimation from KG data
- [ ] 3.3.5 Implement covariate ranking by information gain
- [ ] 3.3.6 Implement stopping criterion (diminishing returns)
- [ ] 3.3.7 Test on KG with known information content
- [ ] 3.3.8 Validate: IG correlates with prediction improvement
- [ ] 3.3.9 Integrate with search query generation
- [ ] 3.3.10 Benchmark: correlation between IG and actual value
- [ ] 3.3.11 Optimize entropy estimators
- [ ] 3.3.12 Document uncertainty quantification methods

### Sprint 3.4: Optimal Search Designer (15 tasks)
- [ ] 3.4.1 Create `causal_reasoning/optimal_search.py`
- [ ] 3.4.2 Implement `OptimalSearchDesigner` class
- [ ] 3.4.3 Implement Fisher Information computation
- [ ] 3.4.4 Implement D-optimality criterion (maximize det(FI))
- [ ] 3.4.5 Implement A-optimality criterion (minimize trace(Cov))
- [ ] 3.4.6 Implement E-optimality criterion (maximize min eigenvalue)
- [ ] 3.4.7 Implement search query design (maximize info gain)
- [ ] 3.4.8 Identify missing links in KG
- [ ] 3.4.9 Prioritize high-impact gaps
- [ ] 3.4.10 Generate optimal literature search queries
- [ ] 3.4.11 Implement adaptive search strategy
- [ ] 3.4.12 Test on KG with known gaps
- [ ] 3.4.13 Benchmark: search efficiency vs random search
- [ ] 3.4.14 Integrate with dynamic KG enrichment
- [ ] 3.4.15 Document optimal experimental design theory

---

## Phase 4: Dynamic KG Enrichment (Weeks 5-6)

### Sprint 4.1: Dynamic KG Manager (18 tasks)
- [ ] 4.1.1 Create `kg_management/dynamic_kg_manager.py`
- [ ] 4.1.2 Implement `DynamicKGManager` class
- [ ] 4.1.3 Implement KG loading with metadata
- [ ] 4.1.4 Implement KG saving with version control
- [ ] 4.1.5 Implement incremental update logic
- [ ] 4.1.6 Implement knowledge gap identification
- [ ] 4.1.7 Detect missing nodes (materials, properties)
- [ ] 4.1.8 Detect missing edges (causal relationships)
- [ ] 4.1.9 Detect low-confidence edges
- [ ] 4.1.10 Detect contradictory edges
- [ ] 4.1.11 Implement enrichment trigger logic
- [ ] 4.1.12 Implement confidence decay over time
- [ ] 4.1.13 Implement enrichment logging
- [ ] 4.1.14 Test on static KG (no changes)
- [ ] 4.1.15 Test incremental updates
- [ ] 4.1.16 Test version rollback
- [ ] 4.1.17 Benchmark: KG growth rate
- [ ] 4.1.18 Document KG management API

### Sprint 4.2: Literature Searcher (15 tasks)
- [ ] 4.2.1 Create `kg_management/literature_searcher.py`
- [ ] 4.2.2 Implement `LiteratureSearcher` class
- [ ] 4.2.3 Integrate OpenAlex API client
- [ ] 4.2.4 Integrate Semantic Scholar API client
- [ ] 4.2.5 Integrate arXiv API client (optional)
- [ ] 4.2.6 Implement search query construction
- [ ] 4.2.7 Template: "{cause} effect on {effect} in {material}"
- [ ] 4.2.8 Implement paper ranking by relevance
- [ ] 4.2.9 Implement result deduplication
- [ ] 4.2.10 Implement rate limiting / polite API usage
- [ ] 4.2.11 Test on known papers (ground truth)
- [ ] 4.2.12 Measure precision@10, recall@10
- [ ] 4.2.13 Optimize query templates
- [ ] 4.2.14 Add caching for search results
- [ ] 4.2.15 Document search API integration

### Sprint 4.3: Ollama Relation Extractor (12 tasks)
- [ ] 4.3.1 Create `kg_management/relation_extractor.py`
- [ ] 4.3.2 Implement `OllamaRelationExtractor` class
- [ ] 4.3.3 Reuse `extract_robust.py` prompt logic
- [ ] 4.3.4 Implement targeted extraction (specific gaps)
- [ ] 4.3.5 Implement general extraction (broad coverage)
- [ ] 4.3.6 Add retry logic for extraction failures
- [ ] 4.3.7 Add validation of extracted relations
- [ ] 4.3.8 Test on abstracts with known relations
- [ ] 4.3.9 Measure extraction precision/recall
- [ ] 4.3.10 Optimize prompts for targeted extraction
- [ ] 4.3.11 Add batch processing
- [ ] 4.3.12 Document extraction pipeline

### Sprint 4.4: KG Validator (10 tasks)
- [ ] 4.4.1 Create `kg_management/kg_validator.py`
- [ ] 4.4.2 Implement `KGValidator` class
- [ ] 4.4.3 Validate relation structure (required fields)
- [ ] 4.4.4 Check for contradictions with existing KG
- [ ] 4.4.5 Compute confidence scores for new relations
- [ ] 4.4.6 Check source quality (journal impact factor, citations)
- [ ] 4.4.7 Detect duplicates
- [ ] 4.4.8 Filter low-quality relations
- [ ] 4.4.9 Test on synthetic data (good/bad relations)
- [ ] 4.4.10 Document validation criteria

### Sprint 4.5: Integration (15 tasks)
- [ ] 4.5.1 Create `ARIAWithDynamicKG` class in aria_full_ollama.py
- [ ] 4.5.2 Integrate DynamicKGManager
- [ ] 4.5.3 Integrate LiteratureSearcher
- [ ] 4.5.4 Integrate OllamaRelationExtractor
- [ ] 4.5.5 Integrate KGValidator
- [ ] 4.5.6 Implement on-demand enrichment logic
- [ ] 4.5.7 Add enrichment trigger (low confidence < 0.5)
- [ ] 4.5.8 Add enrichment trigger (missing path)
- [ ] 4.5.9 Add enrichment trigger (user request)
- [ ] 4.5.10 Implement retry after enrichment
- [ ] 4.5.11 Test end-to-end enrichment workflow
- [ ] 4.5.12 Measure: KG growth per 100 queries
- [ ] 4.5.13 Measure: prediction improvement after enrichment
- [ ] 4.5.14 Optimize enrichment latency (<30s target)
- [ ] 4.5.15 Document enrichment workflow

---

## Phase 5: Evaluation & Benchmarking (Week 6)

### Sprint 5.1: Rigorous Ablation Framework (20 tasks)
- [ ] 5.1.1 Create `evaluation/rigorous_ablation.py`
- [ ] 5.1.2 Implement unified test harness for all 6 variants
- [ ] 5.1.3 Load/create test cases (forward + inverse)
- [ ] 5.1.4 Run baseline_ollama on all test cases
- [ ] 5.1.5 Run kg_only on all test cases
- [ ] 5.1.6 Run naive_kg_ollama on all test cases
- [ ] 5.1.7 Run aria_core_ollama on all test cases
- [ ] 5.1.8 Run aria_search_ollama on all test cases
- [ ] 5.1.9 Run aria_full_ollama on all test cases
- [ ] 5.1.10 Compute accuracy metrics (similarity to ground truth)
- [ ] 5.1.11 Compute confidence calibration (Pearson correlation)
- [ ] 5.1.12 Compute robustness (perturbation analysis)
- [ ] 5.1.13 Compute reasoning quality (human eval rubric)
- [ ] 5.1.14 Compute computational cost (latency, tokens)
- [ ] 5.1.15 Compute KG utilization (% paths used)
- [ ] 5.1.16 Generate comparison tables
- [ ] 5.1.17 Generate visualization plots
- [ ] 5.1.18 Statistical significance testing (t-tests, ANOVA)
- [ ] 5.1.19 Save results to CSV
- [ ] 5.1.20 Document evaluation methodology

### Sprint 5.2: Component Contribution Analysis (15 tasks)
- [ ] 5.2.1 Create `evaluation/component_analysis.py`
- [ ] 5.2.2 Compute KG contribution: naive_kg - baseline
- [ ] 5.2.3 Compute tier contribution: aria_core - naive_kg
- [ ] 5.2.4 Compute search contribution: aria_search - aria_core
- [ ] 5.2.5 Compute CoT contribution: aria_full - aria_search
- [ ] 5.2.6 Compute ToT contribution: aria_full(ToT) - aria_full(no-ToT)
- [ ] 5.2.7 Compute MCTS contribution: similar comparison
- [ ] 5.2.8 Compute dynamic KG contribution: similar comparison
- [ ] 5.2.9 Generate contribution breakdown chart
- [ ] 5.2.10 Identify which components matter most
- [ ] 5.2.11 Test for interaction effects (components synergy)
- [ ] 5.2.12 Perform sensitivity analysis
- [ ] 5.2.13 Create LaTeX table for paper
- [ ] 5.2.14 Save detailed analysis to JSON
- [ ] 5.2.15 Document findings

### Sprint 5.3: Failure Analysis (10 tasks)
- [ ] 5.3.1 Create `evaluation/failure_analysis.py`
- [ ] 5.3.2 Categorize failures: Missing KG knowledge
- [ ] 5.3.3 Categorize failures: Incorrect reasoning
- [ ] 5.3.4 Categorize failures: Poor transfer learning
- [ ] 5.3.5 Categorize failures: Search failure
- [ ] 5.3.6 Categorize failures: Extraction errors
- [ ] 5.3.7 Analyze failure patterns per variant
- [ ] 5.3.8 Identify systematic issues
- [ ] 5.3.9 Generate failure report
- [ ] 5.3.10 Document failure mitigation strategies

---

## Phase 6: Documentation & Polish (Week 7)

### Sprint 6.1: Code Documentation (15 tasks)
- [ ] 6.1.1 Add docstrings to all classes
- [ ] 6.1.2 Add docstrings to all methods
- [ ] 6.1.3 Add type hints throughout
- [ ] 6.1.4 Write README for each module
- [ ] 6.1.5 Create API reference documentation
- [ ] 6.1.6 Add usage examples for each variant
- [ ] 6.1.7 Document configuration options
- [ ] 6.1.8 Document prompt templates
- [ ] 6.1.9 Document evaluation metrics
- [ ] 6.1.10 Create quickstart guide
- [ ] 6.1.11 Create troubleshooting guide
- [ ] 6.1.12 Add inline comments for complex logic
- [ ] 6.1.13 Generate API docs with Sphinx
- [ ] 6.1.14 Review all documentation for clarity
- [ ] 6.1.15 Proofread and finalize

### Sprint 6.2: Testing & Quality Assurance (12 tasks)
- [ ] 6.2.1 Write unit tests for ollama_client
- [ ] 6.2.2 Write unit tests for each variant
- [ ] 6.2.3 Write unit tests for causal reasoning components
- [ ] 6.2.4 Write unit tests for KG management components
- [ ] 6.2.5 Write integration tests (end-to-end)
- [ ] 6.2.6 Run pytest with coverage report (>80% target)
- [ ] 6.2.7 Fix all failing tests
- [ ] 6.2.8 Add regression tests for known issues
- [ ] 6.2.9 Performance testing (latency benchmarks)
- [ ] 6.2.10 Memory profiling (identify leaks)
- [ ] 6.2.11 Code linting (flake8, black)
- [ ] 6.2.12 Security review (API keys, injection risks)

### Sprint 6.3: Packaging & Deployment (8 tasks)
- [ ] 6.3.1 Create requirements.txt with all dependencies
- [ ] 6.3.2 Create setup.py for package installation
- [ ] 6.3.3 Add configuration file support (YAML/JSON)
- [ ] 6.3.4 Create Docker container for reproducibility
- [ ] 6.3.5 Write deployment instructions
- [ ] 6.3.6 Create demo Jupyter notebook
- [ ] 6.3.7 Prepare release notes
- [ ] 6.3.8 Tag version 2.0.0

---

## 🎯 Total Tasks: 398

### By Phase:
- Phase 1: 50 tasks (12.6%)
- Phase 2: 38 tasks (9.5%)
- Phase 3: 60 tasks (15.1%)
- Phase 4: 70 tasks (17.6%)
- Phase 5: 45 tasks (11.3%)
- Phase 6: 35 tasks (8.8%)

### Critical Path (Must Complete):
1. Ollama client (Foundation)
2. All 6 variants (Ablation rigor)
3. Tree of Thought (Advanced reasoning)
4. Dynamic KG Manager (Persistent learning)
5. Rigorous evaluation (Scientific validation)

### Optional Enhancements:
- MCTS Explorer (if time permits)
- Uncertainty Quantifier (advanced feature)
- Optimal Search Designer (research contribution)

---

## 🚀 Quick Start Instructions

To begin implementation:

```bash
cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/26ARIA/26KDD/engine

# Create new directories
mkdir -p causal_reasoning
mkdir -p kg_management
mkdir -p evaluation

# Start with Phase 1, Sprint 1.1
# Create ollama_client.py and begin implementing
```

---

## 📊 Progress Tracking

Update this file as tasks are completed. Use:
- [x] for completed tasks
- [ ] for pending tasks
- [~] for in-progress tasks
- [!] for blocked tasks

Track time spent per sprint to estimate remaining effort.

---

This TODO list will guide the complete transformation of ARIA into a state-of-the-art causal reasoning system!
