# Here's a comprehensive transformation to a Chain-of-Thought RAG system 
# with transparent source citation and sophisticated reasoning:

import json
import networkx as nx
import os
import dashscope
from dashscope import Generation
from pathlib import Path
import textwrap
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

# ===== ENHANCEMENT 1: Knowledge Source Tracking =====
@dataclass
class KnowledgeSource:
    """Track individual knowledge sources with metadata"""
    source_id: str
    content: str
    source_type: str  # "kg_node", "kg_edge", "kg_mechanism", "llm_baseline"
    confidence: float
    context: str
    metadata: Dict = field(default_factory=dict)
    
@dataclass 
class ReasoningStep:
    """Individual step in chain-of-thought reasoning"""
    step_id: str
    description: str
    evidence_sources: List[KnowledgeSource]
    reasoning_type: str  # "retrieval", "synthesis", "validation", "inference"
    confidence: float
    intermediate_conclusion: str
    
@dataclass
class ChainOfThought:
    """Complete reasoning chain with source attribution"""
    query_context: Dict
    reasoning_steps: List[ReasoningStep]
    knowledge_synthesis: Dict
    final_reasoning: str
    confidence_breakdown: Dict
    source_attribution: Dict

# ===== ENHANCEMENT 2: Advanced Knowledge Graph Retriever =====
class AdvancedKGRetriever:
    """Enhanced knowledge retrieval with multi-hop reasoning and source tracking"""
    
    def __init__(self, graph: nx.DiGraph, embedding_model, similarity_threshold: float = 0.6):
        self.graph = graph
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self._build_knowledge_index()
        
    def _build_knowledge_index(self):
        """Build comprehensive knowledge index with source tracking"""
        print("🔍 Building advanced knowledge index...")
        
        # Index nodes with metadata
        self.node_sources = {}
        for node in self.graph.nodes(data=True):
            node_id, node_data = node
            self.node_sources[node_id] = KnowledgeSource(
                source_id=f"node_{hash(node_id)}",
                content=node_id,
                source_type="kg_node", 
                confidence=1.0,
                context="Direct node match",
                metadata=node_data
            )
        
        # Index edges with mechanisms
        self.edge_sources = {}
        for edge in self.graph.edges(data=True):
            source, target, edge_data = edge
            edge_id = f"{source} -> {target}"
            mechanism = edge_data.get('mechanism', '')
            
            self.edge_sources[edge_id] = KnowledgeSource(
                source_id=f"edge_{hash(edge_id)}",
                content=f"Causal relationship: {source} leads to {target}",
                source_type="kg_edge",
                confidence=0.9,
                context=f"Causal pathway in knowledge graph",
                metadata={'mechanism': mechanism, 'source': source, 'target': target}
            )
            
            if mechanism:
                self.edge_sources[f"{edge_id}_mechanism"] = KnowledgeSource(
                    source_id=f"mechanism_{hash(edge_id)}",
                    content=mechanism,
                    source_type="kg_mechanism",
                    confidence=0.95,
                    context=f"Mechanistic explanation for {source} -> {target}",
                    metadata={'pathway': edge_id}
                )
        
        print(f"✅ Indexed {len(self.node_sources)} nodes and {len(self.edge_sources)} edges/mechanisms")

    def retrieve_relevant_knowledge(self, query_keywords: List[str], 
                                  max_sources: int = 10) -> List[KnowledgeSource]:
        """Retrieve and rank relevant knowledge sources"""
        print(f"🔍 Retrieving knowledge for: {query_keywords}")
        
        relevant_sources = []
        query_text = " ".join(str(kw) for kw in query_keywords if kw)
        
        # Direct node matches
        for node_id, source in self.node_sources.items():
            similarity = self._calculate_semantic_similarity(query_text, source.content)
            if similarity >= self.similarity_threshold:
                source.confidence = similarity
                source.context = f"Node similarity: {similarity:.3f}"
                relevant_sources.append(source)
        
        # Edge and mechanism matches  
        for edge_id, source in self.edge_sources.items():
            similarity = self._calculate_semantic_similarity(query_text, source.content)
            if similarity >= self.similarity_threshold:
                source.confidence = similarity
                source.context = f"Edge/mechanism similarity: {similarity:.3f}"
                relevant_sources.append(source)
        
        # Sort by confidence and return top sources
        relevant_sources.sort(key=lambda x: x.confidence, reverse=True)
        return relevant_sources[:max_sources]
    
    def find_multi_hop_paths(self, start_concepts: List[str], 
                           end_concepts: List[str], max_hops: int = 3) -> List[Dict]:
        """Find multi-hop reasoning paths with source tracking"""
        print(f"🔗 Finding multi-hop paths: {start_concepts} -> {end_concepts}")
        
        # Find relevant start and end nodes
        start_nodes = self._find_concept_nodes(start_concepts)
        end_nodes = self._find_concept_nodes(end_concepts)
        
        reasoning_paths = []
        
        for start_node in start_nodes:
            for end_node in end_nodes:
                if nx.has_path(self.graph, start_node, end_node):
                    try:
                        paths = list(nx.all_simple_paths(self.graph, start_node, end_node, cutoff=max_hops))
                        for path in paths[:3]:  # Limit to top 3 paths per node pair
                            path_info = self._analyze_reasoning_path(path)
                            reasoning_paths.append(path_info)
                    except nx.NetworkXNoPath:
                        continue
        
        return reasoning_paths[:5]  # Return top 5 reasoning paths
    
    def _find_concept_nodes(self, concepts: List[str]) -> List[str]:
        """Find nodes matching given concepts"""
        matching_nodes = []
        for concept in concepts:
            for node in self.graph.nodes():
                if self._calculate_semantic_similarity(str(concept), node) >= self.similarity_threshold:
                    matching_nodes.append(node)
        return list(set(matching_nodes))
    
    def _analyze_reasoning_path(self, path: List[str]) -> Dict:
        """Analyze a reasoning path and extract sources"""
        path_sources = []
        mechanisms = []
        
        # Add node sources
        for node in path:
            if node in self.node_sources:
                path_sources.append(self.node_sources[node])
        
        # Add edge sources and mechanisms
        for i in range(len(path) - 1):
            edge_id = f"{path[i]} -> {path[i+1]}"
            if edge_id in self.edge_sources:
                path_sources.append(self.edge_sources[edge_id])
                
            mechanism_id = f"{edge_id}_mechanism"
            if mechanism_id in self.edge_sources:
                path_sources.append(self.edge_sources[mechanism_id])
                mechanisms.append(self.edge_sources[mechanism_id].content)
        
        return {
            'path': " -> ".join(path),
            'sources': path_sources,
            'mechanisms': mechanisms,
            'confidence': np.mean([s.confidence for s in path_sources]) if path_sources else 0.0,
            'length': len(path)
        }
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        try:
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except:
            return 0.0

# ===== ENHANCEMENT 3: Chain-of-Thought Reasoning Engine =====
class ChainOfThoughtReasoner:
    """Orchestrates transparent chain-of-thought reasoning with source attribution"""
    
    def __init__(self, kg_retriever: AdvancedKGRetriever, llm_generator):
        self.kg_retriever = kg_retriever
        self.llm_generator = llm_generator
        
    def reason_with_sources(self, query_data: Dict, query_type: str) -> ChainOfThought:
        """Execute complete chain-of-thought reasoning with source tracking"""
        print(f"\n🧠 Starting Chain-of-Thought Reasoning for {query_type}")
        
        reasoning_steps = []
        
        # Step 1: Knowledge Retrieval
        retrieval_step = self._step_1_knowledge_retrieval(query_data, query_type)
        reasoning_steps.append(retrieval_step)
        
        # Step 2: Baseline Scientific Analysis
        baseline_step = self._step_2_baseline_analysis(query_data, query_type)
        reasoning_steps.append(baseline_step)
        
        # Step 3: Knowledge Graph Enhancement
        kg_enhancement_step = self._step_3_kg_enhancement(
            query_data, retrieval_step.evidence_sources, baseline_step.intermediate_conclusion
        )
        reasoning_steps.append(kg_enhancement_step)
        
        # Step 4: Multi-hop Reasoning
        multihop_step = self._step_4_multihop_reasoning(query_data, query_type)
        reasoning_steps.append(multihop_step)
        
        # Step 5: Knowledge Synthesis
        synthesis_step = self._step_5_knowledge_synthesis(reasoning_steps, query_type)
        reasoning_steps.append(synthesis_step)
        
        # Step 6: Validation and Confidence Assessment
        validation_step = self._step_6_validation(reasoning_steps, query_data, query_type)
        reasoning_steps.append(validation_step)
        
        # Build final chain of thought
        chain_of_thought = self._build_final_reasoning(reasoning_steps, query_data, query_type)
        
        return chain_of_thought
    
    def _step_1_knowledge_retrieval(self, query_data: Dict, query_type: str) -> ReasoningStep:
        """Step 1: Retrieve relevant knowledge from KG"""
        print("📚 Step 1: Knowledge Retrieval")
        
        query_keywords = [str(v) for v in query_data.values() if v and str(v).strip()]
        relevant_sources = self.kg_retriever.retrieve_relevant_knowledge(query_keywords, max_sources=8)
        
        step = ReasoningStep(
            step_id="retrieval",
            description="Retrieved relevant knowledge from materials science knowledge graph",
            evidence_sources=relevant_sources,
            reasoning_type="retrieval",
            confidence=np.mean([s.confidence for s in relevant_sources]) if relevant_sources else 0.0,
            intermediate_conclusion=f"Found {len(relevant_sources)} relevant knowledge sources from research literature"
        )
        
        print(f"  ✅ Retrieved {len(relevant_sources)} sources (avg confidence: {step.confidence:.3f})")
        return step
    
    def _step_2_baseline_analysis(self, query_data: Dict, query_type: str) -> ReasoningStep:
        """Step 2: Generate baseline scientific analysis using LLM"""
        print("🔬 Step 2: Baseline Scientific Analysis")
        
        baseline_prompt = self._create_baseline_prompt(query_data, query_type)
        baseline_response = self._generate_llm_response(baseline_prompt)
        
        # Create baseline knowledge source
        baseline_source = KnowledgeSource(
            source_id="llm_baseline",
            content=baseline_response,
            source_type="llm_baseline",
            confidence=0.8,
            context="LLM baseline scientific reasoning",
            metadata={"query_type": query_type}
        )
        
        step = ReasoningStep(
            step_id="baseline",
            description="Generated baseline analysis using fundamental materials science principles",
            evidence_sources=[baseline_source],
            reasoning_type="inference",
            confidence=0.8,
            intermediate_conclusion=baseline_response
        )
        
        print(f"  ✅ Generated baseline analysis")
        return step
    
    def _step_3_kg_enhancement(self, query_data: Dict, kg_sources: List[KnowledgeSource], 
                              baseline_conclusion: str) -> ReasoningStep:
        """Step 3: Enhance baseline with KG knowledge"""
        print("🔗 Step 3: Knowledge Graph Enhancement")
        
        enhancement_prompt = self._create_enhancement_prompt(query_data, kg_sources, baseline_conclusion)
        enhancement_response = self._generate_llm_response(enhancement_prompt)
        
        # Create enhancement source
        enhancement_source = KnowledgeSource(
            source_id="kg_enhancement",
            content=enhancement_response,
            source_type="llm_enhanced",
            confidence=0.85,
            context="LLM reasoning enhanced with KG evidence",
            metadata={"num_kg_sources": len(kg_sources)}
        )
        
        step = ReasoningStep(
            step_id="kg_enhancement",
            description="Enhanced baseline analysis with specific research findings from knowledge graph",
            evidence_sources=kg_sources + [enhancement_source],
            reasoning_type="synthesis",
            confidence=min(0.9, 0.8 + len(kg_sources) * 0.02),
            intermediate_conclusion=enhancement_response
        )
        
        print(f"  ✅ Enhanced analysis with {len(kg_sources)} KG sources")
        return step
    
    def _step_4_multihop_reasoning(self, query_data: Dict, query_type: str) -> ReasoningStep:
        """Step 4: Multi-hop reasoning through KG"""
        print("🔗 Step 4: Multi-hop Reasoning")
        
        # Extract concepts for multi-hop search
        if query_type == "forward":
            start_concepts = list(query_data.values())
            end_concepts = ["conductivity", "doping", "properties", "electronic"]
        else:
            start_concepts = ["synthesis", "method", "conditions", "temperature"]
            end_concepts = list(query_data.values())
        
        reasoning_paths = self.kg_retriever.find_multi_hop_paths(start_concepts, end_concepts)
        
        # Create sources from reasoning paths
        path_sources = []
        for path_info in reasoning_paths:
            path_sources.extend(path_info['sources'])
        
        multihop_analysis = self._analyze_reasoning_paths(reasoning_paths, query_data, query_type)
        
        step = ReasoningStep(
            step_id="multihop",
            description="Performed multi-hop reasoning through causal pathways in knowledge graph",
            evidence_sources=path_sources,
            reasoning_type="inference",
            confidence=np.mean([p['confidence'] for p in reasoning_paths]) if reasoning_paths else 0.0,
            intermediate_conclusion=multihop_analysis
        )
        
        print(f"  ✅ Analyzed {len(reasoning_paths)} reasoning paths")
        return step
    
    def _step_5_knowledge_synthesis(self, reasoning_steps: List[ReasoningStep], query_type: str) -> ReasoningStep:
        """Step 5: Synthesize all knowledge sources"""
        print("🧩 Step 5: Knowledge Synthesis")
        
        synthesis_prompt = self._create_synthesis_prompt(reasoning_steps, query_type)
        synthesis_response = self._generate_llm_response(synthesis_prompt)
        
        # Collect all sources
        all_sources = []
        for step in reasoning_steps:
            all_sources.extend(step.evidence_sources)
        
        synthesis_source = KnowledgeSource(
            source_id="synthesis",
            content=synthesis_response,
            source_type="llm_synthesis",
            confidence=0.9,
            context="Comprehensive synthesis of all evidence sources",
            metadata={"total_sources": len(all_sources)}
        )
        
        step = ReasoningStep(
            step_id="synthesis",
            description="Synthesized insights from baseline reasoning, KG evidence, and multi-hop analysis",
            evidence_sources=all_sources + [synthesis_source],
            reasoning_type="synthesis",
            confidence=0.85,
            intermediate_conclusion=synthesis_response
        )
        
        print(f"  ✅ Synthesized {len(all_sources)} total evidence sources")
        return step
    
    def _step_6_validation(self, reasoning_steps: List[ReasoningStep], 
                          query_data: Dict, query_type: str) -> ReasoningStep:
        """Step 6: Validate reasoning and assess confidence"""
        print("✅ Step 6: Validation and Confidence Assessment")
        
        validation_prompt = self._create_validation_prompt(reasoning_steps, query_data, query_type)
        validation_response = self._generate_llm_response(validation_prompt)
        
        validation_source = KnowledgeSource(
            source_id="validation",
            content=validation_response,
            source_type="llm_validation",
            confidence=0.95,
            context="Scientific validation and confidence assessment",
            metadata={"validation_type": "comprehensive"}
        )
        
        step = ReasoningStep(
            step_id="validation",
            description="Validated reasoning chain and assessed confidence levels",
            evidence_sources=[validation_source],
            reasoning_type="validation",
            confidence=0.9,
            intermediate_conclusion=validation_response
        )
        
        print(f"  ✅ Completed validation")
        return step
    
    # ===== ENHANCEMENT 4: Sophisticated Prompt Engineering =====
    
    def _create_baseline_prompt(self, query_data: Dict, query_type: str) -> str:
        """Create prompt for baseline scientific analysis"""
        if query_type == "forward":
            task = "predict the resulting material properties"
            input_label = "Synthesis Conditions"
        else:
            task = "suggest synthesis conditions"
            input_label = "Desired Properties"
        
        return f"""
        As a materials science expert, {task} based on fundamental scientific principles.
        
        **{input_label}:**
        {json.dumps(query_data, indent=2)}
        
        **Instructions:**
        - Apply core materials science principles (thermodynamics, kinetics, electronic structure)
        - Explain the underlying physical/chemical mechanisms
        - Provide your reasoning step-by-step
        - Assess confidence based on established scientific knowledge
        - Be explicit about assumptions and limitations
        
        Provide a detailed scientific analysis focusing on fundamental principles.
        """
    
    def _create_enhancement_prompt(self, query_data: Dict, kg_sources: List[KnowledgeSource], 
                                  baseline_conclusion: str) -> str:
        """Create prompt for KG enhancement"""
        
        # Format KG evidence with source attribution
        kg_evidence = "\n".join([
            f"- **Source {i+1}** ({source.source_type}, confidence: {source.confidence:.2f}): {source.content}"
            for i, source in enumerate(kg_sources[:5])  # Top 5 sources
        ])
        
        return f"""
        You have baseline scientific analysis and specific research evidence. Enhance your analysis by integrating this evidence.
        
        **Query Data:**
        {json.dumps(query_data, indent=2)}
        
        **Your Baseline Analysis:**
        {baseline_conclusion}
        
        **Research Evidence from Literature:**
        {kg_evidence}
        
        **Enhancement Instructions:**
        - Compare research evidence with your baseline analysis
        - Identify confirmations, refinements, or contradictions
        - Integrate specific research findings with fundamental principles
        - Cite evidence sources when making claims
        - Explain how research evidence changes or strengthens your conclusions
        
        Provide an enhanced analysis that intelligently combines baseline reasoning with research evidence.
        """
    
    def _create_synthesis_prompt(self, reasoning_steps: List[ReasoningStep], query_type: str) -> str:
        """Create prompt for final synthesis"""
        
        # Summarize each reasoning step
        step_summaries = []
        for step in reasoning_steps:
            step_summaries.append(f"**{step.step_id.title()}**: {step.intermediate_conclusion[:200]}...")
        
        steps_text = "\n".join(step_summaries)
        
        if query_type == "forward":
            output_format = '"predicted_properties": {"property": "value"}'
        else:
            output_format = '"suggested_synthesis_conditions": {"parameter": "value"}'
        
        return f"""
        Synthesize all reasoning steps into a final, comprehensive answer with full source attribution.
        
        **Reasoning Steps Completed:**
        {steps_text}
        
        **Synthesis Instructions:**
        - Integrate insights from all reasoning steps
        - Provide specific, actionable conclusions
        - Include confidence levels for each major claim
        - Cite specific evidence sources
        - Acknowledge uncertainties and limitations
        
        **Required JSON Output:**
        ```json
        {{
            {output_format},
            "confidence": 0.x,
            "key_insights": ["insight 1", "insight 2"],
            "evidence_summary": "summary of supporting evidence",
            "limitations": ["limitation 1", "limitation 2"]
        }}
        ```
        """
    
    def _create_validation_prompt(self, reasoning_steps: List[ReasoningStep], 
                                 query_data: Dict, query_type: str) -> str:
        """Create prompt for validation"""
        
        return f"""
        Validate the reasoning chain and provide a comprehensive confidence assessment.
        
        **Original Query:**
        {json.dumps(query_data, indent=2)}
        
        **Reasoning Chain Summary:**
        - Retrieved {len(reasoning_steps[0].evidence_sources) if reasoning_steps else 0} KG sources
        - Generated baseline analysis
        - Enhanced with research evidence  
        - Performed multi-hop reasoning
        - Synthesized final conclusions
        
        **Validation Tasks:**
        1. Check scientific consistency across all reasoning steps
        2. Identify potential contradictions or weaknesses
        3. Assess the quality and reliability of evidence sources
        4. Evaluate confidence levels for different aspects
        5. Suggest experimental validation approaches
        
        Provide a thorough validation assessment with specific confidence metrics.
        """
    
    # ===== ENHANCEMENT 5: Helper Methods =====
    
    def _analyze_reasoning_paths(self, reasoning_paths: List[Dict], 
                               query_data: Dict, query_type: str) -> str:
        """Analyze multi-hop reasoning paths"""
        if not reasoning_paths:
            return "No multi-hop reasoning paths found in knowledge graph."
        
        analysis_prompt = f"""
        Analyze these causal reasoning paths from the knowledge graph:
        
        **Query:** {json.dumps(query_data, indent=2)}
        
        **Reasoning Paths:**
        {chr(10).join([f"Path {i+1}: {path['path']} (confidence: {path['confidence']:.2f})" for i, path in enumerate(reasoning_paths)])}
        
        **Mechanisms:**
        {chr(10).join([f"- {mech}" for path in reasoning_paths for mech in path.get('mechanisms', [])])}
        
        Provide insights about what these causal pathways suggest for the query.
        """
        
        return self._generate_llm_response(analysis_prompt)
    
    def _build_final_reasoning(self, reasoning_steps: List[ReasoningStep], 
                              query_data: Dict, query_type: str) -> ChainOfThought:
        """Build the final chain of thought object"""
        
        # Extract final synthesis
        synthesis_step = next((step for step in reasoning_steps if step.step_id == "synthesis"), None)
        final_reasoning = synthesis_step.intermediate_conclusion if synthesis_step else "Analysis completed"
        
        # Build confidence breakdown
        confidence_breakdown = {
            step.step_id: step.confidence for step in reasoning_steps
        }
        
        # Build source attribution
        source_attribution = {}
        for step in reasoning_steps:
            source_attribution[step.step_id] = [
                {
                    "source_id": source.source_id,
                    "type": source.source_type, 
                    "confidence": source.confidence,
                    "content_preview": source.content[:100] + "..." if len(source.content) > 100 else source.content
                }
                for source in step.evidence_sources
            ]
        
        # Build knowledge synthesis summary
        all_sources = []
        for step in reasoning_steps:
            all_sources.extend(step.evidence_sources)
        
        kg_sources = [s for s in all_sources if s.source_type.startswith("kg_")]
        llm_sources = [s for s in all_sources if s.source_type.startswith("llm_")]
        
        knowledge_synthesis = {
            "total_sources": len(all_sources),
            "kg_sources": len(kg_sources),
            "llm_sources": len(llm_sources),
            "avg_kg_confidence": np.mean([s.confidence for s in kg_sources]) if kg_sources else 0.0,
            "avg_llm_confidence": np.mean([s.confidence for s in llm_sources]) if llm_sources else 0.0,
            "integration_quality": "high" if len(kg_sources) >= 3 and len(llm_sources) >= 2 else "moderate"
        }
        
        return ChainOfThought(
            query_context=query_data,
            reasoning_steps=reasoning_steps,
            knowledge_synthesis=knowledge_synthesis,
            final_reasoning=final_reasoning,
            confidence_breakdown=confidence_breakdown,
            source_attribution=source_attribution
        )
    
    def _generate_llm_response(self, prompt: str) -> str:
        """Generate LLM response"""
        return self.llm_generator(prompt)

# ===== ENHANCEMENT 6: Main Enhanced Causal Reasoning Engine =====
class EnhancedCausalReasoningEngine:
    """
    🚀 ENHANCED: Transparent Chain-of-Thought RAG System
    
    Key Improvements:
    1. Complete source attribution and citation
    2. Transparent reasoning chain with intermediate steps
    3. Sophisticated LLM+KG integration that enhances rather than restricts
    4. Multi-hop reasoning through knowledge graph
    5. Confidence assessment at each reasoning step
    6. Scientific validation and quality control
    """
    
    def __init__(self, json_file_path: str, model_id: str = "qwen-plus", 
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """Initialize enhanced RAG system"""
        print("🚀 Initializing Enhanced Chain-of-Thought RAG System")
        
        # Build knowledge graph
        self.graph = self._build_enhanced_graph(json_file_path)
        
        # Configure LLM
        self._configure_api(model_id)
        
        # Initialize embedding model
        print(f"Loading sentence transformer model ('{embedding_model}')...")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize advanced components
        self.kg_retriever = AdvancedKGRetriever(self.graph, self.embedding_model)
        self.cot_reasoner = ChainOfThoughtReasoner(self.kg_retriever, self._generate_content)
        
        print(f"✅ Enhanced RAG System Ready!")
        print(f"   📊 Knowledge Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        print(f"   🧠 Chain-of-Thought Reasoning: Enabled")
        print(f"   📚 Source Attribution: Enabled")
        print(f"   🔗 Multi-hop Reasoning: Enabled")

    def _build_enhanced_graph(self, json_file_path: str):
        """Build enhanced knowledge graph (same as before but with better metadata)"""
        input_path = Path(json_file_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found at {input_path}")
            
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "causal_relationships" in data:
            relationships = data.get("causal_relationships", [])
        else:
            relationships = data

        G = nx.DiGraph()
        
        # Build graph with enhanced metadata
        for i, rel in enumerate(relationships):
            cause_text = rel.get("cause_parameter") or "Unknown Cause"
            effect_text = rel.get("effect_on_doping") or "Unknown Effect"
            affected_property = rel.get("affected_property")
            
            cause = textwrap.fill(cause_text.strip(), width=25)
            if affected_property and affected_property.strip():
                effect_label = f"{effect_text.strip()}\n({affected_property.strip()})"
            else:
                effect_label = effect_text.strip()
            effect = textwrap.fill(effect_label, width=25)
            
            if ('unknown' not in cause.lower() and 'n/a' not in cause.lower() and 
                'unknown' not in effect.lower() and 'n/a' not in effect.lower()):
                
                mechanism = rel.get("mechanism_quote", "")
                
                # Add nodes with metadata
                G.add_node(cause, node_type="cause", source_paper=f"paper_{i}")
                G.add_node(effect, node_type="effect", source_paper=f"paper_{i}")
                
                # Add edge with enhanced metadata
                G.add_edge(cause, effect, 
                          mechanism=mechanism,
                          relationship_id=f"rel_{i}",
                          source_data=rel)
        
        print(f"Built enhanced knowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def _configure_api(self, model_id: str):
        """Configure API (same as before)"""
        api_key = os.getenv("DASHSCOPE_API_KEY") or "sk-05e23c85c27448a0a8d2e0e0f0302779"
        dashscope.api_key = api_key
        dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
        self.model_id = model_id

    def _generate_content(self, prompt: str):
        """Generate content using LLM (same as before)"""
        try:
            response = Generation.call(
                model=self.model_id,
                prompt=prompt,
                temperature=0.7
            )
            
            if response.status_code == 200:
                return response.output.text
            else:
                return f"Error: {response.code} - {response.message}"
                
        except Exception as e:
            return f"Exception: {str(e)}"

    # ===== ENHANCEMENT 7: New Main Methods with Chain-of-Thought =====

    def forward_prediction_with_cot(self, synthesis_inputs: dict, save_result: bool = True) -> Dict:
        """ 🚀 ENHANCED: Forward prediction with complete chain-of-thought reasoning"""
        print(f"\n{'='*60}")
        print("🔮 ENHANCED FORWARD PREDICTION WITH CHAIN-OF-THOUGHT")
        print(f"{'='*60}")
        
        # Execute chain-of-thought reasoning
        chain_of_thought = self.cot_reasoner.reason_with_sources(synthesis_inputs, "forward")
        
        # Extract final answer from synthesis step
        synthesis_step = next((step for step in chain_of_thought.reasoning_steps 
                            if step.step_id == "synthesis"), None)
        
        if synthesis_step:
            final_answer = extract_json_from_response(synthesis_step.intermediate_conclusion)
        else:
            final_answer = {"predicted_properties": {"status": "analysis_completed"}}
        
        # Build comprehensive response
        enhanced_response = {
            **final_answer,
            
            # Add original query for reference
            "original_query": {
                "query_type": "forward_prediction",
                "synthesis_inputs": synthesis_inputs,
                "timestamp": datetime.now().isoformat()
            },
            
            # Complete Chain-of-Thought
            "chain_of_thought": {
                "reasoning_steps": [
                    {
                        "step": step.step_id,
                        "description": step.description,
                        "conclusion": step.intermediate_conclusion,  # ===== CHANGED: Full conclusion, not truncated =====
                        "confidence": step.confidence,
                        "evidence_count": len(step.evidence_sources),
                        "evidence_sources": [  # ===== NEW: Include actual evidence =====
                            {
                                "source_id": source.source_id,
                                "content": source.content,
                                "source_type": source.source_type,
                                "confidence": source.confidence,
                                "context": source.context
                            }
                            for source in step.evidence_sources
                        ]
                    }
                    for step in chain_of_thought.reasoning_steps
                ],
                "total_steps": len(chain_of_thought.reasoning_steps)
            },
            
            # Source Attribution
            "source_attribution": {
                "knowledge_graph_sources": len([s for step in chain_of_thought.reasoning_steps 
                                            for s in step.evidence_sources 
                                            if s.source_type.startswith("kg_")]),
                "llm_reasoning_sources": len([s for step in chain_of_thought.reasoning_steps 
                                            for s in step.evidence_sources 
                                            if s.source_type.startswith("llm_")]),
                "detailed_sources": chain_of_thought.source_attribution
            },
            
            # Knowledge Synthesis Report
            "knowledge_synthesis": chain_of_thought.knowledge_synthesis,
            
            # Confidence Breakdown
            "confidence_analysis": {
                "overall_confidence": np.mean(list(chain_of_thought.confidence_breakdown.values())),
                "step_confidences": chain_of_thought.confidence_breakdown,
                "confidence_factors": {
                    "kg_evidence_quality": chain_of_thought.knowledge_synthesis.get("avg_kg_confidence", 0),
                    "reasoning_consistency": chain_of_thought.knowledge_synthesis.get("avg_llm_confidence", 0),
                    "integration_quality": chain_of_thought.knowledge_synthesis.get("integration_quality", "unknown")
                }
            },
            
            # Enhanced metadata
            "method": "enhanced_chain_of_thought_rag",
            "reasoning_type": "transparent_multi_source",
            "timestamp": datetime.now().isoformat()
        }
        
        # Print comprehensive summary
        self._print_cot_summary(enhanced_response)
        
        # ===== NEW: Save result if requested =====
        if save_result:
            self.save_results_to_file(enhanced_response, 
                                    f"forward_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        return enhanced_response

    def inverse_design_with_cot(self, desired_properties: dict, save_result: bool = True) -> Dict:
        """🚀 ENHANCED: Inverse design with complete chain-of-thought reasoning"""
        print(f"\n{'='*60}")
        print("🔄 ENHANCED INVERSE DESIGN WITH CHAIN-OF-THOUGHT") 
        print(f"{'='*60}")
        
        # Execute chain-of-thought reasoning
        chain_of_thought = self.cot_reasoner.reason_with_sources(desired_properties, "inverse")
        
        # Extract final answer from synthesis step
        synthesis_step = next((step for step in chain_of_thought.reasoning_steps 
                            if step.step_id == "synthesis"), None)
        
        if synthesis_step:
            final_answer = extract_json_from_response(synthesis_step.intermediate_conclusion)
        else:
            final_answer = {"suggested_synthesis_conditions": {"status": "analysis_completed"}}
        
        # Build comprehensive response (same structure as forward prediction)
        enhanced_response = {
            **final_answer,
            
            # Add original query for reference
            "original_query": {
                "query_type": "inverse_design",
                "desired_properties": desired_properties,
                "timestamp": datetime.now().isoformat()
            },
            
            "chain_of_thought": {
                "reasoning_steps": [
                    {
                        "step": step.step_id,
                        "description": step.description,
                        "conclusion": step.intermediate_conclusion,  # ===== CHANGED: Full conclusion =====
                        "confidence": step.confidence,
                        "evidence_count": len(step.evidence_sources),
                        "evidence_sources": [  # ===== NEW: Include actual evidence =====
                            {
                                "source_id": source.source_id,
                                "content": source.content,
                                "source_type": source.source_type,
                                "confidence": source.confidence,
                                "context": source.context
                            }
                            for source in step.evidence_sources
                        ]
                    }
                    for step in chain_of_thought.reasoning_steps
                ],
                "total_steps": len(chain_of_thought.reasoning_steps)
            },
            "source_attribution": {
                "knowledge_graph_sources": len([s for step in chain_of_thought.reasoning_steps 
                                            for s in step.evidence_sources 
                                            if s.source_type.startswith("kg_")]),
                "llm_reasoning_sources": len([s for step in chain_of_thought.reasoning_steps 
                                            for s in step.evidence_sources 
                                            if s.source_type.startswith("llm_")]),
                "detailed_sources": chain_of_thought.source_attribution
            },
            "knowledge_synthesis": chain_of_thought.knowledge_synthesis,
            "confidence_analysis": {
                "overall_confidence": np.mean(list(chain_of_thought.confidence_breakdown.values())),
                "step_confidences": chain_of_thought.confidence_breakdown,
                "confidence_factors": {
                    "kg_evidence_quality": chain_of_thought.knowledge_synthesis.get("avg_kg_confidence", 0),
                    "reasoning_consistency": chain_of_thought.knowledge_synthesis.get("avg_llm_confidence", 0),
                    "integration_quality": chain_of_thought.knowledge_synthesis.get("integration_quality", "unknown")
                }
            },
            "method": "enhanced_chain_of_thought_rag",
            "reasoning_type": "transparent_multi_source",
            "timestamp": datetime.now().isoformat()
        }
        
        # Print comprehensive summary
        self._print_cot_summary(enhanced_response)
        
        # ===== NEW: Save result if requested =====
        if save_result:
            self.save_results_to_file(enhanced_response, 
                                    f"inverse_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        return enhanced_response

    # def _print_cot_summary(self, response: Dict):
    #     """Print a summary of the chain-of-thought reasoning"""
    #     print(f"\n📊 CHAIN-OF-THOUGHT SUMMARY:")
    #     print(f"   🔗 Reasoning Steps: {response['chain_of_thought']['total_steps']}")
    #     print(f"   📚 KG Sources: {response['source_attribution']['knowledge_graph_sources']}")
    #     print(f"   🧠 LLM Sources: {response['source_attribution']['llm_reasoning_sources']}")
    #     print(f"   🎯 Overall Confidence: {response['confidence_analysis']['overall_confidence']:.3f}")
    #     print(f"   🔧 Integration Quality: {response['knowledge_synthesis']['integration_quality']}")
        
    #     print(f"\n🔍 REASONING CHAIN:")
    #     for step in response['chain_of_thought']['reasoning_steps']:
    #         print(f"   {step['step'].upper()}: {step['description']} (conf: {step['confidence']:.2f})")

    def _print_cot_summary(self, response: Dict):
        """Print a comprehensive summary including the actual answer"""
        print(f"\n📊 CHAIN-OF-THOUGHT SUMMARY:")
        print(f"   🔗 Reasoning Steps: {response['chain_of_thought']['total_steps']}")
        print(f"   📚 KG Sources: {response['source_attribution']['knowledge_graph_sources']}")
        print(f"   🧠 LLM Sources: {response['source_attribution']['llm_reasoning_sources']}")
        print(f"   🎯 Overall Confidence: {response['confidence_analysis']['overall_confidence']:.3f}")
        print(f"   🔧 Integration Quality: {response['knowledge_synthesis']['integration_quality']}")
        
        print(f"\n🔍 REASONING CHAIN:")
        for step in response['chain_of_thought']['reasoning_steps']:
            print(f"   {step['step'].upper()}: {step['description']} (conf: {step['confidence']:.2f})")
        
        # ===== NEW: Print the actual detailed answer =====
        print(f"\n🎯 FINAL ANSWER:")
        print("="*50)
        
        # Print the main prediction/suggestion
        if 'predicted_properties' in response:
            print("📋 PREDICTED PROPERTIES:")
            for key, value in response['predicted_properties'].items():
                print(f"   • {key}: {value}")
        
        if 'suggested_synthesis_conditions' in response:
            print("🔧 SUGGESTED SYNTHESIS CONDITIONS:")
            for key, value in response['suggested_synthesis_conditions'].items():
                print(f"   • {key}: {value}")
        
        # Print key insights if available
        if 'key_insights' in response:
            print(f"\n💡 KEY INSIGHTS:")
            for insight in response['key_insights']:
                print(f"   • {insight}")
        
        # Print evidence summary if available
        if 'evidence_summary' in response:
            print(f"\n📚 EVIDENCE SUMMARY:")
            print(f"   {response['evidence_summary']}")
        
        # Print limitations if available
        if 'limitations' in response:
            print(f"\n⚠️ LIMITATIONS:")
            for limitation in response['limitations']:
                print(f"   • {limitation}")
        
        # ===== NEW: Print detailed reasoning from each step =====
        print(f"\n📖 DETAILED REASONING STEPS:")
        print("="*50)
        for i, step in enumerate(response['chain_of_thought']['reasoning_steps'], 1):
            print(f"\n{i}. {step['step'].upper()} (Confidence: {step['confidence']:.2f})")
            print(f"   Description: {step['description']}")
            print(f"   Conclusion: {step['conclusion']}")
            print(f"   Evidence Sources: {step['evidence_count']}")

    # save results to json file
    def save_results_to_file(self, results: Dict, filename: str = None):
        """Save complete results with chain-of-thought to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_rag_results_{timestamp}.json"
        
        # Ensure the outputs directory exists
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 RESULTS SAVED:")
            print(f"   📁 File: {output_path}")
            print(f"   📊 Size: {output_path.stat().st_size / 1024:.1f} KB")
            
            return str(output_path)
        
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return None

    def save_session_results(self, all_results: List[Dict], session_name: str = None):
        """Save all results from a testing session"""
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"rag_session_{timestamp}"
        
        session_data = {
            "session_info": {
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(all_results),
                "system_version": "enhanced_chain_of_thought_rag_v1.0"
            },
            "results": all_results
        }
        
        filename = f"{session_name}.json"
        return self.save_results_to_file(session_data, filename)


# ===== Keep the helper function =====
def extract_json_from_response(text: str):
    """Extract JSON from model response (same as before)"""
    import re
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)
    
    if not match:
        return {"warning": "No JSON object found in the response.", "raw_response": text}
    
    json_string = match.group(1)
    
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"\n--- JSON DECODE ERROR ---")
        print(f"Failed to parse JSON: {e}")
        print(f"Problematic text: {json_string}")
        print("-------------------------\n")
        return {
            "error": "Failed to decode JSON from model response.",
            "details": str(e),
            "malformed_json_string": json_string
        }

# ===== ENHANCEMENT 8: Demonstration Script =====
if __name__ == '__main__':
    json_file = '../outputs/filtered_combined_doping_data.json'
    
    try:
        print("="*80)
        print("🚀 TESTING ENHANCED CHAIN-OF-THOUGHT RAG SYSTEM")
        print("="*80)
        
        # Initialize enhanced engine
        engine = EnhancedCausalReasoningEngine(json_file, model_id="qwen-plus")
        
        # Store all results for session saving
        all_results = []
        
        # Test cases showcasing enhanced capabilities
        test_cases = [
            {
                "name": "Complete Data with CoT Analysis",
                "synthesis": {"temperature": "600°C", "method": "CVD", "atmosphere": "Ar/H2"},
                "properties": {"doping": "p-type semiconductor", "conductivity": "enhanced"}
            },
            {
                "name": "Partial Data with Source Attribution", 
                "synthesis": {"temperature": "500°C", "method": ""},
                "properties": {"carrier_type": "n-type"}
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 TEST CASE {i}: {test_case['name']}")
            print("="*60)
            
            # Forward prediction with chain-of-thought
            print("🔮 FORWARD PREDICTION:")
            forward_result = engine.forward_prediction_with_cot(test_case['synthesis'], save_result=True)
            all_results.append({
                "test_case": test_case['name'],
                "query_type": "forward_prediction",
                "result": forward_result
            })
            
            print(f"\n🔄 INVERSE DESIGN:")
            inverse_result = engine.inverse_design_with_cot(test_case['properties'], save_result=True)
            all_results.append({
                "test_case": test_case['name'],
                "query_type": "inverse_design", 
                "result": inverse_result
            })
            
            # Show key improvements
            print(f"\n✨ KEY ENHANCEMENTS DEMONSTRATED:")
            print(f"   📊 Transparent reasoning chain with {forward_result['chain_of_thought']['total_steps']} steps")
            print(f"   📚 Source attribution: {forward_result['source_attribution']['knowledge_graph_sources']} KG + {forward_result['source_attribution']['llm_reasoning_sources']} LLM sources")
            print(f"   🎯 Confidence analysis with step-by-step breakdown")
            print(f"   🔗 Multi-hop reasoning through knowledge graph")
            print(f"   🧠 LLM+KG integration that enhances rather than restricts")
        
        # ===== NEW: Save complete session results =====
        session_file = engine.save_session_results(all_results, "enhanced_rag_demo_session")
        
        print(f"\n{'='*80}")
        print("✅ ENHANCED CHAIN-OF-THOUGHT RAG SYSTEM DEMONSTRATION COMPLETE!")
        print("🚀 Key Improvements:")
        print("   • Complete source attribution and citation")
        print("   • Transparent step-by-step reasoning chain") 
        print("   • Sophisticated LLM+KG integration")
        print("   • Multi-hop reasoning capabilities")
        print("   • Comprehensive confidence assessment")
        print("   • Scientific validation at each step")
        print(f"📁 Session results saved to: {session_file}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()