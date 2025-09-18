import json
import networkx as nx
import os
import google.generativeai as genai
from google.generativeai.types import Tool
from pathlib import Path
import textwrap
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import re
from copy import deepcopy
import traceback

class KnowledgeGraphEnrichmentEngine:
    """
    Enhanced engine specifically designed to enrich knowledge graphs by filling null values
    using comprehensive literature search and validation.
    """
    
    def __init__(self, model_id: str = "gemini-1.5-pro-latest"):
        self._configure_api(model_id)
        print("Knowledge Graph Enrichment Engine initialized.")

    def _configure_api(self, model_id: str):
        api_key = os.getenv("GOOGLE_API_KEY")
        # Enhanced grounding tool for comprehensive search
        grounding_tool = Tool(google_search_retrieval={
            'dynamic_retrieval_config': {
                'mode': 'MODE_DYNAMIC',
                'dynamic_threshold': 0.7
            }
        })

        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id, tools=[grounding_tool])

    def _safe_str_conversion(self, value: Any) -> str:
        """
        Safely convert any value to string, handling various data types.
        """
        if value is None:
            return ""
        elif isinstance(value, (list, tuple)):
            # Handle lists/tuples by joining their string representations
            return " ".join(str(item) for item in value if item is not None)
        elif isinstance(value, dict):
            # Handle dictionaries by extracting meaningful values
            meaningful_values = []
            for k, v in value.items():
                if v is not None and str(v).lower() not in ["null", "none", ""]:
                    meaningful_values.append(str(v))
            return " ".join(meaningful_values)
        else:
            return str(value)

    def _extract_nested_values(self, obj: Any, exclude_keys: set = None) -> List[str]:
        """
        Recursively extract all non-null values from nested dictionaries and lists.
        """
        if exclude_keys is None:
            exclude_keys = {"experiment_id", "source_file", "enrichment_metadata"}
        
        values = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in exclude_keys:
                    continue
                if value is not None:
                    if isinstance(value, (dict, list)):
                        values.extend(self._extract_nested_values(value, exclude_keys))
                    else:
                        str_val = self._safe_str_conversion(value)
                        if str_val and str_val.lower() not in ["null", "none", ""]:
                            values.append(str_val)
        elif isinstance(obj, list):
            for item in obj:
                if item is not None:
                    if isinstance(item, (dict, list)):
                        values.extend(self._extract_nested_values(item, exclude_keys))
                    else:
                        str_val = self._safe_str_conversion(item)
                        if str_val and str_val.lower() not in ["null", "none", ""]:
                            values.append(str_val)
        
        return values

    def analyze_missing_data(self, kg_data: Dict) -> Dict[str, Any]:
        """
        Analyze the knowledge graph to identify missing data patterns and prioritize enrichment.
        """
        missing_analysis = {
            "total_experiments": len(kg_data.get("doping_experiments", [])),
            "total_relationships": len(kg_data.get("causal_relationships", [])),
            "missing_data_summary": {},
            "enrichment_priorities": [],
            "material_systems": set(),
            "dopant_elements": set()
        }
        
        # Analyze experiments
        experiments = kg_data.get("doping_experiments", [])
        missing_fields = {}
        
        for exp in experiments:
            # Track material systems and dopants with safe extraction
            try:
                host_material = exp.get("host_material")
                if host_material:
                    missing_analysis["material_systems"].add(self._safe_str_conversion(host_material))
                
                dopant_element = exp.get("dopant", {})
                if isinstance(dopant_element, dict):
                    element = dopant_element.get("element")
                    if element:
                        missing_analysis["dopant_elements"].add(self._safe_str_conversion(element))
            except Exception as e:
                print(f"   ⚠️ Warning: Error extracting material info from experiment: {e}")
            
            # Count missing fields
            try:
                self._count_missing_fields(exp, missing_fields, "")
            except Exception as e:
                print(f"   ⚠️ Warning: Error counting missing fields: {e}")
        
        missing_analysis["missing_data_summary"] = missing_fields
        missing_analysis["material_systems"] = list(missing_analysis["material_systems"])
        missing_analysis["dopant_elements"] = list(missing_analysis["dopant_elements"])
        
        # Prioritize enrichment based on importance and searchability
        try:
            missing_analysis["enrichment_priorities"] = self._prioritize_enrichment(missing_fields)
        except Exception as e:
            print(f"   ⚠️ Warning: Error prioritizing enrichment: {e}")
            missing_analysis["enrichment_priorities"] = []
        
        return missing_analysis

    def _count_missing_fields(self, obj: Any, missing_count: Dict, prefix: str):
        """Recursively count missing (null) fields in the data structure."""
        try:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "enrichment_metadata":  # Skip metadata fields
                        continue
                    current_path = f"{prefix}.{key}" if prefix else key
                    if value is None:
                        missing_count[current_path] = missing_count.get(current_path, 0) + 1
                    elif isinstance(value, (dict, list)):
                        self._count_missing_fields(value, missing_count, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{prefix}[{i}]"
                    self._count_missing_fields(item, missing_count, current_path)
        except Exception as e:
            print(f"   ⚠️ Warning: Error in _count_missing_fields: {e}")

    def _prioritize_enrichment(self, missing_fields: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Prioritize which fields to enrich based on importance and likelihood of finding data.
        """
        # Define importance weights for different field categories
        importance_weights = {
            "synthesis_conditions": 0.9,  # Very important and often available
            "property_changes.electronic": 0.95,  # Critical for doping studies
            "property_changes.thermal": 0.7,
            "property_changes.mechanical": 0.6,
            "property_changes.optical": 0.7,
            "doping_outcome": 0.85,
            "characterization_evidence": 0.8,
            "dopant.precursor": 0.6,
            "mechanism_quote": 0.9  # Very important for understanding
        }
        
        priorities = []
        for field, count in missing_fields.items():
            try:
                # Calculate priority score
                base_score = count  # More missing instances = higher priority
                
                # Apply importance weighting
                importance = 0.5  # Default importance
                for category, weight in importance_weights.items():
                    if category in field:
                        importance = weight
                        break
                
                priority_score = base_score * importance
                
                priorities.append({
                    "field": field,
                    "missing_count": count,
                    "importance": importance,
                    "priority_score": priority_score,
                    "searchability": self._assess_searchability(field)
                })
            except Exception as e:
                print(f"   ⚠️ Warning: Error processing field {field}: {e}")
        
        # Sort by priority score (descending)
        priorities.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
        return priorities

    def _assess_searchability(self, field: str) -> str:
        """Assess how likely we are to find this data through literature search."""
        high_searchability = [
            "synthesis_conditions.method",
            "synthesis_conditions.temperature_c",
            "property_changes.electronic.carrier_type",
            "property_changes.electronic.band_gap_ev",
            "dopant.precursor"
        ]
        
        medium_searchability = [
            "synthesis_conditions.time_hours",
            "synthesis_conditions.atmosphere",
            "property_changes.electronic.mobility_cm2_v_s",
            "property_changes.thermal",
            "doping_outcome.site_distribution"
        ]
        
        if any(pattern in field for pattern in high_searchability):
            return "high"
        elif any(pattern in field for pattern in medium_searchability):
            return "medium"
        else:
            return "low"

    def enrich_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single experiment by filling missing values through literature search.
        """
        try:
            exp_id = experiment.get('experiment_id', 'unknown')
            host_material = self._safe_str_conversion(experiment.get('host_material', 'unknown'))
            dopant_element = self._safe_str_conversion(experiment.get('dopant', {}).get('element', 'unknown'))
            
            print(f"\n🔬 Enriching experiment: {exp_id}")
            print(f"   Material: {host_material}")
            print(f"   Dopant: {dopant_element}")
            
            enriched_exp = deepcopy(experiment)
            
            # Add enrichment metadata
            enriched_exp["enrichment_metadata"] = {
                "enrichment_timestamp": datetime.now().isoformat(),
                "enriched_fields": [],
                "search_sources": [],
                "confidence_scores": {},
                "errors": []
            }
            
            # Generate comprehensive search queries for this experiment
            try:
                search_context = self._build_search_context(experiment)
            except Exception as e:
                print(f"   ⚠️ Error building search context: {e}")
                enriched_exp["enrichment_metadata"]["errors"].append(f"Search context error: {e}")
                search_context = {"search_string": f"{host_material} {dopant_element} doping"}
            
            # Enrich different categories of missing data
            try:
                enriched_exp = self._enrich_synthesis_conditions(enriched_exp, search_context)
            except Exception as e:
                print(f"   ⚠️ Error enriching synthesis conditions: {e}")
                enriched_exp["enrichment_metadata"]["errors"].append(f"Synthesis enrichment error: {e}")
            
            try:
                enriched_exp = self._enrich_electronic_properties(enriched_exp, search_context)
            except Exception as e:
                print(f"   ⚠️ Error enriching electronic properties: {e}")
                enriched_exp["enrichment_metadata"]["errors"].append(f"Electronic properties error: {e}")
            
            try:
                enriched_exp = self._enrich_doping_outcomes(enriched_exp, search_context)
            except Exception as e:
                print(f"   ⚠️ Error enriching doping outcomes: {e}")
                enriched_exp["enrichment_metadata"]["errors"].append(f"Doping outcomes error: {e}")
            
            try:
                enriched_exp = self._enrich_other_properties(enriched_exp, search_context)
            except Exception as e:
                print(f"   ⚠️ Error enriching other properties: {e}")
                enriched_exp["enrichment_metadata"]["errors"].append(f"Other properties error: {e}")
            
            return enriched_exp
            
        except Exception as e:
            print(f"   ❌ Critical error enriching experiment: {e}")
            traceback.print_exc()
            
            # Return original experiment with error metadata
            error_exp = deepcopy(experiment)
            error_exp["enrichment_metadata"] = {
                "enrichment_timestamp": datetime.now().isoformat(),
                "critical_error": str(e),
                "enriched_fields": [],
                "search_sources": [],
                "confidence_scores": {}
            }
            return error_exp

    def _build_search_context(self, experiment: Dict[str, Any]) -> Dict[str, str]:
        """Build search context from available experiment data with robust parsing."""
        context = {}
        
        try:
            # Extract key identifiers with safe conversion
            context["host_material"] = self._safe_str_conversion(experiment.get("host_material", ""))
            
            dopant_info = experiment.get("dopant", {})
            if isinstance(dopant_info, dict):
                context["dopant_element"] = self._safe_str_conversion(dopant_info.get("element", ""))
                context["dopant_concentration"] = self._safe_str_conversion(dopant_info.get("concentration", ""))
            else:
                context["dopant_element"] = ""
                context["dopant_concentration"] = ""
            
            # Extract known synthesis info
            synth_conditions = experiment.get("synthesis_conditions", {})
            if isinstance(synth_conditions, dict):
                context["synthesis_method"] = self._safe_str_conversion(synth_conditions.get("method", ""))
                context["temperature"] = self._safe_str_conversion(synth_conditions.get("temperature_c", ""))
            else:
                context["synthesis_method"] = ""
                context["temperature"] = ""
            
            # Extract known outcomes
            property_changes = experiment.get("property_changes", {})
            if isinstance(property_changes, dict):
                electronic = property_changes.get("electronic", {})
                if isinstance(electronic, dict):
                    context["carrier_type"] = self._safe_str_conversion(electronic.get("carrier_type", ""))
                else:
                    context["carrier_type"] = ""
            else:
                context["carrier_type"] = ""
            
            doping_outcome = experiment.get("doping_outcome", {})
            if isinstance(doping_outcome, dict):
                site_dist = doping_outcome.get("site_distribution", {})
                if isinstance(site_dist, dict):
                    context["primary_site"] = self._safe_str_conversion(site_dist.get("primary_site", ""))
                else:
                    context["primary_site"] = ""
            else:
                context["primary_site"] = ""
            
            # Build search string with safe joining
            search_terms = []
            for key, value in context.items():
                if value and str(value).lower() not in ["null", "none", "", "unknown"]:
                    search_terms.append(str(value))
            
            context["search_string"] = " ".join(search_terms) if search_terms else "doping experiment"
            
            return context
            
        except Exception as e:
            print(f"   ⚠️ Error in _build_search_context: {e}")
            # Return minimal context
            return {
                "host_material": "unknown",
                "dopant_element": "unknown", 
                "dopant_concentration": "",
                "synthesis_method": "",
                "temperature": "",
                "carrier_type": "",
                "primary_site": "",
                "search_string": "materials doping experiment"
            }

    def _enrich_synthesis_conditions(self, experiment: Dict[str, Any], search_context: Dict[str, str]) -> Dict[str, Any]:
        """Enrich synthesis conditions through targeted literature search."""
        
        try:
            synth_conditions = experiment.get("synthesis_conditions", {})
            if not isinstance(synth_conditions, dict):
                synth_conditions = {}
            
            missing_synth_fields = [k for k, v in synth_conditions.items() if v is None]
            
            if not missing_synth_fields:
                return experiment
            
            print(f"   🔍 Searching for synthesis conditions: {missing_synth_fields}")
            
            prompt = f"""
            You are a materials science literature expert. Search for detailed synthesis conditions for the following doping experiment:

            **Experiment Details:**
            - Host Material: {search_context.get('host_material', 'unknown')}
            - Dopant: {search_context.get('dopant_element', 'unknown')} at {search_context.get('dopant_concentration', 'unknown')}
            - Known Method: {search_context.get('synthesis_method', 'Unknown')}
            - Known Temperature: {search_context.get('temperature', 'Unknown')}

            **CRITICAL REQUIREMENTS:**
            1. **MANDATORY SEARCH**: You MUST use your search tool to find experimental papers
            2. **SPECIFIC DATA**: Look for exact synthesis parameters, not general methods
            3. **RECENT STUDIES**: Prioritize papers from 2020-2024
            4. **MULTIPLE SOURCES**: Find at least 2-3 independent studies if possible

            **Missing Information to Find:**
            {', '.join(missing_synth_fields)}

            **Search Strategy:**
            - Search for: "{search_context.get('dopant_element', '')} doped {search_context.get('host_material', '')} synthesis"
            - Search for: "{search_context.get('host_material', '')} {search_context.get('dopant_element', '')} experimental conditions"
            - Search for: "synthesis parameters {search_context.get('host_material', '')} doping"

            **Output Requirements:**
            Provide a detailed JSON response with found synthesis parameters and their sources:

            ```json
            {{
              "synthesis_conditions": {{
                "method": {{"value": "specific method", "source": "Author et al. 2024", "confidence": "high"}},
                "temperature_c": {{"value": 800, "source": "Author et al. 2024", "confidence": "high"}},
                "time_hours": {{"value": 2, "source": "Author et al. 2024", "confidence": "medium"}},
                "atmosphere": {{"value": "Ar", "source": "Author et al. 2024", "confidence": "high"}},
                "pressure_pa": {{"value": 101325, "source": "Author et al. 2024", "confidence": "medium"}},
                "additional_parameters": {{"value": "specific details", "source": "Author et al. 2024", "confidence": "medium"}}
              }},
              "search_summary": {{
                "total_sources_found": 3,
                "most_relevant_studies": ["Study 1", "Study 2", "Study 3"],
                "data_availability": "high",
                "notes": "Any important observations about the data"
              }}
            }}
            ```

            **REMEMBER**: Every value must be backed by a specific, cited source from your search!
            """
            
            response = self.model.generate_content(prompt)
            result = self._extract_json_from_response(response.text)
            
            if result and "synthesis_conditions" in result:
                # Update experiment with found data
                experiment = self._update_experiment_with_sources(
                    experiment, result["synthesis_conditions"], "synthesis_conditions", response
                )
                
        except Exception as e:
            print(f"   ⚠️ Error enriching synthesis conditions: {e}")
        
        return experiment

    def _enrich_electronic_properties(self, experiment: Dict[str, Any], search_context: Dict[str, str]) -> Dict[str, Any]:
        """Enrich electronic properties through targeted literature search."""
        
        try:
            property_changes = experiment.get("property_changes", {})
            if not isinstance(property_changes, dict):
                return experiment
            
            electronic_props = property_changes.get("electronic", {})
            if not isinstance(electronic_props, dict):
                return experiment
            
            missing_electronic_fields = []
            
            # Check for missing electronic properties
            for key, value in electronic_props.items():
                if value is None:
                    missing_electronic_fields.append(key)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subvalue is None:
                            missing_electronic_fields.append(f"{key}.{subkey}")
            
            if not missing_electronic_fields:
                return experiment
            
            print(f"   🔍 Searching for electronic properties: {missing_electronic_fields}")
            
            prompt = f"""
            You are a materials science expert specializing in electronic properties of doped materials. 
            Search for detailed electronic characterization data for this doping system:

            **System Information:**
            - Host Material: {search_context.get('host_material', 'unknown')}
            - Dopant: {search_context.get('dopant_element', 'unknown')} at {search_context.get('dopant_concentration', 'unknown')}
            - Known Carrier Type: {search_context.get('carrier_type', 'Unknown')}
            - Doping Site: {search_context.get('primary_site', 'Unknown')}

            **SEARCH REQUIREMENTS:**
            1. **MANDATORY LITERATURE SEARCH**: Use your search tool extensively
            2. **QUANTITATIVE DATA**: Focus on numerical values with units
            3. **EXPERIMENTAL VALIDATION**: Prioritize experimental over theoretical data
            4. **RECENT RESEARCH**: Emphasize studies from 2020-2024

            **Missing Electronic Properties to Find:**
            {', '.join(missing_electronic_fields)}

            **Targeted Search Queries:**
            - "{search_context.get('dopant_element', '')} doped {search_context.get('host_material', '')} electronic properties"
            - "{search_context.get('host_material', '')} {search_context.get('dopant_element', '')} carrier concentration mobility"
            - "{search_context.get('host_material', '')} {search_context.get('dopant_element', '')} band gap fermi level"
            - "{search_context.get('host_material', '')} doping electrical characterization"

            **Output Format:**
            ```json
            {{
              "electronic_properties": {{
                "carrier_concentration": {{
                  "value": "1.2e14 cm^-2", 
                  "source": "Smith et al. 2024", 
                  "measurement_method": "Hall effect",
                  "confidence": "high"
                }},
                "mobility_cm2_v_s": {{
                  "value": 150, 
                  "source": "Johnson et al. 2023", 
                  "measurement_method": "field effect",
                  "confidence": "medium"
                }},
                "band_gap_ev": {{
                  "value": 1.2, 
                  "source": "Brown et al. 2024", 
                  "measurement_method": "optical absorption",
                  "confidence": "high"
                }},
                "conductivity_s_cm": {{
                  "value": 100, 
                  "source": "Davis et al. 2023", 
                  "measurement_method": "four-probe",
                  "confidence": "high"
                }},
                "fermi_level_shift_ev": {{
                  "value": 0.3, 
                  "source": "Wilson et al. 2024", 
                  "measurement_method": "photoemission spectroscopy",
                  "confidence": "medium"
                }}
              }},
              "search_summary": {{
                "experimental_studies_found": 5,
                "theoretical_studies_found": 3,
                "data_quality": "high",
                "measurement_techniques": ["Hall effect", "photoemission", "optical"],
                "notes": "Key observations about data consistency"
              }}
            }}
            ```
            """
            
            response = self.model.generate_content(prompt)
            result = self._extract_json_from_response(response.text)
            
            if result and "electronic_properties" in result:
                # Update experiment with found data
                experiment = self._update_experiment_with_sources(
                    experiment, result["electronic_properties"], "property_changes.electronic", response
                )
                
        except Exception as e:
            print(f"   ⚠️ Error enriching electronic properties: {e}")
        
        return experiment

    def _enrich_doping_outcomes(self, experiment: Dict[str, Any], search_context: Dict[str, str]) -> Dict[str, Any]:
        """Enrich doping outcome information through literature search."""
        
        try:
            doping_outcome = experiment.get("doping_outcome", {})
            if not isinstance(doping_outcome, dict):
                return experiment
            
            missing_outcome_fields = []
            
            # Check for missing doping outcome data
            for category, data in doping_outcome.items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        if value is None:
                            missing_outcome_fields.append(f"{category}.{key}")
            
            if not missing_outcome_fields:
                return experiment
            
            print(f"   🔍 Searching for doping outcomes: {missing_outcome_fields}")
            
            prompt = f"""
            You are an expert in materials characterization and doping mechanisms. 
            Search for detailed information about doping outcomes and structural changes:

            **Doping System:**
            - Host: {search_context.get('host_material', 'unknown')}
            - Dopant: {search_context.get('dopant_element', 'unknown')} ({search_context.get('dopant_concentration', 'unknown')})
            - Known Primary Site: {search_context.get('primary_site', 'Unknown')}

            **SEARCH FOCUS:**
            1. **STRUCTURAL CHARACTERIZATION**: XRD, STEM, TEM studies
            2. **SITE DETERMINATION**: Spectroscopy, diffraction analysis
            3. **DISTRIBUTION ANALYSIS**: Mapping, depth profiling
            4. **RECENT STUDIES**: Prioritize 2020-2024 publications

            **Missing Information:**
            {', '.join(missing_outcome_fields)}

            **Search Queries:**
            - "{search_context.get('dopant_element', '')} {search_context.get('host_material', '')} structural characterization"
            - "{search_context.get('host_material', '')} doping site distribution XRD STEM"
            - "{search_context.get('dopant_element', '')} substitutional interstitial {search_context.get('host_material', '')}"
            - "{search_context.get('host_material', '')} lattice parameter doping"

            **Output Format:**
            ```json
            {{
              "doping_outcomes": {{
                "site_distribution": {{
                  "substitutional": {{"value": 85, "unit": "%", "source": "Author 2024", "method": "XRD Rietveld"}},
                  "interstitial": {{"value": 15, "unit": "%", "source": "Author 2024", "method": "STEM"}},
                  "site_specificity": {{"value": "Mo sites preferred", "source": "Author 2024", "confidence": "high"}}
                }},
                "structural_changes": {{
                  "lattice_parameter_change": {{"value": "+0.02", "unit": "Å", "source": "Smith 2023", "method": "XRD"}},
                  "layer_spacing_change_angstrom": {{"value": 0.1, "source": "Johnson 2024", "method": "HRTEM"}},
                  "defect_formation": {{"value": "vacancy clusters", "source": "Brown 2024", "method": "STEM"}}
                }},
                "distribution_characteristics": {{
                  "uniformity": {{"value": "homogeneous", "source": "Davis 2023", "method": "EDS mapping"}},
                  "penetration_depth": {{"value": "5 nm", "source": "Wilson 2024", "method": "depth profiling"}}
                }}
              }},
              "characterization_summary": {{
                "techniques_used": ["XRD", "STEM", "TEM", "EDS"],
                "studies_found": 4,
                "data_consistency": "high"
              }}
            }}
            ```
            """
            
            response = self.model.generate_content(prompt)
            result = self._extract_json_from_response(response.text)
            
            if result and "doping_outcomes" in result:
                experiment = self._update_experiment_with_sources(
                    experiment, result["doping_outcomes"], "doping_outcome", response
                )
                
        except Exception as e:
            print(f"   ⚠️ Error enriching doping outcomes: {e}")
        
        return experiment

    def _enrich_other_properties(self, experiment: Dict[str, Any], search_context: Dict[str, str]) -> Dict[str, Any]:
        """Enrich thermal, mechanical, and optical properties."""
        
        try:
            property_changes = experiment.get("property_changes", {})
            if not isinstance(property_changes, dict):
                return experiment
            
            categories_to_enrich = ["thermal", "mechanical", "optical"]
            
            for category in categories_to_enrich:
                try:
                    category_data = property_changes.get(category, {})
                    if not isinstance(category_data, dict):
                        continue
                    
                    missing_fields = [k for k, v in category_data.items() if v is None]
                    
                    if missing_fields:
                        print(f"   🔍 Searching for {category} properties: {missing_fields}")
                        
                        prompt = f"""
                        Search for {category} properties of {search_context.get('dopant_element', '')} doped {search_context.get('host_material', '')}:

                        **Missing {category.title()} Properties:**
                        {', '.join(missing_fields)}

                        **Search Queries:**
                        - "{search_context.get('dopant_element', '')} doped {search_context.get('host_material', '')} {category} properties"
                        - "{search_context.get('host_material', '')} {search_context.get('dopant_element', '')} {category} characterization"

                        **Output Format:**
                        ```json
                        {{
                          "{category}_properties": {{
                            // Include found properties with sources and confidence
                          }}
                        }}
                        ```
                        """
                        
                        response = self.model.generate_content(prompt)
                        result = self._extract_json_from_response(response.text)
                        
                        if result and f"{category}_properties" in result:
                            experiment = self._update_experiment_with_sources(
                                experiment, result[f"{category}_properties"], f"property_changes.{category}", response
                            )
                            
                except Exception as e:
                    print(f"   ⚠️ Error enriching {category} properties: {e}")
        
        except Exception as e:
            print(f"   ⚠️ Error in _enrich_other_properties: {e}")
        
        return experiment

    def _update_experiment_with_sources(self, experiment: Dict[str, Any], new_data: Dict[str, Any], 
                                      field_path: str, response) -> Dict[str, Any]:
        """Update experiment data with new information and track sources."""
        
        try:
            # Navigate to the correct field in the experiment
            path_parts = field_path.split('.')
            current = experiment
            
            # Navigate to the parent of the target field
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Update the target field
            target_field = path_parts[-1]
            if target_field not in current:
                current[target_field] = {}
            
            # Track enrichment metadata
            enrichment_meta = experiment.get("enrichment_metadata", {})
            
            # Extract sources from response metadata if available
            sources = self._extract_sources_from_response(response)
            
            # Update fields and track changes
            for key, value_info in new_data.items():
                try:
                    if isinstance(value_info, dict) and "value" in value_info:
                        # Store the actual value
                        current[target_field][key] = value_info["value"]
                        
                        # Track enrichment metadata
                        enrichment_meta["enriched_fields"].append(f"{field_path}.{key}")
                        enrichment_meta["confidence_scores"][f"{field_path}.{key}"] = value_info.get("confidence", "medium")
                        
                        # Add source information
                        if "source" in value_info:
                            source_info = {
                                "field": f"{field_path}.{key}",
                                "source": value_info["source"],
                                "method": value_info.get("measurement_method", ""),
                                "confidence": value_info.get("confidence", "medium")
                            }
                            enrichment_meta["search_sources"].append(source_info)
                except Exception as e:
                    print(f"   ⚠️ Error updating field {key}: {e}")
            
            # Add web sources from search metadata
            if sources:
                enrichment_meta["web_sources"] = sources
            
            experiment["enrichment_metadata"] = enrichment_meta
            
        except Exception as e:
            print(f"   ⚠️ Error in _update_experiment_with_sources: {e}")
        
        return experiment

    def _extract_sources_from_response(self, response) -> List[Dict[str, str]]:
        """Extract source information from Gemini response metadata."""
        sources = []
        
        try:
            if response.candidates and response.candidates[0].grounding_metadata:
                metadata = response.candidates[0].grounding_metadata
                
                if hasattr(metadata, 'grounding_chunks'):
                    for chunk in metadata.grounding_chunks:
                        sources.append({
                            "title": chunk.web.title,
                            "uri": chunk.web.uri,
                            "timestamp": datetime.now().isoformat()
                        })
        except Exception as e:
            print(f"   ⚠️ Error extracting sources: {e}")
        
        return sources

    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """Extract JSON from model response with robust error handling."""
        try:
            import re
            
            # Match the JSON block
            match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)
            
            if not match:
                return {}
            
            json_string = match.group(1)
            
            try:
                return json.loads(json_string)
            except json.JSONDecodeError as e:
                print(f"   ⚠️ JSON decode error: {e}")
                return {}
        except Exception as e:
            print(f"   ⚠️ Error extracting JSON: {e}")
            return {}

    def enrich_causal_relationships(self, relationships: List[Dict[str, Any]], 
                                  material_systems: List[str]) -> List[Dict[str, Any]]:
        """Enrich causal relationships by filling missing mechanism quotes and other data."""
        
        enriched_relationships = []
        
        for rel in relationships:
            try:
                rel_id = rel.get('relationship_id', 'unknown')
                print(f"\n🔗 Enriching relationship: {rel_id}")
                
                enriched_rel = deepcopy(rel)
                
                # Add enrichment metadata
                enriched_rel["enrichment_metadata"] = {
                    "enrichment_timestamp": datetime.now().isoformat(),
                    "enriched_fields": [],
                    "search_sources": [],
                    "errors": []
                }
                
                # Focus on missing mechanism quotes
                if not rel.get("mechanism_quote"):
                    try:
                        enriched_rel = self._enrich_mechanism_quote(enriched_rel, material_systems)
                    except Exception as e:
                        print(f"   ⚠️ Error enriching mechanism quote: {e}")
                        enriched_rel["enrichment_metadata"]["errors"].append(f"Mechanism quote error: {e}")
                
                # Enrich competing processes if missing
                if not rel.get("competing_processes"):
                    try:
                        enriched_rel = self._enrich_competing_processes(enriched_rel)
                    except Exception as e:
                        print(f"   ⚠️ Error enriching competing processes: {e}")
                        enriched_rel["enrichment_metadata"]["errors"].append(f"Competing processes error: {e}")
                
                enriched_relationships.append(enriched_rel)
                
            except Exception as e:
                print(f"   ❌ Critical error enriching relationship: {e}")
                # Add original relationship with error metadata
                error_rel = deepcopy(rel)
                error_rel["enrichment_metadata"] = {
                    "enrichment_timestamp": datetime.now().isoformat(),
                    "critical_error": str(e),
                    "enriched_fields": [],
                    "search_sources": []
                }
                enriched_relationships.append(error_rel)
        
        return enriched_relationships

    def _enrich_mechanism_quote(self, relationship: Dict[str, Any], 
                               material_systems: List[str]) -> Dict[str, Any]:
        """Search for mechanism quotes for causal relationships."""
        
        try:
            cause = self._safe_str_conversion(relationship.get("cause_parameter", ""))
            effect = self._safe_str_conversion(relationship.get("effect_on_doping", ""))
            property_affected = self._safe_str_conversion(relationship.get("affected_property", ""))
            
            print(f"   🔍 Searching for mechanism: {cause} → {effect}")
            
            # Build search context
            materials_context = " ".join(material_systems[:3]) if material_systems else "materials"
            
            prompt = f"""
            You are a materials science expert. Search for detailed mechanistic explanations for this causal relationship:

            **Causal Relationship:**
            - Cause: {cause}
            - Effect: {effect}
            - Affected Property: {property_affected}
            - Material Context: {materials_context}

            **SEARCH REQUIREMENTS:**
            1. **MECHANISTIC FOCUS**: Look for papers that explain the underlying physics/chemistry
            2. **DIRECT QUOTES**: Find actual quotes from papers that explain the mechanism
            3. **RECENT LITERATURE**: Prioritize 2020-2024 studies
            4. **MULTIPLE PERSPECTIVES**: Look for different mechanistic explanations

            **Search Queries:**
            - "{cause} {effect} mechanism materials science"
            - "{cause} {property_affected} mechanistic explanation"
            - "mechanism {effect} {materials_context}"

            **Output Format:**
            ```json
            {{
              "mechanism_data": {{
                "primary_mechanism_quote": {{
                  "quote": "Exact quote from paper explaining the mechanism",
                  "source": "Author et al. 2024",
                  "context": "Additional context about the mechanism"
                }},
                "alternative_mechanisms": [
                  {{
                    "quote": "Alternative explanation quote",
                    "source": "Different Author 2023",
                    "mechanism_type": "electronic/structural/thermodynamic"
                  }}
                ],
                "competing_processes": [
                  "Process 1 that competes",
                  "Process 2 that interferes"
                ]
              }},
              "search_quality": {{
                "mechanistic_papers_found": 3,
                "direct_quotes_available": true,
                "consensus_level": "high"
              }}
            }}
            ```
            """
            
            response = self.model.generate_content(prompt)
            result = self._extract_json_from_response(response.text)
            
            if result and "mechanism_data" in result:
                mechanism_data = result["mechanism_data"]
                
                # Update mechanism quote
                if "primary_mechanism_quote" in mechanism_data:
                    primary = mechanism_data["primary_mechanism_quote"]
                    relationship["mechanism_quote"] = primary.get("quote", "")
                    
                    # Track enrichment
                    enrichment_meta = relationship["enrichment_metadata"]
                    enrichment_meta["enriched_fields"].append("mechanism_quote")
                    enrichment_meta["search_sources"].append({
                        "field": "mechanism_quote",
                        "source": primary.get("source", ""),
                        "context": primary.get("context", "")
                    })
                
                # Update competing processes
                if "competing_processes" in mechanism_data:
                    relationship["competing_processes"] = mechanism_data["competing_processes"]
                    enrichment_meta["enriched_fields"].append("competing_processes")
                
                # Add web sources
                sources = self._extract_sources_from_response(response)
                if sources:
                    enrichment_meta["web_sources"] = sources
                    
        except Exception as e:
            print(f"   ⚠️ Error enriching mechanism: {e}")
        
        return relationship

    def _enrich_competing_processes(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        """Search for competing processes that might affect the causal relationship."""
        # Implementation similar to mechanism enrichment but focused on competing processes
        return relationship

    def enrich_knowledge_graph(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        Main function to enrich the entire knowledge graph with robust error handling.
        """
        print("🚀 Starting Knowledge Graph Enrichment Process")
        print("="*60)
        
        try:
            # Load original knowledge graph
            with open(input_file, 'r', encoding='utf-8') as f:
                original_kg = json.load(f)
        except Exception as e:
            print(f"❌ Error loading input file: {e}")
            raise
        
        # Analyze missing data
        print("\n📊 Analyzing missing data patterns...")
        try:
            missing_analysis = self.analyze_missing_data(original_kg)
            
            print(f"   - Total experiments: {missing_analysis['total_experiments']}")
            print(f"   - Total relationships: {missing_analysis['total_relationships']}")
            print(f"   - Material systems: {len(missing_analysis['material_systems'])}")
            print(f"   - Top missing fields: {[p.get('field', 'unknown') for p in missing_analysis['enrichment_priorities'][:5]]}")
        except Exception as e:
            print(f"⚠️ Error analyzing missing data: {e}")
            missing_analysis = {"total_experiments": 0, "total_relationships": 0, "material_systems": []}
        
        # Create enriched copy
        enriched_kg = deepcopy(original_kg)
        
        # Add enrichment metadata to the entire KG
        enriched_kg["enrichment_metadata"] = {
            "original_file": input_file,
            "enrichment_timestamp": datetime.now().isoformat(),
            "enrichment_engine_version": "1.1",
            "missing_data_analysis": missing_analysis,
            "total_experiments_enriched": 0,
            "total_relationships_enriched": 0,
            "errors": []
        }
        
        # Enrich experiments
        experiments = original_kg.get("doping_experiments", [])
        print(f"\n🧪 Enriching {len(experiments)} experiments...")
        enriched_experiments = []
        
        for i, experiment in enumerate(experiments):
            try:
                print(f"\n--- Experiment {i+1}/{len(experiments)} ---")
                enriched_exp = self.enrich_experiment(experiment)
                enriched_experiments.append(enriched_exp)
                
                # Update progress
                if "enriched_fields" in enriched_exp.get("enrichment_metadata", {}):
                    if enriched_exp["enrichment_metadata"]["enriched_fields"]:
                        enriched_kg["enrichment_metadata"]["total_experiments_enriched"] += 1
                        
            except Exception as e:
                print(f"   ❌ Critical error with experiment {i+1}: {e}")
                enriched_kg["enrichment_metadata"]["errors"].append(f"Experiment {i+1}: {e}")
                # Add original experiment
                enriched_experiments.append(experiment)
        
        enriched_kg["doping_experiments"] = enriched_experiments
        
        # Enrich causal relationships
        relationships = original_kg.get("causal_relationships", [])
        print(f"\n🔗 Enriching {len(relationships)} causal relationships...")
        try:
            enriched_relationships = self.enrich_causal_relationships(
                relationships,
                missing_analysis.get("material_systems", [])
            )
            enriched_kg["causal_relationships"] = enriched_relationships
        except Exception as e:
            print(f"⚠️ Error enriching relationships: {e}")
            enriched_kg["enrichment_metadata"]["errors"].append(f"Relationships enrichment: {e}")
            enriched_kg["causal_relationships"] = relationships
        
        # Calculate enrichment statistics
        try:
            total_enriched_fields = 0
            total_sources_added = 0
            
            for exp in enriched_experiments:
                meta = exp.get("enrichment_metadata", {})
                total_enriched_fields += len(meta.get("enriched_fields", []))
                total_sources_added += len(meta.get("search_sources", []))
            
            total_missing_fields = sum(missing_analysis.get("missing_data_summary", {}).values())
            success_rate = total_enriched_fields / max(1, total_missing_fields)
            
            enriched_kg["enrichment_metadata"]["enrichment_statistics"] = {
                "total_fields_enriched": total_enriched_fields,
                "total_sources_added": total_sources_added,
                "enrichment_success_rate": success_rate,
                "total_missing_fields": total_missing_fields
            }
        except Exception as e:
            print(f"⚠️ Error calculating statistics: {e}")
            enriched_kg["enrichment_metadata"]["enrichment_statistics"] = {
                "total_fields_enriched": 0,
                "total_sources_added": 0,
                "enrichment_success_rate": 0.0
            }
        
        # Save enriched knowledge graph
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enriched_kg, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving output file: {e}")
            raise
        
        # Print final summary
        stats = enriched_kg["enrichment_metadata"]["enrichment_statistics"]
        print(f"\n✅ Enrichment Complete!")
        print(f"   - Fields enriched: {stats['total_fields_enriched']}")
        print(f"   - Sources added: {stats['total_sources_added']}")
        print(f"   - Success rate: {stats['enrichment_success_rate']:.2%}")
        print(f"   - Output saved to: {output_file}")
        
        return enriched_kg


def main():
    """
    Main function to run the knowledge graph enrichment process.
    """
    
    # Initialize the enrichment engine
    try:
        enrichment_engine = KnowledgeGraphEnrichmentEngine()
    except Exception as e:
        print(f"❌ Error initializing enrichment engine: {e}")
        return
    
    # Define input and output files
    # input_file = "../outputs/filtered_combined_doping_data.json"  # Your input file
    # output_file = "../outputs/enriched_knowledge_graph.json"  # Output file
    input_file = "../outputs/filtered_test_doping_data.json"  # Your input file for the test
    output_file = "../outputs/enriched_test_knowledge_graph.json"  # Output file for the test
    # Run the enrichment process
    try:
        enriched_kg = enrichment_engine.enrich_knowledge_graph(input_file, output_file)
        
        # Print summary
        print("\n" + "="*60)
        print("ENRICHMENT SUMMARY")
        print("="*60)
        
        meta = enriched_kg["enrichment_metadata"]
        stats = meta["enrichment_statistics"]
        
        print(f"Original experiments: {meta['missing_data_analysis']['total_experiments']}")
        print(f"Experiments enriched: {meta['total_experiments_enriched']}")
        print(f"Total fields enriched: {stats['total_fields_enriched']}")
        print(f"Total sources added: {stats['total_sources_added']}")
        print(f"Success rate: {stats['enrichment_success_rate']:.2%}")
        
        if meta.get("errors"):
            print(f"\nErrors encountered: {len(meta['errors'])}")
            for error in meta["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
    except Exception as e:
        print(f"❌ Error during enrichment: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()