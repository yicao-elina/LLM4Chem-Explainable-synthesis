"""
KG Normalization Pipeline

Purpose: Convert raw relations (kg_raw_2d_doping.json) → normalized causal KG

Steps:
1. Normalize entity text (temperature, method, phase, property)
2. Filter low-quality relations
3. Generate kg_example_2d_doping_enriched.json matching schema
4. Validate with build_graph.py

Output: kg_example_2d_doping_enriched.json
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

# ============= Configuration =============
INPUT_FILE = Path("outputs/kg_raw_2d_doping.json")
OUTPUT_FILE = Path("outputs/kg_example_2d_doping_enriched.json")
STATS_FILE = Path("outputs/normalization_stats.txt")

# ============= Normalization Rules =============

# Material name normalization
MATERIAL_ALIASES = {
    'mos2': 'MoS2',
    'ws2': 'WS2',
    'wse2': 'WSe2',
    'mose2': 'MoSe2',
    'bi2te3': 'Bi2Te3',
    'bi2se3': 'Bi2Se3',
    'sb2te3': 'Sb2Te3',
    'hbn': 'hBN',
    'h-bn': 'hBN',
    'hexagonal boron nitride': 'hBN',
    'graphene': 'graphene',
    'black phosphorus': 'black phosphorus',
}

# Synthesis method normalization
METHOD_ALIASES = {
    'cvd': 'CVD',
    'chemical vapor deposition': 'CVD',
    'mbe': 'MBE',
    'molecular beam epitaxy': 'MBE',
    'dft': 'DFT',
    'density functional theory': 'DFT',
    'first-principles': 'DFT',
    'ab initio': 'DFT',
    'molecular dynamics': 'MD simulation',
    'md': 'MD simulation',
    'monte carlo': 'Monte Carlo',
    'ion implantation': 'ion implantation',
    'thermal evaporation': 'thermal evaporation',
    'sputtering': 'sputtering',
    'electrochemical': 'electrochemical',
    'solid-state reaction': 'solid-state reaction',
    'hydrothermal': 'hydrothermal',
}

# Property normalization
PROPERTY_ALIASES = {
    'carrier concentration': 'carrier concentration',
    'carrier density': 'carrier concentration',
    'electron concentration': 'carrier concentration',
    'hole concentration': 'carrier concentration',
    'mobility': 'carrier mobility',
    'carrier mobility': 'carrier mobility',
    'electron mobility': 'carrier mobility',
    'hole mobility': 'carrier mobility',
    'conductivity': 'electrical conductivity',
    'electrical conductivity': 'electrical conductivity',
    'band gap': 'band gap',
    'bandgap': 'band gap',
    'energy gap': 'band gap',
    'seebeck coefficient': 'Seebeck coefficient',
    'thermal conductivity': 'thermal conductivity',
}

# Doping site normalization
SITE_ALIASES = {
    'substitutional': 'substitutional',
    'interstitial': 'interstitial',
    'surface': 'surface_adsorption',
    'surface adsorption': 'surface_adsorption',
    'vdw gap': 'vdw_gap',
    'van der waals gap': 'vdw_gap',
    'interlayer': 'vdw_gap',
}


def normalize_text(text: str, alias_dict: Dict[str, str]) -> str:
    """Normalize text using alias dictionary."""
    if not text or not isinstance(text, str):
        return text

    text_lower = text.lower().strip()

    # Direct match
    if text_lower in alias_dict:
        return alias_dict[text_lower]

    # Partial match (for materials in compound names)
    for alias, canonical in alias_dict.items():
        if alias in text_lower:
            return canonical

    return text


def normalize_temperature(temp_value) -> Optional[float]:
    """Normalize temperature to float (Celsius)."""
    if temp_value is None:
        return None

    if isinstance(temp_value, (int, float)):
        return float(temp_value)

    if isinstance(temp_value, str):
        # Extract number from string
        match = re.search(r'(\d+(?:\.\d+)?)', temp_value)
        if match:
            return float(match.group(1))

    return None


def normalize_concentration(conc_str: Optional[str]) -> Optional[str]:
    """Normalize concentration format."""
    if not conc_str:
        return None

    # Keep as-is if already formatted
    if any(unit in str(conc_str).lower() for unit in ['at%', 'wt%', 'cm^-3', 'cm-3']):
        return str(conc_str)

    return str(conc_str)


def is_valid_relation(relation: Dict) -> Tuple[bool, str]:
    """
    Validate if a causal relation is high quality.

    Returns:
        (is_valid, reason)
    """
    # Required fields
    if not relation.get('cause_parameter'):
        return False, "Missing cause_parameter"

    if not relation.get('effect_on_doping'):
        return False, "Missing effect_on_doping"

    if not relation.get('affected_property'):
        return False, "Missing affected_property"

    # Check for generic/vague terms
    vague_terms = ['unknown', 'n/a', 'not specified', 'unclear']

    cause = str(relation.get('cause_parameter', '')).lower()
    effect = str(relation.get('effect_on_doping', '')).lower()
    prop = str(relation.get('affected_property', '')).lower()

    if any(term in cause or term in effect or term in prop for term in vague_terms):
        return False, "Contains vague/unknown terms"

    # Check for meaningful content (not too short)
    if len(cause) < 3 or len(effect) < 3 or len(prop) < 3:
        return False, "Content too short/uninformative"

    return True, "Valid"


def normalize_experiment(exp: Dict) -> Dict:
    """Normalize a single experiment."""
    normalized = exp.copy()

    # Normalize host material
    if 'host_material' in normalized:
        normalized['host_material'] = normalize_text(
            normalized['host_material'],
            MATERIAL_ALIASES
        )

    # Normalize synthesis method
    if 'synthesis_conditions' in normalized and isinstance(normalized['synthesis_conditions'], dict):
        method = normalized['synthesis_conditions'].get('method')
        if method:
            normalized['synthesis_conditions']['method'] = normalize_text(
                method,
                METHOD_ALIASES
            )

        # Normalize temperature
        temp = normalized['synthesis_conditions'].get('temperature_c')
        normalized['synthesis_conditions']['temperature_c'] = normalize_temperature(temp)

    # Normalize doping site
    if 'doping_outcome' in normalized and isinstance(normalized['doping_outcome'], dict):
        if 'site_distribution' in normalized['doping_outcome'] and isinstance(normalized['doping_outcome']['site_distribution'], dict):
            site = normalized['doping_outcome']['site_distribution'].get('primary_site')
            if site:
                normalized['doping_outcome']['site_distribution']['primary_site'] = normalize_text(
                    site,
                    SITE_ALIASES
                )

    # Normalize dopant concentration
    if 'dopant' in normalized and isinstance(normalized['dopant'], dict):
        conc = normalized['dopant'].get('concentration')
        normalized['dopant']['concentration'] = normalize_concentration(conc)

    return normalized


def normalize_relation(rel: Dict) -> Dict:
    """Normalize a single causal relationship."""
    normalized = rel.copy()

    # Normalize affected property
    if 'affected_property' in normalized:
        normalized['affected_property'] = normalize_text(
            normalized['affected_property'],
            PROPERTY_ALIASES
        )

    # Clean up cause parameter (lowercase common terms)
    cause = normalized.get('cause_parameter', '')
    if isinstance(cause, str):
        # Standardize temperature references
        cause = re.sub(r'\btemperature\b', 'temperature', cause, flags=re.IGNORECASE)
        cause = re.sub(r'\bpressure\b', 'pressure', cause, flags=re.IGNORECASE)
        cause = re.sub(r'\btime\b', 'time', cause, flags=re.IGNORECASE)
        normalized['cause_parameter'] = cause.strip()

    return normalized


def filter_experiments(experiments: List[Dict]) -> List[Dict]:
    """Filter low-quality experiments."""
    filtered = []

    for exp in experiments:
        # Must have host material
        if not exp.get('host_material'):
            continue

        # Must have dopant element
        if not exp.get('dopant') or not exp['dopant'].get('element'):
            continue

        filtered.append(exp)

    return filtered


def filter_relations(relations: List[Dict]) -> Tuple[List[Dict], Counter]:
    """Filter low-quality relations and track reasons."""
    filtered = []
    rejection_reasons = Counter()

    for rel in relations:
        is_valid, reason = is_valid_relation(rel)

        if is_valid:
            filtered.append(rel)
        else:
            rejection_reasons[reason] += 1

    return filtered, rejection_reasons


def build_psp_chains(relations: List[Dict]) -> List[Dict]:
    """
    Identify complete PSP (Processing → Structure → Property) chains.

    A complete chain has:
    - cause_parameter (Processing)
    - effect_on_doping (Structure)
    - affected_property (Property)
    """
    complete_chains = []

    for rel in relations:
        if (rel.get('cause_parameter') and
            rel.get('effect_on_doping') and
            rel.get('affected_property')):
            complete_chains.append(rel)

    return complete_chains


def generate_statistics(raw_data: Dict, normalized_data: Dict, rejection_reasons: Counter) -> str:
    """Generate normalization statistics report."""
    raw_exp = raw_data.get('doping_experiments', [])
    raw_rel = raw_data.get('causal_relationships', [])

    norm_exp = normalized_data.get('doping_experiments', [])
    norm_rel = normalized_data.get('causal_relationships', [])

    complete_chains = build_psp_chains(norm_rel)

    # Count unique materials, methods, properties
    materials = Counter(exp.get('host_material') for exp in norm_exp if exp.get('host_material'))
    methods = Counter(
        exp.get('synthesis_conditions', {}).get('method')
        for exp in norm_exp
        if exp.get('synthesis_conditions', {}).get('method')
    )
    properties = Counter(rel.get('affected_property') for rel in norm_rel if rel.get('affected_property'))

    report = f"""{'='*60}
KNOWLEDGE GRAPH NORMALIZATION REPORT
{'='*60}

INPUT (Raw):
  Experiments: {len(raw_exp)}
  Causal Relationships: {len(raw_rel)}

OUTPUT (Normalized & Filtered):
  Experiments: {len(norm_exp)} ({len(norm_exp) - len(raw_exp):+d})
  Causal Relationships: {len(norm_rel)} ({len(norm_rel) - len(raw_rel):+d})
  Complete PSP Chains: {len(complete_chains)}

FILTERING:
  Experiments removed: {len(raw_exp) - len(norm_exp)}
  Relationships removed: {len(raw_rel) - len(norm_rel)}

REJECTION REASONS:
"""
    for reason, count in rejection_reasons.most_common():
        report += f"  {reason}: {count}\n"

    report += f"""
NORMALIZED ENTITIES:

Top 10 Host Materials:
"""
    for material, count in materials.most_common(10):
        report += f"  {material}: {count}\n"

    report += f"""
Top 10 Synthesis Methods:
"""
    for method, count in methods.most_common(10):
        report += f"  {method}: {count}\n"

    report += f"""
Top 10 Affected Properties:
"""
    for prop, count in properties.most_common(10):
        report += f"  {prop}: {count}\n"

    report += f"""
{'='*60}
VALIDATION METRICS:

Graph Metrics:
  Nodes (unique entities): {len(materials) + len(methods) + len(properties)}
  Edges (causal relations): {len(norm_rel)}
  Complete PSP chains: {len(complete_chains)}

Quality Metrics:
  Experiments with valid synthesis method: {sum(1 for e in norm_exp if e.get('synthesis_conditions', {}).get('method'))}
  Experiments with temperature data: {sum(1 for e in norm_exp if e.get('synthesis_conditions', {}).get('temperature_c') is not None)}
  Relations with mechanism quotes: {sum(1 for r in norm_rel if r.get('mechanism_quote'))}

Coverage:
  Unique materials: {len(materials)}
  Unique methods: {len(methods)}
  Unique properties: {len(properties)}

{'='*60}
"""

    return report


def main():
    """Main normalization pipeline."""
    print(f"\n{'='*60}")
    print("KG Normalization Pipeline")
    print(f"{'='*60}\n")

    # Load raw KG
    if not INPUT_FILE.exists():
        print(f"✗ ERROR: Input file not found: {INPUT_FILE}")
        print(f"  Run extract_robust.py first")
        return 1

    print(f"Loading raw KG from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    raw_experiments = raw_data.get('doping_experiments', [])
    raw_relations = raw_data.get('causal_relationships', [])

    print(f"  Raw experiments: {len(raw_experiments)}")
    print(f"  Raw relationships: {len(raw_relations)}\n")

    # Normalize experiments
    print("Normalizing experiments...")
    normalized_experiments = [normalize_experiment(exp) for exp in raw_experiments]
    filtered_experiments = filter_experiments(normalized_experiments)
    print(f"  After filtering: {len(filtered_experiments)} experiments\n")

    # Normalize and filter relations
    print("Normalizing and filtering causal relationships...")
    normalized_relations = [normalize_relation(rel) for rel in raw_relations]
    filtered_relations, rejection_reasons = filter_relations(normalized_relations)
    print(f"  After filtering: {len(filtered_relations)} relationships")
    print(f"  Rejected: {len(raw_relations) - len(filtered_relations)}\n")

    # Build complete PSP chains
    complete_chains = build_psp_chains(filtered_relations)
    print(f"Complete PSP chains: {len(complete_chains)}\n")

    # Create output matching kg_example.json schema
    output_data = {
        'doping_experiments': filtered_experiments,
        'causal_relationships': filtered_relations,
        'metadata': {
            'source': 'OpenAlex + Ollama extraction',
            'normalization_date': raw_data.get('metadata', {}).get('extraction_date', ''),
            'total_experiments': len(filtered_experiments),
            'total_relationships': len(filtered_relations),
            'complete_psp_chains': len(complete_chains),
            'model_used': raw_data.get('metadata', {}).get('model_used', ''),
        }
    }

    # Save normalized KG
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, indent=2, ensure_ascii=False, fp=f)

    print(f"✓ Saved normalized KG to {OUTPUT_FILE}\n")

    # Generate and save statistics
    stats_report = generate_statistics(raw_data, output_data, rejection_reasons)
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        f.write(stats_report)

    print(stats_report)
    print(f"✓ Statistics saved to {STATS_FILE}\n")

    # Validate with build_graph.py
    print("Validating with build_graph.py...")
    try:
        from build_graph import build_causal_graph

        test_output = Path("outputs/test_graph.png")
        build_causal_graph(str(OUTPUT_FILE), str(test_output))
        print(f"✓ Graph validation successful: {test_output}")
    except Exception as e:
        print(f"⚠ Warning: Graph validation failed: {e}")
        print("  (This may be normal if there are no causal relationships)")

    print(f"\n{'='*60}")
    print("NORMALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Stats: {STATS_FILE}")
    print(f"\nNext steps:")
    print(f"  1. Review: cat {OUTPUT_FILE} | python3 -m json.tool | head -50")
    print(f"  2. Check stats: cat {STATS_FILE}")
    print(f"  3. Visualize: python3 visualize_kg.py")

    return 0


if __name__ == "__main__":
    exit(main())
