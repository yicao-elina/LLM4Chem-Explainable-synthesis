"""
Generate publication-ready outputs from the knowledge graph:
- CSV tables of experiments and causal relationships
- LaTeX tables for paper inclusion
- Summary statistics
- High-resolution visualizations

Input: outputs/kg_from_abstracts.json (or any KG JSON file)
Output: Multiple formats in outputs/publication/
"""

import json
import csv
from pathlib import Path
from typing import List, Dict
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# ============= Configuration =============
INPUT_FILE = Path("outputs/kg_from_abstracts.json")
OUTPUT_DIR = Path("outputs/publication")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_kg(file_path: Path) -> Dict:
    """Load knowledge graph from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_experiments_csv(experiments: List[Dict], output_path: Path):
    """Generate CSV table of all experiments."""
    if not experiments:
        print("No experiments to export")
        return

    # Define columns
    fieldnames = [
        'experiment_id',
        'host_material',
        'dopant_element',
        'dopant_concentration',
        'dopant_precursor',
        'synthesis_method',
        'temperature_c',
        'time_hours',
        'atmosphere',
        'primary_site',
        'site_specificity',
        'structural_changes',
        'carrier_type',
        'carrier_concentration',
        'band_gap_ev',
        'characterization',
        'source_file'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for exp in experiments:
            # Flatten nested structure
            row = {
                'experiment_id': exp.get('experiment_id', ''),
                'host_material': exp.get('host_material', ''),
                'dopant_element': exp.get('dopant', {}).get('element', ''),
                'dopant_concentration': exp.get('dopant', {}).get('concentration', ''),
                'dopant_precursor': exp.get('dopant', {}).get('precursor', ''),
                'synthesis_method': exp.get('synthesis_conditions', {}).get('method', ''),
                'temperature_c': exp.get('synthesis_conditions', {}).get('temperature_c', ''),
                'time_hours': exp.get('synthesis_conditions', {}).get('time_hours', ''),
                'atmosphere': exp.get('synthesis_conditions', {}).get('atmosphere', ''),
                'primary_site': exp.get('doping_outcome', {}).get('site_distribution', {}).get('primary_site', ''),
                'site_specificity': exp.get('doping_outcome', {}).get('site_distribution', {}).get('site_specificity', ''),
                'structural_changes': str(exp.get('doping_outcome', {}).get('structural_changes', '')),
                'carrier_type': exp.get('property_changes', {}).get('electronic', {}).get('carrier_type', ''),
                'carrier_concentration': str(exp.get('property_changes', {}).get('electronic', {}).get('carrier_concentration', '')),
                'band_gap_ev': exp.get('property_changes', {}).get('electronic', {}).get('band_gap_ev', ''),
                'characterization': ', '.join(exp.get('characterization_evidence', [])),
                'source_file': exp.get('source_file', '')
            }
            writer.writerow(row)

    print(f"✓ Exported {len(experiments)} experiments to {output_path}")


def generate_relationships_csv(relationships: List[Dict], output_path: Path):
    """Generate CSV table of causal relationships."""
    if not relationships:
        print("No causal relationships to export")
        return

    fieldnames = [
        'cause_parameter',
        'effect_on_doping',
        'affected_property',
        'mechanism_quote',
        'source_file'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rel in relationships:
            writer.writerow({
                'cause_parameter': rel.get('cause_parameter', ''),
                'effect_on_doping': rel.get('effect_on_doping', ''),
                'affected_property': rel.get('affected_property', ''),
                'mechanism_quote': rel.get('mechanism_quote', ''),
                'source_file': rel.get('source_file', '')
            })

    print(f"✓ Exported {len(relationships)} causal relationships to {output_path}")


def generate_latex_table_experiments(experiments: List[Dict], output_path: Path, max_rows: int = 20):
    """Generate LaTeX table of selected experiments (top N most cited)."""
    if not experiments:
        print("No experiments to export")
        return

    # Sample experiments (take first N)
    sample = experiments[:max_rows]

    latex = r"""\begin{table}[h]
\centering
\caption{Sample Doping Experiments Extracted from Literature}
\label{tab:experiments}
\small
\begin{tabular}{llllll}
\toprule
\textbf{Host} & \textbf{Dopant} & \textbf{Method} & \textbf{Site} & \textbf{Property} & \textbf{Characterization} \\
\midrule
"""

    for exp in sample:
        host = exp.get('host_material', 'N/A')[:15]  # Truncate long names
        dopant = exp.get('dopant', {}).get('element', 'N/A')
        method = exp.get('synthesis_conditions', {}).get('method', 'N/A')[:20]
        site = exp.get('doping_outcome', {}).get('site_distribution', {}).get('primary_site', 'N/A')
        carrier = exp.get('property_changes', {}).get('electronic', {}).get('carrier_type', 'N/A')
        char = ', '.join(exp.get('characterization_evidence', [])[:2])[:25]

        # Escape LaTeX special characters
        for char_to_escape in ['_', '%', '&', '#']:
            host = host.replace(char_to_escape, '\\' + char_to_escape)
            method = method.replace(char_to_escape, '\\' + char_to_escape)
            char = char.replace(char_to_escape, '\\' + char_to_escape)

        latex += f"{host} & {dopant} & {method} & {site} & {carrier} & {char} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    output_path.write_text(latex, encoding='utf-8')
    print(f"✓ Generated LaTeX table with {len(sample)} experiments: {output_path}")


def generate_latex_table_relationships(relationships: List[Dict], output_path: Path, max_rows: int = 15):
    """Generate LaTeX table of causal relationships."""
    if not relationships:
        print("No causal relationships to export")
        return

    # Sample relationships
    sample = relationships[:max_rows]

    latex = r"""\begin{table}[h]
\centering
\caption{Causal Relationships: Processing $\rightarrow$ Structure $\rightarrow$ Property}
\label{tab:causal_relations}
\small
\begin{tabular}{lllp{4cm}}
\toprule
\textbf{Cause} & \textbf{Effect on Doping} & \textbf{Property} & \textbf{Mechanism} \\
\midrule
"""

    for rel in sample:
        cause = rel.get('cause_parameter', 'N/A')[:20]
        effect = rel.get('effect_on_doping', 'N/A')[:25]
        prop = rel.get('affected_property', 'N/A')[:20]
        mech = rel.get('mechanism_quote', 'N/A')[:60]

        # Escape LaTeX special characters
        for char_to_escape in ['_', '%', '&', '#', '$']:
            cause = cause.replace(char_to_escape, '\\' + char_to_escape)
            effect = effect.replace(char_to_escape, '\\' + char_to_escape)
            prop = prop.replace(char_to_escape, '\\' + char_to_escape)
            mech = mech.replace(char_to_escape, '\\' + char_to_escape)

        latex += f"{cause} & {effect} & {prop} & {mech} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    output_path.write_text(latex, encoding='utf-8')
    print(f"✓ Generated LaTeX table with {len(sample)} relationships: {output_path}")


def generate_statistics(kg_data: Dict, output_path: Path):
    """Generate comprehensive statistics about the KG."""
    experiments = kg_data.get('doping_experiments', [])
    relationships = kg_data.get('causal_relationships', [])

    # Count host materials
    host_materials = Counter(exp.get('host_material', 'Unknown') for exp in experiments)

    # Count dopants
    dopants = Counter(exp.get('dopant', {}).get('element', 'Unknown') for exp in experiments)

    # Count synthesis methods
    methods = Counter(exp.get('synthesis_conditions', {}).get('method', 'Unknown') for exp in experiments)

    # Count doping sites
    sites = Counter(exp.get('doping_outcome', {}).get('site_distribution', {}).get('primary_site', 'Unknown') for exp in experiments)

    # Count characterization techniques
    all_char = []
    for exp in experiments:
        all_char.extend(exp.get('characterization_evidence', []))
    characterization = Counter(all_char)

    # Count causal parameters
    cause_params = Counter(rel.get('cause_parameter', 'Unknown') for rel in relationships)

    # Write statistics
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("KNOWLEDGE GRAPH STATISTICS\n")
        f.write("="*60 + "\n\n")

        f.write(f"Total Experiments: {len(experiments)}\n")
        f.write(f"Total Causal Relationships: {len(relationships)}\n")
        f.write(f"Unique Source Files: {len(set(exp.get('source_file') for exp in experiments))}\n\n")

        f.write("-"*60 + "\n")
        f.write("TOP 10 HOST MATERIALS\n")
        f.write("-"*60 + "\n")
        for material, count in host_materials.most_common(10):
            f.write(f"{material:30} {count:5}\n")

        f.write("\n" + "-"*60 + "\n")
        f.write("TOP 10 DOPANTS\n")
        f.write("-"*60 + "\n")
        for dopant, count in dopants.most_common(10):
            f.write(f"{dopant:30} {count:5}\n")

        f.write("\n" + "-"*60 + "\n")
        f.write("TOP 10 SYNTHESIS METHODS\n")
        f.write("-"*60 + "\n")
        for method, count in methods.most_common(10):
            f.write(f"{method:40} {count:5}\n")

        f.write("\n" + "-"*60 + "\n")
        f.write("DOPING SITE DISTRIBUTION\n")
        f.write("-"*60 + "\n")
        for site, count in sites.most_common():
            f.write(f"{site:30} {count:5}\n")

        f.write("\n" + "-"*60 + "\n")
        f.write("TOP 10 CHARACTERIZATION TECHNIQUES\n")
        f.write("-"*60 + "\n")
        for tech, count in characterization.most_common(10):
            f.write(f"{tech:30} {count:5}\n")

        f.write("\n" + "-"*60 + "\n")
        f.write("TOP 10 CAUSAL PARAMETERS\n")
        f.write("-"*60 + "\n")
        for param, count in cause_params.most_common(10):
            f.write(f"{param:40} {count:5}\n")

    print(f"✓ Generated statistics: {output_path}")

    return {
        'host_materials': host_materials,
        'dopants': dopants,
        'methods': methods,
        'sites': sites,
        'characterization': characterization
    }


def generate_visualization_charts(stats: Dict, output_dir: Path):
    """Generate publication-quality charts."""

    # Chart 1: Top host materials
    fig, ax = plt.subplots(figsize=(10, 6))
    materials = stats['host_materials'].most_common(10)
    names = [m[0][:20] for m in materials]
    counts = [m[1] for m in materials]

    ax.barh(names, counts, color='#1f78b4')
    ax.set_xlabel('Number of Experiments', fontsize=12)
    ax.set_title('Top 10 Host Materials in KG', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'host_materials.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'host_materials.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated chart: host_materials.pdf/png")

    # Chart 2: Synthesis methods pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    methods = stats['methods'].most_common(8)
    labels = [m[0][:30] for m in methods]
    sizes = [m[1] for m in methods]

    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title('Synthesis Methods Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'synthesis_methods.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'synthesis_methods.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated chart: synthesis_methods.pdf/png")

    # Chart 3: Doping site distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    sites = stats['sites'].most_common()
    names = [s[0] for s in sites if s[0] != 'Unknown']
    counts = [s[1] for s in sites if s[0] != 'Unknown']

    ax.bar(names, counts, color='#2ca02c')
    ax.set_ylabel('Number of Experiments', fontsize=12)
    ax.set_xlabel('Doping Site', fontsize=12)
    ax.set_title('Doping Site Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'doping_sites.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'doping_sites.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated chart: doping_sites.pdf/png")


def main():
    """Main workflow to generate all publication outputs."""
    print(f"\n{'='*60}")
    print("Generate Publication Outputs from Knowledge Graph")
    print(f"{'='*60}\n")

    if not INPUT_FILE.exists():
        print(f"✗ ERROR: Input file not found: {INPUT_FILE}")
        print(f"  Run extract_from_abstracts.py first")
        return

    # Load KG
    print(f"Loading KG from {INPUT_FILE}...")
    kg_data = load_kg(INPUT_FILE)
    experiments = kg_data.get('doping_experiments', [])
    relationships = kg_data.get('causal_relationships', [])
    print(f"  Loaded {len(experiments)} experiments, {len(relationships)} relationships\n")

    # Generate CSV tables
    print("Generating CSV tables...")
    generate_experiments_csv(experiments, OUTPUT_DIR / 'experiments.csv')
    generate_relationships_csv(relationships, OUTPUT_DIR / 'causal_relationships.csv')

    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    generate_latex_table_experiments(experiments, OUTPUT_DIR / 'table_experiments.tex')
    generate_latex_table_relationships(relationships, OUTPUT_DIR / 'table_relationships.tex')

    # Generate statistics
    print("\nGenerating statistics...")
    stats = generate_statistics(kg_data, OUTPUT_DIR / 'statistics.txt')

    # Generate charts
    print("\nGenerating visualization charts...")
    generate_visualization_charts(stats, OUTPUT_DIR)

    # Summary
    print(f"\n{'='*60}")
    print("All Outputs Generated!")
    print(f"{'='*60}")
    print(f"Location: {OUTPUT_DIR}/")
    print(f"\nFiles created:")
    print(f"  - experiments.csv")
    print(f"  - causal_relationships.csv")
    print(f"  - table_experiments.tex")
    print(f"  - table_relationships.tex")
    print(f"  - statistics.txt")
    print(f"  - host_materials.pdf/png")
    print(f"  - synthesis_methods.pdf/png")
    print(f"  - doping_sites.pdf/png")


if __name__ == "__main__":
    main()
