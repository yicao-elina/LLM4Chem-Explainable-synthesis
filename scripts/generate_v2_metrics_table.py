#!/usr/bin/env python3
"""
Parse v2 metrics from markdown evaluation reports and generate LaTeX table.

Usage:
    python generate_v2_metrics_table.py \
        --forward-id evaluation_report_id_v2.md \
        --forward-ood evaluation_report_ood_v2.md \
        --inverse-id inverse_report_id_v2_default.md \
        --inverse-ood inverse_report_ood_v2_default.md \
        --output v2_metrics_table.tex \
        --columns base_score,literature_grounding,physics_score,overall_score
"""

import argparse
import pandas as pd
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class MetricValue:
    """Store metric value with mean and std."""
    mean: float
    std: float
    
    def __str__(self):
        return f"{self.mean:.3f}±{self.std:.3f}"
    
    def value(self):
        """Return mean value for calculations."""
        return self.mean


def parse_markdown_table(content: str) -> pd.DataFrame:
    """Parse markdown table from report content."""
    lines = content.strip().split('\n')
    
    # Find table start
    table_start = None
    for i, line in enumerate(lines):
        if '|' in line and 'variant' in line.lower():
            table_start = i
            break
    
    if table_start is None:
        raise ValueError("Could not find table in markdown")
    
    # Extract header and data rows
    header_line = lines[table_start]
    separator_line = lines[table_start + 1]
    data_lines = lines[table_start + 2:]
    
    # Parse header
    headers = [h.strip() for h in header_line.split('|')[1:-1]]
    
    # Parse data rows
    data = []
    for line in data_lines:
        if '|' not in line or line.strip().startswith('---'):
            continue
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) == len(headers):
            data.append(dict(zip(headers, cells)))
    
    return pd.DataFrame(data)


def parse_metric_value(value_str: str) -> MetricValue:
    """Parse metric value from string like '0.353 ± 0.039' or '0.353±0.039'."""
    # Handle both formats: with and without spaces around ±
    value_str = value_str.replace(' ', '')
    
    if '±' in value_str:
        parts = value_str.split('±')
        mean = float(parts[0])
        std = float(parts[1])
    else:
        # If no std, assume 0.000
        mean = float(value_str)
        std = 0.0
    
    return MetricValue(mean, std)


def load_reports(forward_id_path: str, forward_ood_path: str, 
                 inverse_id_path: str, inverse_ood_path: str) -> Dict:
    """Load all four evaluation reports."""
    reports = {}
    
    for label, path in [
        ('forward_id', forward_id_path),
        ('forward_ood', forward_ood_path),
        ('inverse_id', inverse_id_path),
        ('inverse_ood', inverse_ood_path),
    ]:
        with open(path, 'r') as f:
            content = f.read()
        
        # Find the Results Summary section
        if '## Results Summary' in content or '## Overall Performance Summary' in content:
            # Extract just the table part
            start_idx = content.find('| variant')
            end_idx = content.find('\n\n', start_idx)
            if start_idx != -1:
                table_content = content[start_idx:end_idx if end_idx != -1 else None]
                df = parse_markdown_table(table_content)
                reports[label] = df
    
    return reports


def extract_metrics(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, MetricValue]]:
    """Extract metrics from dataframe, parsing mean±std format."""
    metrics = {}
    
    for idx, row in df.iterrows():
        variant = row['variant'].strip()
        metrics[variant] = {}
        
        for col in columns:
            if col in row:
                metrics[variant][col] = parse_metric_value(row[col])
    
    return metrics


def calculate_domain_gap(id_metrics: Dict, ood_metrics: Dict) -> Tuple[float, str]:
    """Calculate domain gap as (OOD - ID) / ID * 100."""
    id_val = id_metrics.value()
    ood_val = ood_metrics.value()
    
    if id_val == 0:
        return 0.0, 'teal'
    
    gap = (ood_val - id_val) / id_val * 100
    
    if gap < -0.5:
        color = 'red'
    elif gap > 0.5:
        color = 'blue'
    else:
        color = 'teal'
    
    return gap, color


def calculate_comparison(baseline_val: float, variant_val: float) -> Tuple[float, str]:
    """Calculate comparison as (variant - baseline) / baseline * 100."""
    if baseline_val == 0:
        return 0.0, 'teal'
    
    diff = (variant_val - baseline_val) / baseline_val * 100
    
    if diff < -0.5:
        color = 'red'
    elif diff > 0.5:
        color = 'blue'
    else:
        color = 'teal'
    
    return diff, color


def generate_latex_table(reports: Dict, columns: List[str], output_path: str):
    """Generate LaTeX table from parsed reports."""
    
    # Get available columns from reports
    forward_cols = list(reports['forward_id'].columns)
    inverse_cols = list(reports['inverse_id'].columns)
    
    # Filter columns to only those that exist and are requested
    forward_columns = [c for c in columns if c in forward_cols]
    inverse_columns = [c for c in columns if c in inverse_cols]
    
    # If inverse doesn't have the same columns, use overall_score at minimum
    if not inverse_columns and 'overall_score' in inverse_cols:
        inverse_columns = ['overall_score']
    
    # Extract metrics from all reports
    forward_id = extract_metrics(reports['forward_id'], forward_columns)
    forward_ood = extract_metrics(reports['forward_ood'], forward_columns)
    inverse_id = extract_metrics(reports['inverse_id'], inverse_columns)
    inverse_ood = extract_metrics(reports['inverse_ood'], inverse_columns)
    
    # Use forward columns for the table structure (they're the main ones requested)
    columns = forward_columns
    
    # Create column headers dynamically
    col_count = len(columns) + 2  # System + Domain + columns
    col_spec = 'l' + 'c' * (col_count - 1)
    
    # Format column names
    column_headers = []
    for col in columns:
        # Convert column name to readable format
        col_name = col.replace('_', ' ').title()
        if col == 'base_score':
            col_name = 'Base Score'
        elif col == 'literature_grounding':
            col_name = 'Literature Grounding'
        elif col == 'tier_calibration':
            col_name = 'Tier Calibration'
        elif col == 'physics_score':
            col_name = 'Physics Score'
        elif col == 'overall_score':
            col_name = 'Overall Score'
        elif col == 'kg_grounding':
            col_name = 'KG Grounding'
        elif col == 'causal_coherence_score':
            col_name = 'Causal Coherence'
        elif col == 'source_grounding_score':
            col_name = 'Source Grounding'
        elif col == 'internal_validity_score':
            col_name = 'Internal Validity'
        column_headers.append(col_name)
    
    with open(output_path, 'w') as f:
        f.write("\\begin{table*}[ht!]\n")
        f.write("\\centering\n")
        f.write("\\caption{\\textbf{In-domain vs. out-of-domain performance analysis (v2 metrics).} \n")
        f.write("We evaluate six systems on in-domain data (materials/protocols covered in KG) and out-of-domain data ")
        f.write("(novel materials/protocols not in KG). ARIA demonstrates superior performance across both ")
        f.write("forward prediction and inverse design tasks using physics-guided v2 metrics.}\n")
        f.write("\\label{tab:domain_generalization_analysis_v2}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{System} & \\textbf{Domain} & ")
        f.write(" & ".join([f"\\textbf{{{h}}}" for h in column_headers]))
        f.write(" \\\\\n")
        f.write("\\midrule\n")
        
        # Store inverse columns info for inverse section header
        inverse_col_headers = []
        for col in inverse_columns:
            col_name = col.replace('_', ' ').title()
            if col == 'overall_score':
                col_name = 'Overall Score'
            inverse_col_headers.append(col_name)
        
        # Helper function to format metric value
        def format_metric(metric_val: MetricValue) -> str:
            return f"{metric_val.mean:.3f}$\\pm${metric_val.std:.3f}"
        
        # Helper function to format comparison
        def format_comp(value: float, color: str) -> str:
            if abs(value) < 0.01:
                return f"\\textit{{0.0\\%}}"
            else:
                return f"\\textcolor{{{color}}}{{\\textit{{{value:+.1f}\\%}}}}"
        
        # Forward Prediction Section
        f.write("\\multicolumn{" + str(col_count) + "}{c}{\\textbf{Forward Prediction}} \\\\\n")
        f.write("\\midrule\n")
        
        # Baseline ID
        f.write("Baseline LLM & In-Domain & ")
        f.write(" & ".join([format_metric(forward_id['BASELINE'][col]) for col in columns]))
        f.write(" \\\\\n")
        
        # Baseline OOD
        f.write("Baseline LLM & Out-of-Domain & ")
        f.write(" & ".join([format_metric(forward_ood['BASELINE'][col]) for col in columns]))
        f.write(" \\\\\n")
        
        # Domain Gap
        f.write("\\textit{Domain Gap} & & ")
        gaps = []
        for col in columns:
            gap, color = calculate_domain_gap(forward_id['BASELINE'][col], forward_ood['BASELINE'][col])
            gaps.append(format_comp(gap, color))
        f.write(" & ".join(gaps))
        f.write(" \\\\\n")
        f.write("\\cmidrule{1-" + str(col_count) + "}\n")
        
        # Process each variant
        variants_order = ['NAIVE_KG', 'ARIA_CORE', 'ARIA_FULL', 'ARIA_SEARCH', 'KG_ONLY']
        for variant in variants_order:
            if variant not in forward_id:
                continue
            
            variant_display = variant.replace('_', '-')
            
            # ID
            f.write(f"{variant_display} & In-Domain & ")
            f.write(" & ".join([format_metric(forward_id[variant][col]) for col in columns]))
            f.write(" \\\\\n")
            
            # OOD
            f.write(f"{variant_display} & Out-of-Domain & ")
            f.write(" & ".join([format_metric(forward_ood[variant][col]) for col in columns]))
            f.write(" \\\\\n")
            
            # Domain Gap
            f.write("\\textit{Domain Gap} & & ")
            gaps = []
            for col in columns:
                gap, color = calculate_domain_gap(forward_id[variant][col], forward_ood[variant][col])
                gaps.append(format_comp(gap, color))
            f.write(" & ".join(gaps))
            f.write(" \\\\\n")
            f.write("\\cmidrule{1-" + str(col_count) + "}\n")
        
        # Performance Comparison
        f.write("\\multicolumn{" + str(col_count) + "}{c}{\\textbf{Performance Comparison}} \\\\\n")
        f.write("\\midrule\n")
        
        baseline_vals = {col: forward_id['BASELINE'][col].value() for col in columns}
        
        for variant in variants_order:
            if variant not in forward_id:
                continue
            
            variant_display = variant.replace('_', '-')
            f.write(f"{variant_display} vs Baseline & & ")
            comps = []
            for col in columns:
                variant_val = forward_id[variant][col].value()
                diff, color = calculate_comparison(baseline_vals[col], variant_val)
                comps.append(format_comp(diff, color))
            f.write(" & ".join(comps))
            f.write(" \\\\\n")
        
        # Inverse Design Section
        f.write("\\midrule\n")
        inverse_col_count = len(inverse_columns) + 2
        f.write("\\multicolumn{" + str(col_count) + "}{c}{\\textbf{Inverse Design}} \\\\\n")
        f.write("\\midrule\n")
        
        # Baseline ID
        f.write("Baseline LLM & In-Domain & ")
        f.write(" & ".join([format_metric(inverse_id['BASELINE'][col]) for col in inverse_columns]))
        # Pad with empty cells if inverse has fewer columns
        padding = " & " * (len(columns) - len(inverse_columns)) if len(inverse_columns) < len(columns) else ""
        f.write(padding)
        f.write(" \\\\\n")
        
        # Baseline OOD
        f.write("Baseline LLM & Out-of-Domain & ")
        f.write(" & ".join([format_metric(inverse_ood['BASELINE'][col]) for col in inverse_columns]))
        f.write(padding)
        f.write(" \\\\\n")
        
        # Domain Gap
        f.write("\\textit{Domain Gap} & & ")
        gaps = []
        for col in inverse_columns:
            gap, color = calculate_domain_gap(inverse_id['BASELINE'][col], inverse_ood['BASELINE'][col])
            gaps.append(format_comp(gap, color))
        f.write(" & ".join(gaps))
        f.write(padding)
        f.write(" \\\\\n")
        f.write("\\cmidrule{1-" + str(col_count) + "}\n")
        
        # Process each variant
        for variant in variants_order:
            if variant not in inverse_id:
                continue
            
            variant_display = variant.replace('_', '-')
            
            # ID
            f.write(f"{variant_display} & In-Domain & ")
            f.write(" & ".join([format_metric(inverse_id[variant][col]) for col in inverse_columns]))
            f.write(padding)
            f.write(" \\\\\n")
            
            # OOD
            f.write(f"{variant_display} & Out-of-Domain & ")
            f.write(" & ".join([format_metric(inverse_ood[variant][col]) for col in inverse_columns]))
            f.write(padding)
            f.write(" \\\\\n")
            
            # Domain Gap
            f.write("\\textit{Domain Gap} & & ")
            gaps = []
            for col in inverse_columns:
                gap, color = calculate_domain_gap(inverse_id[variant][col], inverse_ood[variant][col])
                gaps.append(format_comp(gap, color))
            f.write(" & ".join(gaps))
            f.write(padding)
            f.write(" \\\\\n")
            f.write("\\cmidrule{1-" + str(col_count) + "}\n")
        
        # Performance Comparison
        f.write("\\multicolumn{" + str(col_count) + "}{c}{\\textbf{Performance Comparison}} \\\\\n")
        f.write("\\midrule\n")
        
        baseline_vals_inv = {col: inverse_id['BASELINE'][col].value() for col in inverse_columns}
        
        for variant in variants_order:
            if variant not in inverse_id:
                continue
            
            variant_display = variant.replace('_', '-')
            f.write(f"{variant_display} vs Baseline & & ")
            comps = []
            for col in inverse_columns:
                variant_val = inverse_id[variant][col].value()
                diff, color = calculate_comparison(baseline_vals_inv[col], variant_val)
                comps.append(format_comp(diff, color))
            f.write(" & ".join(comps))
            f.write(padding)
            f.write(" \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}}\n")
        f.write("\\end{table*}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX table from v2 metrics markdown reports'
    )
    parser.add_argument('--forward-id', required=True,
                       help='Path to forward prediction in-domain report')
    parser.add_argument('--forward-ood', required=True,
                       help='Path to forward prediction out-of-domain report')
    parser.add_argument('--inverse-id', required=True,
                       help='Path to inverse design in-domain report')
    parser.add_argument('--inverse-ood', required=True,
                       help='Path to inverse design out-of-domain report')
    parser.add_argument('--output', default='v2_metrics_table.tex',
                       help='Output LaTeX file path')
    parser.add_argument('--columns', 
                       default='base_score,literature_grounding,physics_score,overall_score',
                       help='Comma-separated list of columns to include (no Tier Calibration by default)')
    
    args = parser.parse_args()
    
    # Parse column list
    columns = [c.strip() for c in args.columns.split(',')]
    
    print(f"Loading reports...")
    reports = load_reports(args.forward_id, args.forward_ood, 
                          args.inverse_id, args.inverse_ood)
    
    print(f"Generating LaTeX table with columns: {columns}")
    generate_latex_table(reports, columns, args.output)
    
    print(f"✓ LaTeX table saved to: {args.output}")


if __name__ == '__main__':
    main()
