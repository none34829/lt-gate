#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate LaTeX tables from statistical results.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.stats_tests import load_metric, ci95


def latex_row(metric: str,
             lt: np.ndarray,
             hl: np.ndarray,
             dsd: np.ndarray) -> str:
    """
    Generate LaTeX table row for a metric across algorithms.
    
    Args:
        metric: Metric name with direction (e.g., 'Accuracy↑')
        lt: LT-Gate values across seeds
        hl: HLOP values across seeds
        dsd: DSD-SNN values across seeds
        
    Returns:
        str: LaTeX table row
    """
    # Calculate mean ± CI95 for each algorithm
    m_lt, c_lt = ci95(pd.Series(lt))
    m_hl, c_hl = ci95(pd.Series(hl))
    m_ds, c_ds = ci95(pd.Series(dsd))
    
    # Format row with best value in bold
    means = [m_lt, m_hl, m_ds]
    if '↑' in metric:  # Higher is better
        best_idx = np.argmax(means)
    else:  # Lower is better
        best_idx = np.argmin(means)
    
    values = [
        f"{m_lt:.3f}±{c_lt:.3f}",
        f"{m_hl:.3f}±{c_hl:.3f}",
        f"{m_ds:.3f}±{c_ds:.3f}"
    ]
    
    # Bold the best value
    values[best_idx] = f"\\textbf{{{values[best_idx]}}}"
    
    return f"{metric} & {values[0]} & {values[1]} & {values[2]} \\\\"


def generate_latex_table(logs_dir: str = 'logs',
                        output_path: str = 'analysis/table_results.tex',
                        num_seeds: int = 5) -> None:
    """
    Generate complete LaTeX table with all metrics.
    
    Args:
        logs_dir: Directory containing seed-wise logs
        output_path: Path to save LaTeX table
        num_seeds: Number of random seeds
    """
    # Define metrics with direction indicators
    metrics = {
        'Accuracy↑': 'acc_task2',
        'Forgetting↓': 'forget',
        'Energy↓(J)': 'energy_J'
    }
    
    # Generate rows for each metric
    rows = []
    for display_name, metric_name in metrics.items():
        # Load values for all algorithms
        lt_vals = load_metric(metric_name, 'ltgate', logs_dir, num_seeds)
        hl_vals = load_metric(metric_name, 'hlop', logs_dir, num_seeds)
        dsd_vals = load_metric(metric_name, 'dsd', logs_dir, num_seeds)
        
        # Generate row
        rows.append(latex_row(display_name, lt_vals, hl_vals, dsd_vals))
    
    # Create complete table
    table = (
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Metric & LT-Gate & HLOP & DSD \\\\\n"
        "\\midrule\n" +
        "\n".join(rows) +
        "\n\\bottomrule\n"
        "\\end{tabular}"
    )
    
    # Save table
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(table)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate LaTeX results table")
    
    parser.add_argument("--logs-dir", type=str, default="logs",
                       help="Directory containing log files")
    parser.add_argument("--output", type=str, default="analysis/table_results.tex",
                       help="Output LaTeX file path")
    parser.add_argument("--num-seeds", type=int, default=5,
                       help="Number of random seeds")
    
    args = parser.parse_args()
    
    # Generate table
    print(f"Generating LaTeX table from {args.logs_dir}...")
    generate_latex_table(args.logs_dir, args.output, args.num_seeds)
    
    # Print preview
    print("\nTable preview:")
    print("-" * 80)
    with open(args.output) as f:
        print(f.read())
    print("-" * 80)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
