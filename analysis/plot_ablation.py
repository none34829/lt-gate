#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate plots and summary tables from ablation experiment results.
"""

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def load_ablation_results() -> pd.DataFrame:
    """Load and aggregate results from all ablation variants.
    
    Returns:
        DataFrame: Summary statistics for each variant
    """
    ablation_dir = Path('logs/ablations')
    variants = sorted(os.listdir(ablation_dir))
    rows = []
    
    for var in variants:
        acc2, forget, energy = [], [], []
        
        # Load metrics from each seed
        for seed_dir in (ablation_dir / var).glob('seed_*'):
            # Load training metrics
            df = pd.read_csv(seed_dir / 'train.csv')
            
            # Task 2 accuracy
            acc2.append(df[df.phase == 'task2'].acc.iloc[-1])
            
            # Forgetting (Task 1 final - current Task 1)
            t1_final = df[df.phase == 'task1'].acc.iloc[-1]
            t1_now = df[df.phase == 'task2'].acc.iloc[-1]
            forget.append(t1_final - t1_now)
            
            # Energy (convert mJ to J)
            with open(seed_dir / 'energy.json') as f:
                energy.append(json.load(f)['energy_mJ'] / 1000)
        
        # Compute statistics
        rows.append({
            'variant': var,
            'acc2': pd.Series(acc2).mean(),
            'acc2_ci': 1.96 * pd.Series(acc2).sem(),
            'forget': pd.Series(forget).mean(),
            'forget_ci': 1.96 * pd.Series(forget).sem(),
            'energy_J': pd.Series(energy).mean(),
            'energy_ci': 1.96 * pd.Series(energy).sem()
        })
    
    return pd.DataFrame(rows).sort_values('forget')


def plot_metric(df: pd.DataFrame,
               metric: str,
               output_path: Optional[str] = None,
               title: Optional[str] = None) -> None:
    """Create bar plot with error bars for a metric.
    
    Args:
        df: DataFrame with ablation results
        metric: Column name to plot
        output_path: Optional path to save figure
        title: Optional plot title
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot bars with error bars
    sns.barplot(
        x='variant',
        y=metric,
        data=df,
        yerr=df[f'{metric}_ci'],
        ax=ax
    )
    
    # Customize appearance
    if title:
        ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel(metric.capitalize())
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()


def generate_latex_table(df: pd.DataFrame,
                        output_path: str = 'analysis/ablation_table.tex') -> None:
    """Generate LaTeX table with ablation results.
    
    Args:
        df: DataFrame with ablation results
        output_path: Path to save LaTeX table
    """
    # Format table rows
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{row['variant']} & "
            f"{row['acc2']:.1f} ± {row['acc2_ci']:.1f} & "
            f"{row['forget']:.1f} ± {row['forget_ci']:.1f} & "
            f"{row['energy_J']*1000:.1f} ± {row['energy_ci']*1000:.1f} \\\\"
        )
    
    # Create complete table
    table = (
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Variant & Task-2 Acc. & Forgetting & Energy (mJ) \\\\\n"
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
    # Load and save summary CSV
    print("Loading ablation results...")
    df = load_ablation_results()
    df.to_csv('analysis/ablations_summary.csv', index=False)
    print(f"Saved summary to analysis/ablations_summary.csv")
    
    # Generate plots
    print("\nGenerating plots...")
    metrics = ['forget', 'acc2', 'energy_J']
    titles = {
        'forget': 'Forgetting (%)',
        'acc2': 'Task-2 Accuracy (%)',
        'energy_J': 'Energy (J)'
    }
    
    for metric in metrics:
        plot_metric(
            df,
            metric,
            output_path=f'analysis/ablation_{metric}.pdf',
            title=titles[metric]
        )
        print(f"Saved {metric} plot to analysis/ablation_{metric}.pdf")
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    generate_latex_table(df)
    print("Saved table to analysis/ablation_table.tex")
    
    # Print key findings
    print("\nKey findings:")
    print("-" * 80)
    print(f"Best forgetting: {df.forget.min():.1f}% ({df.iloc[0].variant})")
    print(f"Worst forgetting: {df.forget.max():.1f}% ({df.iloc[-1].variant})")
    print(f"Energy range: {df.energy_J.min()*1000:.1f}-{df.energy_J.max()*1000:.1f} mJ")
    

if __name__ == '__main__':
    main()
