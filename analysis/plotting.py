#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plotting utilities for LT-Gate evaluation metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict


def set_style():
    """Set consistent style for all plots."""
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette('deep')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })


def barplot_metrics(df: pd.DataFrame,
                   metrics: Optional[List[str]] = None,
                   output_path: str = 'analysis/metric_bars.pdf'):
    """
    Create bar plots for multiple metrics.
    
    Args:
        df: DataFrame containing metrics
        metrics: List of metrics to plot (default: acc_task2, forgetting, energy_J)
        output_path: Path to save the figure
    """
    if metrics is None:
        metrics = ['acc_task2', 'forgetting', 'energy_J']
    
    # Set style
    set_style()
    
    # Create subplots
    fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot titles
    titles = {
        'acc_task2': 'Task 2 Accuracy',
        'forgetting': 'Forgetting',
        'energy_J': 'Energy (J)',
        'latency_ms': 'Latency (ms)',
        'spikes_per_sample': 'Spikes/Sample'
    }
    
    # Create bar plots
    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=df,
            x='algorithm',
            y=metric,
            ax=ax,
            capsize=0.1,
            errwidth=1.5
        )
        ax.set_title(titles.get(metric, metric))
        ax.set_xlabel('')
        
        # Rotate x-labels if needed
        if len(df['algorithm'].unique()) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(df[metric]):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def violin(df: pd.DataFrame,
         metric: str,
         output_path: Optional[str] = None) -> None:
    """Create violin plot with confidence interval strip.
    
    Args:
        df: DataFrame containing algorithm-wise metrics
        metric: Column name to plot
        output_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(4, 2.5))
    
    # Create violin plot without inner points
    sns.violinplot(
        x='algorithm',
        y=metric,
        data=df,
        inner=None,
        ax=ax,
        cut=0
    )
    
    # Add point plot with 95% CI
    sns.pointplot(
        x='algorithm',
        y=metric,
        data=df,
        ci=95,
        ax=ax,
        color='k',
        errwidth=1.5,
        join=False
    )
    
    # Customize plot
    ax.set_xlabel('')
    ax.set_ylabel(metric)
    sns.despine()
    fig.tight_layout()
    
    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()


def plot_training_curves(csv_paths: Dict[str, str],
                        output_path: str = 'analysis/training_curves.pdf'):
    """
    Plot training curves for multiple algorithms.
    
    Args:
        csv_paths: Dictionary mapping algorithm names to their training CSV paths
        output_path: Path to save the figure
    """
    set_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    for algo, path in csv_paths.items():
        df = pd.read_csv(path)
        
        # Plot accuracy
        sns.lineplot(
            data=df,
            x='epoch',
            y='acc',
            hue='phase',
            label=algo,
            ax=ax1,
            alpha=0.7
        )
        
        # Plot spikes if available
        if 'spikes' in df.columns:
            sns.lineplot(
                data=df,
                x='epoch',
                y='spikes',
                hue='phase',
                label=algo,
                ax=ax2,
                alpha=0.7
            )
    
    # Customize accuracy plot
    ax1.set_title('Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    
    # Customize spikes plot
    ax2.set_title('Spikes per Sample')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Spikes')
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_ablation_curves(df: pd.DataFrame,
                        x: str,
                        metrics: List[str],
                        output_path: str = 'analysis/ablation.pdf'):
    """
    Plot ablation study curves.
    
    Args:
        df: DataFrame containing ablation results
        x: Column name for x-axis (e.g., 'tau_ratio')
        metrics: List of metrics to plot
        output_path: Path to save the figure
    """
    set_style()
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        # Plot curve
        sns.lineplot(
            data=df,
            x=x,
            y=metric,
            marker='o',
            ax=ax
        )
        
        # Add error bars if standard deviation columns exist
        std_col = f'{metric}_std'
        if std_col in df.columns:
            ax.fill_between(
                df[x],
                df[metric] - df[std_col],
                df[metric] + df[std_col],
                alpha=0.2
            )
        
        # Customize plot
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    
    parser.add_argument("--metrics-csv", type=str, default="logs/master_metrics.csv",
                       help="Path to master metrics CSV")
    parser.add_argument("--output-dir", type=str, default="analysis",
                       help="Directory to save plots")
    parser.add_argument("--violin", action="store_true",
                       help="Generate violin plots instead of bar plots")
    
    args = parser.parse_args()
    
    # Load master metrics
    df = pd.read_csv(args.metrics_csv)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate metric plots
    metrics = ['acc_task2', 'forget', 'energy_J']
    
    if args.violin:
        print("Generating violin plots...")
        for metric in metrics:
            violin(
                df,
                metric,
                output_path=output_dir / f'violin_{metric}.pdf'
            )
    else:
        print("Generating bar plots...")
        barplot_metrics(
            df,
            metrics=metrics,
            output_path=output_dir / 'metric_bars.pdf'
        )
    
    # Generate training curves
    print("Generating training curves...")
    csv_paths = {
        algo: f"logs/{algo}_train.csv"
        for algo in df['algorithm'].unique()
    }
    plot_training_curves(csv_paths, output_path=output_dir / 'training_curves.pdf')
    
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
