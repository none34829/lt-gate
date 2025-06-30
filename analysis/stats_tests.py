#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Statistical significance tests for LT-Gate evaluation.
Computes means ± CI95, paired t/Wilcoxon tests, and effect sizes.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import pingouin as pg
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.eval_metrics import load_energy, load_training_metrics


def ci95(series: pd.Series) -> Tuple[float, float]:
    """Calculate mean and 95% confidence interval.
    
    Args:
        series: Series of values
        
    Returns:
        tuple: (mean, confidence interval)
    """
    m = series.mean()
    se = series.sem()
    return float(m), float(1.96 * se)


def paired_test(x: np.ndarray,
               y: np.ndarray,
               label: str = 'metric') -> Dict[str, Any]:
    """
    Run comprehensive paired tests between two sets of measurements.
    
    Args:
        x: First set of measurements (from same seeds as y)
        y: Second set of measurements (from same seeds as x)
        label: Name of metric being compared
        
    Returns:
        dict: Test results including t/Wilcoxon tests and effect sizes
    """
    # Ensure equal lengths
    if len(x) != len(y):
        raise ValueError("Data sets must have equal length")
    
    # Frequentist paired t-test
    t_res = pg.ttest(x, y, paired=True)
    
    # Non-parametric Wilcoxon test
    w_res = pg.wilcoxon(x, y)
    
    # Effect sizes
    d = pg.compute_effsize(x, y, eftype='cohen', paired=True)
    cd = pg.compute_effsize(x, y, eftype='cliff')
    
    return {
        'metric': label,
        'mean_x': float(np.mean(x)),
        'mean_y': float(np.mean(y)),
        't': float(t_res['T'].iloc[0]),
        'p_t': float(t_res['p-val'].iloc[0]),
        'w': float(w_res['W-val'].iloc[0]),
        'p_w': float(w_res['p-val'].iloc[0]),
        'cohen_d': float(d),
        'cliffs_delta': float(cd)
    }


def load_metric(metric: str,
              alg: str,
              logs_dir: str = 'logs',
              num_seeds: int = 5) -> np.ndarray:
    """Load metric values across all seeds for an algorithm.
    
    Args:
        metric: One of 'acc_task2', 'forget', 'energy_J'
        alg: Algorithm name
        logs_dir: Directory containing seed-wise logs
        num_seeds: Number of random seeds (default: 5)
        
    Returns:
        array: Metric values across seeds
    """
    vals = []
    
    for seed in range(num_seeds):
        seed_dir = Path(logs_dir) / f'seed_{seed}'
        
        if metric == 'energy_J':
            # Load from JSON energy measurements
            path = seed_dir / f'{alg}_energy_akida.json'
            with open(path) as f:
                data = json.load(f)
                energy = data.get('energy_J', data.get('energy_mJ', 0) / 1000)
                vals.append(energy)
        else:
            # Load from training CSV
            csv_path = seed_dir / f'{alg}_train.csv'
            df = pd.read_csv(csv_path)
            
            if metric == 'acc_task2':
                val = df[df.phase == 'task2'].acc.iloc[-1]
            elif metric == 'forget':
                a1 = df[df.phase == 'task1'].acc.iloc[-1]
                a1_now = df[df.phase == 'task2'].acc.iloc[-1]
                val = a1 - a1_now
            else:
                raise ValueError(f'Unknown metric: {metric}')
            
            vals.append(val)
    
    return np.array(vals)


def compare_algorithms(algo1: str,
                     algo2: str,
                     metric: str,
                     logs_dir: str = 'logs',
                     num_seeds: int = 5) -> Dict[str, Any]:
    """
    Compare two algorithms on a specific metric across multiple seeds.
    
    Args:
        algo1: First algorithm name
        algo2: Second algorithm name
        metric: Metric to compare ('energy_J', 'forgetting', etc.)
        logs_dir: Directory containing log files
        num_seeds: Number of random seeds
        
    Returns:
        dict: Statistical test results
    """
    # Load metrics for both algorithms
    x = load_metric(metric, algo1, logs_dir, num_seeds)
    y = load_metric(metric, algo2, logs_dir, num_seeds)
    
    # Run paired tests
    return paired_test(x, y, f"{metric}:{algo1}-vs-{algo2}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run statistical tests on metrics")
    
    parser.add_argument("--logs-dir", type=str, default="logs",
                       help="Directory containing log files")
    parser.add_argument("--output", type=str, default="analysis/sig_tests.csv",
                       help="Output CSV path")
    parser.add_argument("--num-seeds", type=int, default=5,
                       help="Number of random seeds")
    parser.add_argument("--metrics", type=str, nargs="+",
                       default=['acc_task2', 'forget', 'energy_J'],
                       help="Metrics to compare")
    
    args = parser.parse_args()
    
    # Run all comparisons against LT-Gate
    results = []
    algos = ['ltgate', 'hlop', 'dsd']
    
    for metric in args.metrics:
        for algo in algos[1:]:  # Compare against baselines
            try:
                res = compare_algorithms(
                    'ltgate', algo, metric,
                    args.logs_dir, args.num_seeds
                )
                results.append(res)
                
                # Print summary
                print(f"\n{metric}: LT-Gate vs {algo.upper()}")
                print(f"Mean difference: {res['mean_x'] - res['mean_y']:.4f}")
                print(f"t-test p-value: {res['p_t']:.2e}")
                print(f"Wilcoxon p-value: {res['p_w']:.2e}")
                print(f"Cohen's d: {res['cohen_d']:.3f}")
                print(f"Cliff's delta: {res['cliffs_delta']:.3f}")
                
            except Exception as e:
                print(f"Error comparing LT-Gate vs {algo} on {metric}: {str(e)}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
