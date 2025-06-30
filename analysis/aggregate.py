#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aggregate training logs and energy measurements into a master metrics CSV.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.eval_metrics import load_energy, load_training_metrics


def aggregate_metrics(algo_list: List[str],
                     logs_dir: str = 'logs',
                     energy_suffix: str = '_energy_akida.json') -> pd.DataFrame:
    """
    Aggregate metrics from training logs and energy measurements.
    
    Args:
        algo_list: List of algorithm names to process
        logs_dir: Directory containing log files
        energy_suffix: Suffix for energy measurement files
        
    Returns:
        DataFrame: Aggregated metrics for all algorithms
    """
    records = []
    logs_path = Path(logs_dir)
    
    for alg in algo_list:
        try:
            # Load training metrics
            train_csv = logs_path / f'{alg}_train.csv'
            metrics = load_training_metrics(train_csv)
            
            # Extract key metrics
            A1 = metrics['task1_final_acc']
            A1_now = metrics.get('task2_final_acc', A1)  # Same as A1 if no task2
            A2 = metrics.get('task2_final_acc', 0.0)
            
            # Calculate average accuracy and forgetting
            avg_acc = np.mean([A1_now, A2])
            forget = A1 - A1_now
            
            # Get spikes per sample if available
            spikes = metrics.get('task2_spikes_per_sample', 0.0)
            
            # Load energy measurements
            energy_path = logs_path / f'{alg}{energy_suffix}'
            energy_J, latency_s = load_energy(energy_path)
            
            # Create record
            record = {
                'algorithm': alg,
                'acc_task1': A1,
                'acc_task1_final': A1_now,
                'acc_task2': A2,
                'avg_accuracy': avg_acc,
                'forgetting': forget,
                'energy_J': energy_J,
                'latency_ms': latency_s * 1000 if latency_s else None,
                'spikes_per_sample': spikes
            }
            
            # Add gamma mean if available (LT-Gate specific)
            if 'task2_gamma_mean' in metrics:
                record['gamma_mean'] = metrics['task2_gamma_mean']
            
            records.append(record)
            
        except Exception as e:
            print(f"Error processing {alg}: {str(e)}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Sort by average accuracy (descending)
    df = df.sort_values('avg_accuracy', ascending=False)
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate metrics from all algorithms")
    
    parser.add_argument("--logs-dir", type=str, default="logs",
                       help="Directory containing log files")
    parser.add_argument("--output", type=str, default="logs/master_metrics.csv",
                       help="Output CSV path")
    parser.add_argument("--energy-suffix", type=str, default="_energy_akida.json",
                       help="Suffix for energy measurement files")
    parser.add_argument("--algorithms", nargs="+", default=['ltgate', 'hlop', 'dsd'],
                       help="List of algorithms to process")
    
    args = parser.parse_args()
    
    # Aggregate metrics
    print(f"Processing algorithms: {args.algorithms}")
    df = aggregate_metrics(
        args.algorithms,
        args.logs_dir,
        args.energy_suffix
    )
    
    # Save to CSV
    print(f"\nSaving to {args.output}")
    df.to_csv(args.output, index=False)
    
    # Print summary
    print("\nMetrics Summary:")
    print("-" * 80)
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
