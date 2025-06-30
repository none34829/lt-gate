#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperparameter grid search script for LT-Gate training.
"""

import os
import sys
import yaml
import json
import itertools
from pathlib import Path
import subprocess
from datetime import datetime

# Grid search parameters
PARAM_GRID = {
    'eta': [0.005, 0.002],  # Learning rates
    'tau_ratios': [20, 50]  # τ₂/τ₁ ratios
}

def create_config(base_config_path, eta, tau_ratio, output_dir):
    """
    Create a new config file with specified hyperparameters.
    
    Args:
        base_config_path (str): Path to base config file
        eta (float): Learning rate
        tau_ratio (float): Ratio of slow to fast time constants
        output_dir (str): Directory to save new config
        
    Returns:
        str: Path to new config file
    """
    # Load base config
    with open(base_config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Update parameters
    cfg['eta'] = eta
    cfg['tau_slow'] = 0.005 * tau_ratio  # Base τ₁ = 0.005
    
    # Create output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = Path(output_dir) / f"ltgate_eta{eta}_ratio{tau_ratio}_{timestamp}.yaml"
    
    # Save new config
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(cfg, f)
    
    return str(out_path)


def run_training(config_path, dataset_path, phase):
    """
    Run training with a specific config.
    
    Args:
        config_path (str): Path to config file
        dataset_path (str): Path to dataset
        phase (str): Training phase (task1 or task2)
        
    Returns:
        float: Validation accuracy
    """
    cmd = [
        "python", "src/train.py",
        "--config", config_path,
        "--dataset", dataset_path,
        "--phase", phase
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extract validation accuracy from output
        for line in reversed(result.stdout.split('\n')):
            if "Validation accuracy:" in line:
                acc = float(line.split(":")[1].strip().rstrip('%'))
                return acc
        
        return 0.0
    
    except subprocess.CalledProcessError as e:
        print(f"Error running training: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return 0.0


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run hyperparameter grid search")
    
    parser.add_argument("--base-config", type=str, default="configs/ltgate.yaml",
                        help="Base configuration file")
    parser.add_argument("--output-dir", type=str, default="configs/grid_search",
                        help="Directory to save generated configs")
    parser.add_argument("--results-file", type=str, default="grid_search_results.json",
                        help="File to save grid search results")
    
    args = parser.parse_args()
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        PARAM_GRID['eta'],
        PARAM_GRID['tau_ratios']
    ))
    
    print(f"Running grid search with {len(param_combinations)} combinations")
    
    results = []
    
    for eta, tau_ratio in param_combinations:
        print(f"\nTesting η={eta}, τ₂/τ₁={tau_ratio}")
        
        # Create config for this combination
        config_path = create_config(args.base_config, eta, tau_ratio, args.output_dir)
        print(f"Created config: {config_path}")
        
        # Run Task 1 training
        print("\nTraining Task 1...")
        task1_acc = run_training(config_path, "data/fast/train.h5", "task1")
        
        # Run Task 2 training
        print("\nTraining Task 2...")
        task2_acc = run_training(config_path, "data/slow/train.h5", "task2")
        
        # Record results
        result = {
            "eta": eta,
            "tau_ratio": tau_ratio,
            "config_path": config_path,
            "task1_accuracy": task1_acc,
            "task2_accuracy": task2_acc,
            "mean_accuracy": (task1_acc + task2_acc) / 2
        }
        results.append(result)
        
        # Save results after each combination
        with open(args.results_file, 'w') as f:
            json.dump({
                "parameters": PARAM_GRID,
                "results": results
            }, f, indent=2)
        
        print(f"\nResults saved to {args.results_file}")
    
    # Find best combination
    best_result = max(results, key=lambda x: x['task2_accuracy'])
    print("\nBest hyperparameters:")
    print(f"η = {best_result['eta']}")
    print(f"τ₂/τ₁ = {best_result['tau_ratio']}")
    print(f"Task 1 accuracy: {best_result['task1_accuracy']:.2f}%")
    print(f"Task 2 accuracy: {best_result['task2_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
