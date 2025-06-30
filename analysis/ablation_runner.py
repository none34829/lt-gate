#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automation script for running LT-Gate ablation experiments.
"""

import os
import sys
import yaml
import subprocess
import argparse
from pathlib import Path

def run_ablation(cfg_path: str, seed: int) -> None:
    """Run complete ablation experiment for one config and seed.
    
    Args:
        cfg_path: Path to ablation config YAML
        seed: Random seed
    """
    # Load config to get variant tag
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    tag = cfg['variant_tag']
    
    # Set up environment with fixed seed
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = str(seed)
    
    # Task 1 training
    cmd1 = [
        'python', 'src/train.py',
        cfg_path,
        '--dataset', 'data/fast/train.h5',
        '--phase', 'task1',
        '--seed', str(seed),
        '--logdir', f'logs/ablations/{tag}/seed_{seed}'
    ]
    subprocess.run(cmd1, env=env, check=True)
    
    # Task 2 training (resume from Task 1)
    cmd2 = [
        'python', 'src/train.py',
        cfg_path,
        '--resume', f'logs/ablations/{tag}/seed_{seed}/ckpt_task1.pt',
        '--dataset', 'data/slow/train.h5',
        '--phase', 'task2',
        '--seed', str(seed),
        '--logdir', f'logs/ablations/{tag}/seed_{seed}'
    ]
    subprocess.run(cmd2, env=env, check=True)
    
    # Energy measurement (Akida)
    cmd3 = [
        'python', 'src/run_akida.py',
        f'logs/ablations/{tag}/seed_{seed}/ckpt_task2.pt',
        'data/slow/test.h5',
        f'logs/ablations/{tag}/seed_{seed}/energy.json'
    ]
    subprocess.run(cmd3, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument('--variant', default='all',
                       help="Specific variant to run, or 'all' for all variants")
    args = parser.parse_args()
    
    # Get list of configs to run
    cfg_dir = Path('configs/ablations')
    if args.variant == 'all':
        cfg_files = sorted(cfg_dir.glob('*.yaml'))
    else:
        cfg_files = [cfg_dir / f'{args.variant}.yaml']
    
    # Run each config
    for cfg_path in cfg_files:
        print(f"\nRunning ablation: {cfg_path.stem}")
        
        # Load config to get seeds
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        seeds = cfg.get('seeds', [0, 1, 2])  # Default 3 seeds
        
        # Run all seeds
        for seed in seeds:
            print(f"\nSeed {seed}:")
            try:
                run_ablation(str(cfg_path), seed)
            except Exception as e:
                print(f"Error running {cfg_path.stem} seed {seed}: {str(e)}")
                continue


if __name__ == '__main__':
    main()
