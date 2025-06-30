#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare calibration dataset for Akida quantization.
Extracts random sequences from training data and converts to uint8.
"""

import os
import sys
import h5py
import numpy as np
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Prepare calibration data for Akida")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to training dataset (H5)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for calibration data (NPZ)")
    parser.add_argument("--samples", type=int, default=1024,
                       help="Number of sequences to extract")
    parser.add_argument("--seed", type=int, default=1234,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print(f"Loading dataset: {args.dataset}")
    with h5py.File(args.dataset, 'r') as f:
        # Get training group
        train_group = f['train']
        
        # Get total number of sequences
        total_sequences = len(train_group.keys())
        print(f"Found {total_sequences} training sequences")
        
        # Randomly select sequence indices
        indices = np.random.choice(total_sequences, args.samples, replace=False)
        
        # Extract sequences
        print(f"\nExtracting {args.samples} random sequences...")
        sequences = []
        for idx in tqdm(indices):
            # Load sequence
            seq = train_group[str(idx)][()]
            
            # Convert to uint8 (0-255 range)
            if seq.dtype != np.uint8:
                seq = (seq * 255).astype(np.uint8)
            
            sequences.append(seq)
    
    # Stack sequences
    sequences = np.stack(sequences)
    
    # Save calibration data
    print(f"\nSaving calibration data: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        sequences=sequences,
        indices=indices,
        metadata={
            'source': args.dataset,
            'samples': args.samples,
            'seed': args.seed
        }
    )
    
    print(f"Saved {len(sequences)} sequences of shape {sequences.shape}")


if __name__ == "__main__":
    main()
