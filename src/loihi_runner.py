#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loihi-2 deployment and energy measurement script.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
import lava.lib.dl.slayer as slayer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_backbone
from src.h5_dataset import H5SequenceDataset


def quantize_model(model, bits=8):
    """
    Quantize model weights to 8-bit signed integers for Loihi.
    
    Args:
        model (nn.Module): Model to quantize
        bits (int): Number of bits for quantization
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Get weight range for scaling
            w_min, w_max = param.data.min(), param.data.max()
            scale = (2 ** (bits - 1) - 1) / max(abs(w_min), abs(w_max))
            
            # Quantize weights
            param.data = torch.round(param.data * scale) / scale
            
            print(f"Quantized {name}: range [{param.data.min():.3f}, {param.data.max():.3f}]")


def compile_for_loihi(model, example_input):
    """
    Compile model for Loihi-2 hardware.
    
    Args:
        model (nn.Module): Model to compile
        example_input (torch.Tensor): Example input for tracing
        
    Returns:
        NetClass: Compiled Loihi network class
    """
    # Convert to SLAYER format
    net = slayer.utils.pytorch.module_to_slayer(model)
    
    # Compile network
    net = slayer.utils.compiler.compile(net, example_input)
    
    # Print core utilization
    print("\nLoihi-2 Resource Usage:")
    print("-" * 40)
    print(f"Cores used: {net.core_utilization}")
    print(f"Memory used: {net.memory_utilization} bytes")
    
    return net


def measure_energy(net, data_loader, num_samples=1000):
    """
    Measure energy consumption and latency on Loihi-2.
    
    Args:
        net: Compiled Loihi network
        data_loader: Test data loader
        num_samples (int): Number of samples to measure
        
    Returns:
        dict: Energy and latency metrics
    """
    energy_samples = []
    latency_samples = []
    
    for i, (seq, _) in enumerate(data_loader):
        if i >= num_samples:
            break
            
        # Run on Loihi
        with slayer.utils.measure.Energy(net) as energy:
            with slayer.utils.measure.Latency(net) as latency:
                net(seq)
        
        energy_samples.append(energy.value)
        latency_samples.append(latency.value)
    
    return {
        "energy_mean": float(np.mean(energy_samples)),
        "energy_std": float(np.std(energy_samples)),
        "latency_mean": float(np.mean(latency_samples)),
        "latency_std": float(np.std(latency_samples)),
        "samples": len(energy_samples)
    }


def main():
    parser = argparse.ArgumentParser(description="Deploy and measure SNN on Loihi-2")
    
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to test dataset (H5)")
    parser.add_argument("--energy_log", type=str, default="logs/energy.json",
                        help="Path to save energy measurements")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to measure")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.energy_log), exist_ok=True)
    
    # Load model and checkpoint
    print(f"Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = build_backbone(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    # Load test dataset
    print(f"Loading test data from {args.dataset}")
    test_data = H5SequenceDataset(args.dataset)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False)
    
    # Get example input for compilation
    example_input, _ = next(iter(test_loader))
    
    # Quantize weights
    print("\nQuantizing weights to 8-bit...")
    quantize_model(model)
    
    # Compile for Loihi
    print("\nCompiling for Loihi-2...")
    net = compile_for_loihi(model, example_input)
    
    # Measure energy and latency
    print(f"\nMeasuring energy over {args.num_samples} samples...")
    metrics = measure_energy(net, test_loader, args.num_samples)
    
    # Save measurements
    print(f"\nSaving measurements to {args.energy_log}")
    with open(args.energy_log, 'w') as f:
        json.dump({
            "checkpoint": args.ckpt,
            "dataset": args.dataset,
            "metrics": metrics
        }, f, indent=2)
    
    # Print summary
    print("\nMeasurement Summary:")
    print("-" * 40)
    print(f"Energy per sample: {metrics['energy_mean']:.2f} ± {metrics['energy_std']:.2f} pJ")
    print(f"Latency per sample: {metrics['latency_mean']:.2f} ± {metrics['latency_std']:.2f} ms")


if __name__ == "__main__":
    main()
