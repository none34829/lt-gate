#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run energy and latency measurements on Akida hardware or virtual device.
"""

import os
import sys
import json
import logging
from pathlib import Path
import time
import h5py  # type: ignore
import numpy as np  # type: ignore
import argparse
from akida import devices, TwoNodesIPv2  # type: ignore


def h5_iterator(path: str) -> iter:
    """Iterate over test sequences in H5 file."""
    with h5py.File(path) as f:
        for k in f['test'].keys():
            yield f['test'][k][()], f['test'][k].attrs['label']


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Akida energy measurements")
    parser.add_argument("model_path", type=str, help="Path to Akida model (.fbz)")
    parser.add_argument("test_path", type=str, help="Path to test dataset (.h5)")
    parser.add_argument("output_path", type=str, help="Path to save results (.json)")
    parser.add_argument("--virtual", action="store_true",
                       help="Force virtual device even if hardware is available")
    
    args = parser.parse_args()
    
    # Load Akida model
    print(f"Loading Akida model from {args.model_path}")
    from akida import Model
    model_ak = Model(args.model_path)
    
    # Find or create device
    print("\nInitializing Akida device...")
    devs = [] if args.virtual else devices()
    device = devs[0] if devs else TwoNodesIPv2()
    print(f"Using device: {device}")
    
    # Map model to device
    print("\nMapping model to device...")
    model_ak.map(device)
    model_ak.summary()
    
    # Enable power measurement if available
    if hasattr(device, 'soc'):
        device.soc.power_measurement_enabled = True
        print("Power measurement enabled")
    else:
        print("Power measurement not available on virtual device")
    
    # Run test set
    print(f"\nProcessing test set: {args.test_path}")
    energy_mJ = 0.0
    frames = 0
    correct = 0
    total = 0
    
    t0 = time.time()
    for seq, label in h5_iterator(args.test_path):
        # Convert to uint8 and run inference
        out = model_ak.forward(seq.astype(np.uint8))
        frames += 1
        
        # Track accuracy
        pred = np.argmax(out)
        correct += (pred == label)
        total += 1
        
        # Progress update every 100 frames
        if frames % 100 == 0:
            print(f"Processed {frames} sequences...")
    
    t1 = time.time()
    
    # Get statistics
    stats = model_ak.statistics
    energy_mJ = stats.get('last_inference_energy', 0) * frames
    
    # Compute metrics
    wallclock_s = t1 - t0
    fps = frames / wallclock_s
    accuracy = 100.0 * correct / total
    
    # Prepare results
    results = {
        "energy_mJ": energy_mJ,
        "fps": stats.get('average_fps', fps),
        "wallclock_s": wallclock_s,
        "total_frames": frames,
        "accuracy": accuracy,
        "device_type": "hardware" if devs else "virtual",
        "device_info": str(device),
        "model_path": args.model_path,
        "test_path": args.test_path
    }
    
    # Save results
    print(f"\nSaving results to {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nMeasurement Summary:")
    print("-" * 40)
    print(f"Device: {results['device_type']}")
    print(f"Sequences processed: {frames}")
    print(f"Total time: {wallclock_s:.2f} s")
    print(f"Average FPS: {fps:.1f}")
    print(f"Accuracy: {accuracy:.2f}%")
    if energy_mJ > 0:
        print(f"Total energy: {energy_mJ:.3f} mJ")
        print(f"Energy per frame: {energy_mJ/frames:.3f} mJ")
    else:
        print("Energy measurement not available (virtual device)")


if __name__ == "__main__":
    main()
