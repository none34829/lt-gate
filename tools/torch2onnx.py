#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert PyTorch model to ONNX format for Akida deployment.
"""

import os
import sys
import torch
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_backbone


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to PyTorch checkpoint")
    parser.add_argument("output", type=str, help="Output ONNX file path")
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    
    # Build model
    model = build_backbone(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    # Create dummy input (single MNIST frame)
    dummy = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    
    # Export to ONNX
    print(f"Exporting to ONNX: {args.output}")
    torch.onnx.export(
        model, 
        dummy,
        args.output,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print("ONNX export complete")


if __name__ == "__main__":
    main()
