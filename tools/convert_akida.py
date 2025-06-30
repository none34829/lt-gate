#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert ONNX model to Akida format with quantization.
"""

import os
import sys
import numpy as np
import h5py
import argparse
import tensorflow as tf
import onnx2tf
from quantizeml import quantize
from quantizeml.models import QuantizationParams
from cnn2snn import convert


def load_calibration_data(calib_path):
    """Load calibration data from NPZ file."""
    print(f"Loading calibration data from {calib_path}")
    with np.load(calib_path) as data:
        return data['sequences']


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to Akida format")
    parser.add_argument("onnx_path", type=str, help="Input ONNX model path")
    parser.add_argument("output_path", type=str, help="Output Akida model path (.fbz)")
    parser.add_argument("calib_path", type=str, help="Calibration data path (.npz)")
    parser.add_argument("--bits", type=int, default=8,
                       help="Quantization bits for weights/activations")
    
    args = parser.parse_args()
    
    # Load ONNX and convert to Keras
    print(f"Converting ONNX model to Keras: {args.onnx_path}")
    model_fp = onnx2tf.convert(args.onnx_path)
    
    # Load calibration data
    calib_data = load_calibration_data(args.calib_path)
    
    # Set up quantization parameters
    print(f"\nQuantizing model to {args.bits}-bit precision")
    qparams = QuantizationParams(
        input_weight_bits=args.bits,
        weight_bits=args.bits,
        activation_bits=args.bits
    )
    
    # Quantize the model
    model_q = quantize(
        model_fp,
        qparams=qparams,
        samples=calib_data
    )
    
    # Save intermediate quantized model
    h5_path = args.output_path.replace('.fbz', '_q8.h5')
    print(f"\nSaving quantized model: {h5_path}")
    model_q.save(h5_path)
    
    # Convert to Akida format
    print("\nConverting to Akida format...")
    model_ak = convert(h5_path)
    
    # Save Akida model
    print(f"Saving Akida model: {args.output_path}")
    model_ak.save(args.output_path)
    
    # Print model summary
    print("\nAkida Model Summary:")
    print("-" * 40)
    model_ak.summary()


if __name__ == "__main__":
    main()
