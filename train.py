#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main training script for LT-Gate and other algorithms.

This script serves as the entry point for training spiking neural networks
with different learning algorithms, including LT-Gate, HLOP, and DSD-SNN.
"""

import os
import sys
import argparse
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import h5py
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

# Hardware-specific PyTorch settings for RTX 4090
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # deterministic matmuls
torch.backends.cuda.matmul.allow_tf32 = False      # bit-exact reproducibility
torch.set_grad_enabled(False)  # All algorithms are gradient-free (Hebbian)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# Global counters for spike tracking
global_spike_counter = 0
spike_counts = defaultdict(int)  # Per-layer spike counts

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import build_backbone, SNNBackbone
from src.data_loader import load_dataset, DatasetSplit
from src.algorithms.ltgate import LTGateTrainer
from src.algorithms.hlop import HLOPTrainer


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train SNN models with different algorithms")
    
    # Configuration file
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,  # Will be set based on algorithm
        help="Path to configuration file")
    
    # Algorithm selection
    parser.add_argument(
        "--alg", 
        type=str, 
        choices=["ltgate", "hlop", "dsdsnn", "baseline"], 
        default="ltgate",
        help="Algorithm to use for training")
    
    # Dataset selection
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["fast", "slow", "both"], 
        default="fast",
        help="Dataset to use for training (fast or slow MNIST)")
    
    # Training parameters
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Number of epochs to train (overrides config file)")
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="Batch size for training (overrides config file)")
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed (overrides config file)")
    
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode (fewer samples)")
    
    parser.add_argument(
        "--gpu", 
        action="store_true", 
        help="Use GPU acceleration if available")
    
    # Edge-case diagnostics parameters
    parser.add_argument(
        "--diagnostics", 
        action="store_true", 
        help="Enable edge-case diagnostics")
    
    parser.add_argument(
        "--diag-interval", 
        type=int, 
        default=100, 
        help="Interval (in steps) between diagnostic checks")
    
    parser.add_argument(
        "--weight-clip", 
        action="store_true", 
        help="Enable weight clipping to prevent explosion")
    
    parser.add_argument(
        "--max-weight", 
        type=float, 
        default=5.0, 
        help="Maximum allowed weight value")
    
    parser.add_argument(
        "--min-weight", 
        type=float, 
        default=-5.0, 
        help="Minimum allowed weight value")
    
    parser.add_argument(
        "--gamma-threshold", 
        type=float, 
        default=0.95, 
        help="Threshold to consider gating variable stuck")
    
    parser.add_argument(
        "--loihi-check", 
        action="store_true", 
        help="Check Loihi hardware mapping constraints")
    
    return parser.parse_args()


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config: {cfg}")
    return cfg


def update_config_with_args(cfg, args):
    """
    Override configuration with command line arguments.
    
    Args:
        cfg (dict): Configuration dictionary from YAML file
        args (Namespace): Command line arguments
        
    Returns:
        dict: Updated configuration dictionary
    """
    print("Config before update:", cfg)
    # Override algorithm
    if args.alg:
        cfg['alg'] = args.alg
        
    # Override training parameters
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
        
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
        
    if args.seed is not None:
        cfg['seed'] = args.seed
        
    if args.debug:
        cfg['debug'] = True
    
    # Add edge-case diagnostic parameters
    if args.diagnostics:
        cfg['enable_diagnostics'] = True
        
    if args.diag_interval:
        cfg['tracking_interval'] = args.diag_interval
        
    if args.weight_clip:
        cfg['enable_weight_clipping'] = True
        
    cfg['max_weight'] = args.max_weight
    cfg['min_weight'] = args.min_weight
    cfg['gamma_stuck_threshold'] = args.gamma_threshold
    
    if args.loihi_check:
        cfg['loihi_constraints'] = True
        
    # Update GPU flag
    if args.gpu:
        cfg['gpu'] = True
        
    return cfg


def setup_training(cfg):
    """
    Set up training environment.
    
    Args:
        cfg (dict): Configuration dictionary
        
    Returns:
        tuple: (device, seed)
    """
    # Set random seed for reproducibility
    seed = cfg.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Device configuration
    use_gpu = torch.cuda.is_available() and cfg.get('gpu', False)
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"Using device: {device} ({'GPU' if use_gpu else 'CPU'})")
    
    if use_gpu:
        # Get GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # in GB
        print(f"GPU: {gpu_name} with {gpu_memory:.1f} GB memory")
        
        # Set default CUDA device
        torch.cuda.set_device(0)
    
    return device, seed


def train_epoch(model, trainer, data_loader, device, debug=False):
    """
    Train for one epoch.
    
    Args:
        model (nn.Module): Neural network model
        trainer: Algorithm-specific trainer (LTGate, HLOP, etc)
        data_loader (DataLoader): Training data loader
        device (torch.device): Device to train on
        debug (bool): If True, will enable verbose logging
        
    Returns:
        dict: Training metrics for the epoch
    """
    model.train()
    total_correct = 0
    total_samples = 0
    total_spikes = 0.0
    total_loss = 0.0
    
    # Progress bar
    pbar = tqdm(data_loader, desc="Training")
    
    try:
        for batch_idx, (sequences, targets) in enumerate(pbar):
            batch_start = time.time()
            
            if debug and batch_idx == 0:
                print(f"\nInput shapes - sequences: {sequences.shape}, targets: {targets.shape}")
            
            # Move data to device
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Reset model state
            model.reset_state()
            
            # Process with mixed precision
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                if isinstance(trainer, LTGateTrainer):
                    # Handle sequence format
                    if sequences.size(0) != 200:
                        sequences = sequences.transpose(0, 1)
                        if debug and batch_idx == 0:
                            print(f"Transposed sequences shape: {sequences.shape}")
                    
                    # LT-Gate forward pass and updates
                    output = trainer.step(sequences)
                else:
                    # Other algorithms
                    time_steps = sequences.shape[0] if sequences.shape[0] == 200 else sequences.shape[1]
                    output = None
                    
                    for t in range(time_steps):
                        x_t = sequences[t] if sequences.shape[0] == 200 else sequences[:, t]
                        timestep_output = model(x_t)
                        
                        # Track spikes for this timestep
                        if hasattr(model, 'spike_counts'):
                            for layer_name, count in model.spike_counts.items():
                                total_spikes += count
                        
                        # Accumulate output
                        output = timestep_output if output is None else output + timestep_output
            
            # Compute accuracy
            predictions = torch.argmax(output, dim=1)
            correct = (predictions == targets).sum().item()
            total_correct += correct
            total_samples += targets.size(0)
            
            # Update progress bar
            acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
            spikes_per_sample = total_spikes / total_samples if total_samples > 0 else 0.0
            avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
            pbar.set_postfix(acc=f"{acc:.2f}%", loss=f"{avg_loss:.4f}", spikes_per_sample=f"{spikes_per_sample:.1f}")

        pbar.close()
        
        # Return metrics
        return {
            "accuracy": acc,
            "spikes_per_sample": spikes_per_sample,
            "loss": avg_loss,
        }        
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "accuracy": 0.0,
            "error": str(e),
            "spikes_per_sample": 0.0
        }
    
    # Compute epoch metrics
    avg_batch_time = sum(batch_times) / max(len(batch_times), 1) if batch_times else 0
    
    metrics = {
        "accuracy": accuracy,
        "loss": avg_loss,
        "batch_time": avg_batch_time,
        "total_time": sum(batch_times) if batch_times else 0
    }
    
    return metrics


def evaluate(model, data_loader, device, debug=False, max_batches=None):
    """
    Evaluate the model on a dataset.
    
    Args:
        model (nn.Module): Neural network model
        data_loader: DataLoader providing evaluation data
        device: Computation device (CPU or GPU)
        debug (bool): If True, will enable verbose logging
        max_batches (int, optional): Maximum number of batches to process (for debugging)
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    batch_times = []
    
    # Limit batches in debug mode
    if max_batches is None and debug:
        max_batches = 5  # Process only 5 batches in debug mode
    
    # Use tqdm for progress bar
    progress_bar = tqdm(data_loader, desc="Evaluating")
    
    try:
        with torch.no_grad():
            for batch_idx, (sequences, targets) in enumerate(progress_bar):
                # Stop after max_batches if specified
                if max_batches is not None and batch_idx >= max_batches:
                    print(f"Reached max_batches limit ({max_batches}). Stopping early.")
                    break
                
                # Start timing this batch
                batch_start = time.time()
                
                # Check and print shapes if in debug mode
                if debug and batch_idx == 0:
                    print(f"\nEval input shapes - sequences: {sequences.shape}, targets: {targets.shape}")
                
                # Move data to device
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                # Reset model state at beginning of each sequence
                model.reset_state()
                
                # Reshape sequences if needed - current format is [T, B, C, H, W]
                # If the first dimension is batch, transpose
                if sequences.size(0) != 200:  # Expected timesteps
                    # Assume batch first: [B, T, C, H, W] -> [T, B, C, H, W]
                    sequences = sequences.transpose(0, 1)
                    if debug and batch_idx == 0:
                        print(f"Transposed eval sequences shape: {sequences.shape}")
                
                # Process sequence
                # Accumulate spike counts over all timesteps
                time_steps = sequences.shape[0]  # Now we're sure this is the time dimension
                
                # Initial output (all zeros)
                output = None
                
                # Process each timestep
                for t in range(time_steps):
                    # Get current timestep data
                    x_t = sequences[t]
                    
                    # Forward pass
                    timestep_output = model(x_t)
                    
                    # Initialize or accumulate output
                    if output is None:
                        output = timestep_output
                    else:
                        output += timestep_output
                
                # The output should now have shape [batch_size, num_classes]
                if debug and batch_idx == 0:
                    print(f"Final eval output shape: {output.shape}, Targets shape: {targets.shape}")
                
                # Make predictions based on accumulated output
                predictions = torch.argmax(output, dim=1)
                correct = (predictions == targets).sum().item()
                total_correct += correct
                total_samples += targets.size(0)
                
                # Record batch processing time
                batch_end = time.time()
                batch_time = batch_end - batch_start
                batch_times.append(batch_time)
                
                if debug and batch_idx == 0:
                    print(f"Eval batch processing time: {batch_time:.4f} seconds")
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"accuracy": 0.0, "error": str(e)}
    
    # Calculate evaluation metrics
    accuracy = total_correct / max(total_samples, 1) * 100.0
    avg_batch_time = sum(batch_times) / max(len(batch_times), 1) if batch_times else 0
    
    metrics = {
        "accuracy": accuracy,
        "batch_time": avg_batch_time,
        "total_time": sum(batch_times) if batch_times else 0,
        "samples": total_samples
    }
    
    return metrics


def create_trainer(model, cfg):
    """
    Create algorithm-specific trainer based on configuration.
    
    Args:
        model (nn.Module): Neural network model
        cfg (dict): Configuration dictionary
        
    Returns:
        Trainer: Algorithm-specific trainer instance
    """
    algorithm = cfg.get('alg', 'ltgate')
    
    if algorithm == 'ltgate':
        return LTGateTrainer(model, cfg)
    elif algorithm == 'hlop':
        return HLOPTrainer(model, cfg)
    elif algorithm == 'dsdsnn':
        # TODO: Implement DSD-SNN trainer
        raise NotImplementedError("DSD-SNN trainer not implemented yet")
    else:
        # For baseline, we don't need a special trainer
        return None


def main():
    """
    Main training function.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Set config path based on algorithm if not explicitly provided
    if args.config is None:
        args.config = f"configs/{args.alg}.yaml"
    
    # Load configuration
    cfg = load_config(args.config)
    cfg = update_config_with_args(cfg, args)
    
    # Set up training environment
    device, seed = setup_training(cfg)
    
    # Load datasets
    dataset_name = args.dataset
    debug_mode = cfg.get('debug', False)
    
    print(f"Loading datasets with debug={debug_mode}")
    
    # Use smaller batch size and fewer samples in debug mode
    if debug_mode:
        batch_size = min(cfg['batch_size'], 16)  # Smaller batch size in debug mode
        max_samples = 100  # Limit samples in debug mode
    else:
        batch_size = cfg['batch_size']
        max_samples = None
    
    train_loader = load_dataset(
        dataset_name, 
        DatasetSplit.TRAIN, 
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        max_samples=max_samples,
        debug=debug_mode
    )
    
    val_loader = load_dataset(
        dataset_name, 
        DatasetSplit.VAL, 
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        max_samples=max_samples,
        debug=debug_mode
    )
    
    # Build model
    model = build_backbone(cfg)
    model = model.to(device)
    
    # Create algorithm-specific trainer
    trainer = create_trainer(model, cfg)
    
    # Print model information
    print(f"Model: {type(model).__name__}")
    print(f"Algorithm: {cfg['alg']}")
    print(f"Dataset: {dataset_name}")
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(cfg['epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        
        # Training phase
        train_metrics = train_epoch(model, trainer, train_loader, device, debug=cfg.get('debug', False))
        print(f"Train accuracy: {train_metrics['accuracy']:.2f}%")
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, device, debug=cfg.get('debug', False))
        print(f"Validation accuracy: {val_metrics['accuracy']:.2f}%")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            print(f"New best validation accuracy: {best_val_acc:.2f}%")
            
            # Create directory for saving models if it doesn't exist
            os.makedirs('checkpoints', exist_ok=True)
            
            # Save model checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f"checkpoints/{cfg['alg']}_{dataset_name}_{timestamp}.pt"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_accuracy': best_val_acc,
                'config': cfg,
            }, checkpoint_path)
            
            print(f"Model saved to {checkpoint_path}")
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
