#!/usr/bin/env python3
"""
Simple evaluation script for LT-Gate models.
"""

import argparse
import sys
import torch
import json
from pathlib import Path
from datetime import datetime

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent))

from checkpoint import load_checkpoint
from model import build_backbone
from data_loader import load_dataset, DatasetSplit

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
    from tqdm import tqdm
    progress_bar = tqdm(data_loader, desc="Evaluating")
    
    try:
        import time
        for batch_idx, (sequences, targets) in enumerate(progress_bar):
            if max_batches and batch_idx >= max_batches:
                break
                
            batch_start = time.time()
            
            # Move data to device
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Get time dimension
            if sequences.dim() == 3:  # [batch, time, features]
                time_steps = sequences.size(1)
            else:  # [time, batch, features]
                time_steps = sequences.size(0)
                sequences = sequences.transpose(0, 1)  # Convert to [batch, time, features]
            
            # Now we're sure this is the time dimension
            # Initial output (all zeros)
            output = None
            
            # Process each timestep
            for t in range(time_steps):
                # Get current timestep data
                x_t = sequences[:, t, :] if sequences.dim() == 3 else sequences[t]
                
                # Forward pass - get scores and features
                spikes_t, scores_t, fc_out_t = model(x_t, return_features=True)
                
                # Initialize or accumulate scores (not spikes)
                if output is None:
                    output = scores_t
                else:
                    output += scores_t
            
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
        return {"accuracy": 0.0, "error": str(e), "samples": 0, "batch_time": 0, "total_time": 0}
    
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

def create_compatible_model(checkpoint):
    """
    Create a model that's compatible with the checkpoint architecture.
    This handles the case where the checkpoint was saved with an older model structure.
    """
    from model import SNNBackbone
    from layers.dual_conv_lif import DualConvLIF
    from layers.dual_lif_neuron import DualLIFNeuron
    import torch.nn as nn
    
    config = checkpoint.get('config', {})
    
    # Create a model with the exact architecture that matches the checkpoint
    class CompatibleSNNBackbone(nn.Module):
        def __init__(self, config):
            super().__init__()
            
            # Override with the exact architecture from the checkpoint
            # Based on the state dict keys, we know it has conv0, conv1, conv2, fc, readout
            
            # Extract time constants
            self.tau_fast = float(config.get('tau_fast', 5e-3))
            self.tau_slow = float(config.get('tau_slow', 100e-3))
            self.dt = float(config.get('dt', 1e-3))
            
            # Create the exact conv layers that match the checkpoint
            # From the config, we know the architecture:
            # conv0: 1->16, conv1: 16->32, conv2: 32->64
            conv_configs = config.get('conv_layers', [])
            if conv_configs:
                # Calculate expected output dimensions for each layer
                # Input: 28x28
                # conv0: 1->16, stride=1, padding=1 -> 28x28
                # conv1: 16->32, stride=2, padding=1 -> 14x14  
                # conv2: 32->64, stride=2, padding=1 -> 7x7
                
                # Create conv layers with pre-calculated neuron counts
                conv0_neurons = 16 * 28 * 28  # 12544
                conv1_neurons = 32 * 14 * 14  # 6272
                conv2_neurons = 64 * 7 * 7    # 3136
                
                self.conv0 = DualConvLIF(
                    in_channels=conv_configs[0]['in_channels'],
                    out_channels=conv_configs[0]['out_channels'],
                    kernel_size=conv_configs[0]['kernel_size'],
                    stride=conv_configs[0]['stride'],
                    padding=conv_configs[0].get('padding', 1),
                    tau_fast=self.tau_fast,
                    tau_slow=self.tau_slow,
                    dt=self.dt
                )
                # Pre-initialize the DualLIFNeuron with correct size
                self.conv0.neuron_count = conv0_neurons
                self.conv0.dual_lif = DualLIFNeuron(
                    conv0_neurons, self.tau_fast, self.tau_slow, self.dt, 
                    threshold=0.5, reset_mechanism="subtract", variance_lambda=0.01
                )
                
                self.conv1 = DualConvLIF(
                    in_channels=conv_configs[1]['in_channels'],
                    out_channels=conv_configs[1]['out_channels'],
                    kernel_size=conv_configs[1]['kernel_size'],
                    stride=conv_configs[1]['stride'],
                    padding=conv_configs[1].get('padding', 1),
                    tau_fast=self.tau_fast,
                    tau_slow=self.tau_slow,
                    dt=self.dt
                )
                # Pre-initialize the DualLIFNeuron with correct size
                self.conv1.neuron_count = conv1_neurons
                self.conv1.dual_lif = DualLIFNeuron(
                    conv1_neurons, self.tau_fast, self.tau_slow, self.dt, 
                    threshold=0.5, reset_mechanism="subtract", variance_lambda=0.01
                )
                
                self.conv2 = DualConvLIF(
                    in_channels=conv_configs[2]['in_channels'],
                    out_channels=conv_configs[2]['out_channels'],
                    kernel_size=conv_configs[2]['kernel_size'],
                    stride=conv_configs[2]['stride'],
                    padding=conv_configs[2].get('padding', 1),
                    tau_fast=self.tau_fast,
                    tau_slow=self.tau_slow,
                    dt=self.dt
                )
                # Pre-initialize the DualLIFNeuron with correct size
                self.conv2.neuron_count = conv2_neurons
                self.conv2.dual_lif = DualLIFNeuron(
                    conv2_neurons, self.tau_fast, self.tau_slow, self.dt, 
                    threshold=0.5, reset_mechanism="subtract", variance_lambda=0.01
                )
            else:
                # Fallback to default
                self.conv0 = DualConvLIF(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    tau_fast=self.tau_fast,
                    tau_slow=self.tau_slow,
                    dt=self.dt
                )
                
                self.conv1 = DualConvLIF(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    tau_fast=self.tau_fast,
                    tau_slow=self.tau_slow,
                    dt=self.dt
                )
                
                self.conv2 = DualConvLIF(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    tau_fast=self.tau_fast,
                    tau_slow=self.tau_slow,
                    dt=self.dt
                )
            
            # Flatten layer
            self.flatten = nn.Flatten()
            
            # Get FC layer sizes from config
            fc_configs = config.get('fc_layers', [])
            if fc_configs:
                self.flat_size = fc_configs[0]['in_features']
                fc_size = fc_configs[0]['out_features']
                readout_size = fc_configs[1]['out_features']
            else:
                # Fallback sizes
                self.flat_size = 3136  # Based on the checkpoint
                fc_size = 256
                readout_size = 10
            
            # FC and readout layers
            self.fc = DualLIFNeuron(
                num_neurons=fc_size,
                tau_fast=self.tau_fast,
                tau_slow=self.tau_slow,
                dt=self.dt
            )
            
            self.readout = DualLIFNeuron(
                num_neurons=readout_size,
                tau_fast=self.tau_fast,
                tau_slow=self.tau_slow,
                dt=self.dt
            )
            
            # FC layer weights
            self.fc_weights = nn.Parameter(torch.randn(fc_size, self.flat_size) * 0.01)
            # Readout weights
            self.readout_weights = nn.Parameter(torch.randn(readout_size, fc_size) * 0.01)
            
            # Set dual flag
            self.is_dual = True
        
        def forward(self, x, return_features=False):
            """Forward pass through the network"""
            # Process through convolutional layers
            x, _ = self.conv0(x)  # DualConvLIF returns (spikes, recon_loss)
            x, _ = self.conv1(x)
            x, _ = self.conv2(x)
            
            # Flatten output for FC layer
            x = self.flatten(x)
            
            # Apply FC layer with weights
            fc_input = torch.matmul(x, self.fc_weights.t())
            
            # Process through FC layer
            _, _, fc_out = self.fc(fc_input)
            
            # Apply readout layer with weights
            readout_input = torch.matmul(fc_out, self.readout_weights.t())
            
            # Process through readout layer
            _, _, spikes = self.readout(readout_input)
            
            if return_features:
                return spikes, spikes, fc_out
            else:
                return spikes
        
        def reset_state(self):
            """Reset all neuron states in the network"""
            self.conv0.reset_state()
            self.conv1.reset_state()
            self.conv2.reset_state()
            self.fc.reset_state()
            self.readout.reset_state()
    
    return CompatibleSNNBackbone(config)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained LT-Gate model")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model checkpoint file")
    parser.add_argument("--dataset", type=str, default="fast", choices=["fast", "slow", "both"],
                       help="Dataset to evaluate on")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                       help="Dataset split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU acceleration")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (fewer samples)")
    
    args = parser.parse_args()
    
    # Check if checkpoint file exists
    checkpoint_path = Path(args.model_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file {args.model_path} not found!")
        print("\nAvailable checkpoints in checkpoints/ directory:")
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            for ckpt in checkpoints_dir.glob("*.pt"):
                print(f"  - {ckpt.name}")
        sys.exit(1)
    
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    if not config:
        print("Warning: No config found in checkpoint, using default config")
        config = {
            'alg': 'ltgate',
            'input_size': 128,
            'hidden_size': 256,
            'num_classes': 10,
            'num_layers': 2,
            'dropout': 0.1,
            'batch_norm': True
        }
    
    # Create compatible model
    print(f"Creating compatible model with algorithm: {config.get('alg', 'ltgate')}")
    model = create_compatible_model(checkpoint)
    model.to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create data loader
    print(f"Creating data loader for {args.dataset} dataset, {args.split} split")
    
    # Convert split string to DatasetSplit enum
    if args.split == "train":
        split = DatasetSplit.TRAIN
    elif args.split == "val":
        split = DatasetSplit.VAL
    else:  # test
        split = DatasetSplit.TEST
    
    data_loader = load_dataset(
        dataset_name=args.dataset,
        split=split,
        batch_size=args.batch_size,
        shuffle=False,  # No shuffling for evaluation
        debug=args.debug
    )
    
    # Run evaluation
    print(f"Starting evaluation on {args.split} split...")
    metrics = evaluate(model, data_loader, device, debug=args.debug)
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Total samples: {metrics['samples']}")
    print(f"Average batch time: {metrics['batch_time']:.4f} seconds")
    print(f"Total evaluation time: {metrics['total_time']:.2f} seconds")
    
    # Save results to JSON file
    results = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get('epoch', 'unknown'),
        "val_accuracy": checkpoint.get('val_accuracy', 'unknown'),
        "dataset": args.dataset,
        "split": args.split,
        "test_accuracy": metrics['accuracy'],
        "test_accuracy_percent": metrics['accuracy'],
        "total_samples": metrics['samples'],
        "batch_time": metrics['batch_time'],
        "total_time": metrics['total_time'],
        "device": str(device)
    }
    
    # Create results filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_{args.dataset}_{args.split}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()

