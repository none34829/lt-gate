"""
Core metric functions for evaluating LT-Gate and baseline algorithms.
"""

import torch
import numpy as np
import h5py
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union


def accuracy(model: torch.nn.Module,
            loader: torch.utils.data.DataLoader,
            device: str = 'cpu') -> float:
    """
    Calculate accuracy for a model on a given dataset.
    
    Args:
        model: Neural network model
        loader: DataLoader providing sequences and labels
        device: Device to run evaluation on
        
    Returns:
        float: Accuracy as a fraction between 0 and 1
    """
    correct, total = 0, 0
    model.eval()
    
    with torch.no_grad():
        for seq, lab in loader:
            seq = seq.to(device)
            # Forward pass - accumulate spikes over time
            out = model(seq)  # [T, B, num_classes] or [B, T, num_classes]
            
            # Handle different output formats
            if out.size(0) == seq.size(0):  # Time-first format
                preds = out.sum(0).argmax(-1)  # Sum over time, then get class
            else:  # Batch-first format
                preds = out.sum(1).argmax(-1)
            
            # Update counts
            correct += (preds.cpu() == lab).sum().item()
            total += lab.size(0)
    
    return correct / total if total > 0 else 0.0


def forgetting(acc_before: List[float], acc_after: List[float]) -> float:
    """
    Calculate forgetting metric between two accuracy measurements.
    
    Args:
        acc_before: List of task-wise accuracies before training new task
        acc_after: List of task-wise accuracies after training new task
        
    Returns:
        float: Average forgetting across tasks
    """
    if len(acc_before) != len(acc_after):
        raise ValueError("Accuracy lists must have same length")
    
    # Convert to numpy for vectorized operations
    before = np.array(acc_before)
    after = np.array(acc_after)
    
    # Calculate forgetting for each task
    forgetting = before - after
    
    # Return mean forgetting
    return float(np.mean(forgetting))


def load_energy(path: Union[str, Path]) -> Tuple[float, Optional[float]]:
    """
    Load energy measurements from Akida or Loihi JSON file.
    
    Args:
        path: Path to energy measurement JSON file
        
    Returns:
        tuple: (energy in Joules, wallclock time in seconds if available)
    """
    with open(path) as f:
        data = json.load(f)
    
    # Handle different energy units
    if 'energy_J' in data:
        energy = data['energy_J']
    elif 'energy_mJ' in data:
        energy = data['energy_mJ'] / 1000.0
    else:
        raise ValueError(f'Energy field not found in {path}')
    
    # Get wallclock time if available
    wallclock = data.get('wallclock_s', None)
    
    return energy, wallclock


def compute_spike_stats(spike_counts: Dict[str, int],
                       num_samples: int) -> Dict[str, float]:
    """
    Compute per-layer and total spike statistics.
    
    Args:
        spike_counts: Dictionary mapping layer names to spike counts
        num_samples: Number of samples processed
        
    Returns:
        dict: Statistics including spikes per sample per layer
    """
    stats = {}
    
    # Compute per-layer stats
    for layer, count in spike_counts.items():
        stats[f'spikes_per_sample_{layer}'] = count / num_samples
    
    # Compute total spikes per sample
    stats['total_spikes_per_sample'] = sum(spike_counts.values()) / num_samples
    
    return stats


def load_training_metrics(csv_path: Union[str, Path]) -> Dict[str, List[float]]:
    """
    Load training metrics from CSV file.
    
    Args:
        csv_path: Path to training metrics CSV
        
    Returns:
        dict: Dictionary of metric lists including accuracy, spikes, etc.
    """
    import pandas as pd
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Group by phase and get final metrics
    metrics = {}
    for phase in df['phase'].unique():
        phase_df = df[df['phase'] == phase]
        metrics[f'{phase}_final_acc'] = phase_df['acc'].iloc[-1]
        if 'spikes' in df.columns:
            metrics[f'{phase}_spikes_per_sample'] = phase_df['spikes'].iloc[-1]
        if 'γ_mean' in df.columns:
            metrics[f'{phase}_gamma_mean'] = phase_df['γ_mean'].iloc[-1]
    
    return metrics
