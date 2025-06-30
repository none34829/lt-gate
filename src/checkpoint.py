"""
Checkpoint management utilities for LT-Gate training.
"""

import os
import torch
from pathlib import Path


def save_checkpoint(model, epoch, tag, metrics=None, ckpt_dir="ckpts"):
    """
    Save model checkpoint with RNG states and metrics.
    
    Args:
        model (nn.Module): Model to save
        epoch (int): Current epoch number
        tag (str): Checkpoint tag (e.g., 'task1_ltgate')
        metrics (dict, optional): Dictionary of metrics to save
        ckpt_dir (str): Directory to save checkpoints
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "metrics": metrics or {}
    }
    
    # Save checkpoint
    path = Path(ckpt_dir) / f"{tag}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, path, device=None):
    """
    Load model checkpoint and restore RNG states.
    
    Args:
        model (nn.Module): Model to load weights into
        path (str): Path to checkpoint file
        device (torch.device, optional): Device to load checkpoint to
        
    Returns:
        tuple: (epoch, metrics) - Last epoch number and saved metrics
    """
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    
    # Restore model state
    model.load_state_dict(checkpoint["state_dict"])
    
    # Restore RNG states
    torch.set_rng_state(checkpoint["rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    
    print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"], checkpoint.get("metrics", {})
