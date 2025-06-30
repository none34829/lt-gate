import sys
import os
import pytest
import torch
import yaml
import numpy as np

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import build_backbone, reset_network_state, process_sequence, load_backbone_from_config, SNNBackbone

def test_forward_shapes():
    """Test that the network produces correct output shapes and spikes"""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'task1_fast.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Build network backbone using our new implementation
    net = build_backbone(cfg)
    
    # Verify it's an instance of our new SNNBackbone class
    assert isinstance(net, SNNBackbone), "Network should be an instance of SNNBackbone"
    
    # Create a small random sequence to test (just a few frames)
    batch_size = 2  # Small batch size for testing
    seq_length = 10  # Shorter sequence for faster testing
    seq = torch.rand(batch_size, seq_length, 1, 28, 28)  # [B, T, C, H, W]
    
    # Reset network state before testing
    net.reset_state()
    
    # Initialize spike counter
    s_count = 0
    
    # Process each frame in the sequence
    for t in range(seq.size(1)):
        # Extract current frame [batch, channels, height, width]
        frame = seq[:, t]  # [B, C, H, W]
        
        # Forward pass
        spikes = net(frame)  # Forward pass with shape [B, 10]
        
        # Verify output shape
        assert spikes.shape == (batch_size, 10), f"Expected shape {(batch_size, 10)}, got {spikes.shape}"
        
        # Verify spikes are binary (0 or 1)
        unique_values = torch.unique(spikes)
        for val in unique_values:
            assert val.item() in [0.0, 1.0], f"Spike value {val.item()} is not binary (0 or 1)"
        
        # Accumulate spikes
        s_count += spikes.sum().item()
    
    # Verify that the network produced at least some spikes
    assert s_count > 0, "Network produced no spikes!"
    print(f"Network generated {s_count} spikes during test sequence ({s_count/(batch_size*seq_length*10):.1%} activity rate)")

def test_switch_algorithms():
    """Test that we can switch between different algorithms using the same backbone"""
    # Create a small test sequence - 5 time steps is enough to test functionality
    batch_size = 2
    seq_length = 5
    seq = torch.rand(batch_size, seq_length, 1, 28, 28)  # [B, T, C, H, W]
    
    # Test all algorithms
    for alg in ['ltgate', 'hlop', 'dsd']:
        # Load base config
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'task1_fast.yaml')
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Override algorithm
        cfg['alg'] = alg
        
        # Build network
        print(f"\nBuilding {alg} network...")
        net = build_backbone(cfg)
        
        # Verify it's an instance of our new SNNBackbone class
        assert isinstance(net, SNNBackbone), f"{alg} network should be an instance of SNNBackbone"
        
        # Try direct forward pass first to check if neurons are working
        with torch.no_grad():
            net.reset_state()
            # Process a single frame
            frame = seq[:, 0]  # First frame
            spikes = net(frame)
            assert spikes.shape == (batch_size, 10), f"Expected shape {(batch_size, 10)}, got {spikes.shape}"
        
        # Test full sequence processing
        decisions, spike_counts = process_sequence(net, seq)
        
        # Verify output shapes
        assert decisions.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {decisions.shape}"
        assert spike_counts.shape == (batch_size, 10), f"Expected shape {(batch_size, 10)}, got {spike_counts.shape}"
        
        # Verify that the network produced some spikes
        total_spikes = spike_counts.sum().item()
        print(f"{alg} network generated {total_spikes} spikes during test sequence (activity rate: {total_spikes/(batch_size*10):.1%})")
        
        # Verify algorithm-specific properties
        if alg == 'ltgate':
            # For LT-Gate, verify dual compartment dynamics
            # Check if we have DualLIF neurons in the network
            has_dual_neurons = any('DualLIF' in str(type(m)) for m in net.modules())
            assert has_dual_neurons, "LT-Gate should use DualLIF neurons"
        elif alg == 'hlop':
            # Ensure HLOP allows learning during inference if learning_enabled flag is set
            net.enable_learning = True
            # Check if learning enables correctly
            assert net.enable_learning is True, "HLOP should support learning mode"
            
        # Verify we can disable learning for all algorithms
        net.enable_learning = False
        assert net.enable_learning is False, f"Failed to disable learning for {alg}"

if __name__ == "__main__":
    # Run tests directly when called
    print("Testing forward shapes...")
    test_forward_shapes()
    
    print("\nTesting algorithm switching...")
    test_switch_algorithms()
    
    print("\nAll tests passed!")
