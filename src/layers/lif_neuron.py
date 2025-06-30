import torch
import torch.nn as nn
import numpy as np

# For Lava-DL compatibility
try:
    from lava.lib.dl.slayer.block import slayer_block
    HAS_LAVA = True
except ImportError:
    HAS_LAVA = False
    # Create dummy decorator if lava is not available
    def slayer_block(cls):
        return cls

@slayer_block
class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with single time constant.
    Used for baseline algorithms (HLOP, DSD-SNN).
    
    Input shape: [batch_size, num_neurons]
    Output shape: [batch_size, num_neurons]
    """
    def __init__(self, num_neurons, tau_mem=5e-3, dt=1e-3, threshold=0.5, reset_mechanism="subtract"):
        """Initialize LIF neuron model
        
        Args:
            num_neurons (int): Number of neurons
            tau_mem (float): Membrane time constant in seconds
            dt (float): Simulation time step in seconds
            threshold (float): Firing threshold
            reset_mechanism (str): Reset mechanism after spike - 'subtract' or 'zero'
        """
        super().__init__()
        
        # Convert parameters to proper types
        self.num_neurons = int(num_neurons)
        self.tau_mem = float(tau_mem)
        self.dt = float(dt)
        self.threshold = float(threshold)
        
        # Compute decay factor
        self.decay = float(np.exp(-dt / tau_mem))
        
        # Initialize state variables
        self.register_buffer('membrane_potential', torch.zeros(num_neurons))
        
        # Store reset mechanism
        assert reset_mechanism in ["subtract", "zero"]
        self.reset_mechanism = reset_mechanism
        
        # Spike count for monitoring
        self.spike_count = 0
    
    def forward(self, input_current):
        """Forward pass - leaky integration and spike generation
        
        Args:
            input_current (torch.Tensor): Input current of shape [batch_size, num_neurons]
            
        Returns:
            torch.Tensor: Output spikes of shape [batch_size, num_neurons]
        """
        batch_size = input_current.shape[0]
        
        # Expand membrane potential to match batch size
        membrane = self.membrane_potential.unsqueeze(0).expand(batch_size, -1)
        
        # Decay and integrate
        membrane = self.decay * membrane + input_current
        
        # Check threshold and generate spikes
        spikes = (membrane >= self.threshold).float()
        
        # Reset membrane potential
        if self.reset_mechanism == "subtract":
            membrane = membrane - spikes * self.threshold
        else:  # reset to zero
            membrane = membrane * (1 - spikes)
        
        # Store last batch element's membrane potential for next time step
        # (could also use mean across batch if desired)
        with torch.no_grad():
            self.membrane_potential.copy_(membrane[-1].detach())
        
        # Update spike counter if in training mode
        if self.training:
            self.spike_count += spikes.sum().item()
            
        return spikes
    
    def reset_state(self):
        """Reset neuron state (membrane potential, spike count)"""
        self.membrane_potential.zero_()
        self.spike_count = 0
