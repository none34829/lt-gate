import torch  # type: ignore
import torch.nn as nn  # type: ignore
import numpy as np

# For Lava-DL compatibility
# type: ignore[import]
try:
    from lava.lib.dl.slayer.block import slayer_block  # type: ignore
    HAS_LAVA = True
except ImportError:
    HAS_LAVA = False
    # Create dummy decorator if lava is not available
    def slayer_block(cls):
        return cls

@slayer_block
class DualLIFNeuron(nn.Module):
    """
    Dual-compartment Leaky Integrate-and-Fire neuron with fast and slow time constants.
    Used for LT-Gate algorithm.
    
    Input shape: [batch_size, num_neurons]
    Output shape: [batch_size, num_neurons]
    """
    def __init__(self, num_neurons, tau_fast=5e-3, tau_slow=100e-3, dt=1e-3, threshold=0.5, 
                 reset_mechanism="subtract", variance_lambda=0.01):
        """Initialize DualLIF neuron model
        
        Args:
            num_neurons (int): Number of neurons
            tau_fast (float): Fast membrane time constant in seconds
            tau_slow (float): Slow membrane time constant in seconds
            dt (float): Simulation time step in seconds
            threshold (float): Firing threshold
            reset_mechanism (str): Reset mechanism after spike - 'subtract' or 'zero'
            variance_lambda (float): Decay constant for variance tracking
        """
        super().__init__()
        
        # Convert parameters to proper types
        self.num_neurons = int(num_neurons)
        self.tau_fast = float(tau_fast)
        self.tau_slow = float(tau_slow)
        self.dt = float(dt)
        self.threshold = float(threshold)
        self.lam = float(variance_lambda)  # For variance tracking
        
        # Compute decay factors
        self.decay_fast = float(np.exp(-dt / tau_fast))
        self.decay_slow = float(np.exp(-dt / tau_slow))
        
        # Initialize state variables
        self.register_buffer('membrane_fast', torch.zeros(num_neurons))
        self.register_buffer('membrane_slow', torch.zeros(num_neurons))
        
        # Initialize variance trackers
        self.register_buffer('variance_fast', torch.zeros(num_neurons))
        self.register_buffer('variance_slow', torch.zeros(num_neurons))
        
        # Store reset mechanism
        assert reset_mechanism in ["subtract", "zero"]
        self.reset_mechanism = reset_mechanism
        
        # For simple neuron layers, we don't need weight matrices
        # Input currents directly affect membrane potentials
        self.register_parameter('weight_fast', nn.Parameter(torch.ones(1)))
        self.register_parameter('weight_slow', nn.Parameter(torch.ones(1)))
        
        # Spike counters for monitoring
        self.spike_count_fast = 0
        self.spike_count_slow = 0
        self.spike_count_total = 0
    
    def forward(self, input_current):
        """Forward pass with dual-compartment dynamics
        
        Args:
            input_current (torch.Tensor): Input current of shape [batch_size, num_neurons]
            
        Returns:
            tuple: (fast_spikes, slow_spikes, merged_spikes) each of shape [batch_size, num_neurons]
        """
        batch_size = input_current.shape[0]
        
        # Map input to membrane potentials for both compartments
        # Use scalar weights for simple scaling
        # Ensure weights are on the same device as input
        weight_fast = self.weight_fast.to(input_current.device)
        weight_slow = self.weight_slow.to(input_current.device)
        current_fast = input_current * weight_fast
        current_slow = input_current * weight_slow
        
        # Expand membrane potentials to match batch size
        # Ensure membrane potentials are on the same device as input
        membrane_fast_batch = self.membrane_fast.to(input_current.device).unsqueeze(0).expand(batch_size, -1)
        membrane_slow_batch = self.membrane_slow.to(input_current.device).unsqueeze(0).expand(batch_size, -1)
        
        # Decay and integrate
        membrane_fast_batch = self.decay_fast * membrane_fast_batch + current_fast
        membrane_slow_batch = self.decay_slow * membrane_slow_batch + current_slow
        
        # Check threshold and generate spikes
        spikes_fast = (membrane_fast_batch >= self.threshold).float()
        spikes_slow = (membrane_slow_batch >= self.threshold).float()
        
        # Reset membrane potentials where spikes occurred
        if self.reset_mechanism == "subtract":
            membrane_fast_batch = membrane_fast_batch - spikes_fast * self.threshold
            membrane_slow_batch = membrane_slow_batch - spikes_slow * self.threshold
        else:  # reset to zero
            membrane_fast_batch = membrane_fast_batch * (1 - spikes_fast)
            membrane_slow_batch = membrane_slow_batch * (1 - spikes_slow)
        
        # Store last batch element's membrane potentials for next time step
        with torch.no_grad():
            self.membrane_fast.copy_(membrane_fast_batch[-1].detach().to(self.membrane_fast.device))
            self.membrane_slow.copy_(membrane_slow_batch[-1].detach().to(self.membrane_slow.device))
        
        # Update spike variance tracking
        if batch_size > 1:  # Only calculate variance if we have a batch
            # Compute spike rate per neuron (mean across batch)
            mean_fast = spikes_fast.mean(dim=0)
            mean_slow = spikes_slow.mean(dim=0)
            
            # Update variance estimates with exponential decay
            # Ensure variance buffers are on the same device as the computation
            variance_fast = self.variance_fast.to(input_current.device)
            variance_slow = self.variance_slow.to(input_current.device)
            variance_fast.mul_(1-self.lam).add_(
                ((mean_fast - mean_fast.mean())**2) * self.lam
            )
            variance_slow.mul_(1-self.lam).add_(
                ((mean_slow - mean_slow.mean())**2) * self.lam
            )
            # Copy back to the original device
            self.variance_fast.copy_(variance_fast.to(self.variance_fast.device))
            self.variance_slow.copy_(variance_slow.to(self.variance_slow.device))
        
        # Merge spikes from both compartments (any spike counts)
        merged_spikes = torch.clamp(spikes_fast + spikes_slow, 0, 1)
        
        # Update spike counters if in training mode
        if self.training:
            self.spike_count_fast += spikes_fast.sum().item()
            self.spike_count_slow += spikes_slow.sum().item()
            self.spike_count_total += merged_spikes.sum().item()
        
        return spikes_fast, spikes_slow, merged_spikes
    
    def to(self, *args, **kwargs):
        """Override to ensure all buffers and parameters are moved to the correct device"""
        result = super().to(*args, **kwargs)
        device = torch._C._nn._parse_to(*args, **kwargs)[0]  # get target device
        # move buffers manually
        self.membrane_fast = self.membrane_fast.to(device)
        self.membrane_slow = self.membrane_slow.to(device)
        self.variance_fast = self.variance_fast.to(device)
        self.variance_slow = self.variance_slow.to(device)
        # Also ensure parameters are on the correct device
        self.weight_fast = self.weight_fast.to(device)
        self.weight_slow = self.weight_slow.to(device)
        return result
    
    def reset_state(self):
        """Reset neuron state (membrane potentials, variance tracking, spike counts)"""
        self.membrane_fast.zero_()
        self.membrane_slow.zero_()
        self.variance_fast.zero_()
        self.variance_slow.zero_()
        self.spike_count_fast = 0
        self.spike_count_slow = 0
        self.spike_count_total = 0
