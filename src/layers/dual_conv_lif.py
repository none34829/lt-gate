import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.dual_lif_neuron import DualLIFNeuron

class DualConvLIF(nn.Module):
    """
    Dual-pathway convolutional layer with DualLIF neurons for the LT-Gate algorithm.
    
    Implements separate fast and slow convolutional pathways connected to
    a dual-compartment LIF neuron model.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 tau_fast=5e-3, tau_slow=100e-3, dt=1e-3, threshold=0.5,
                 reset_mechanism="subtract", variance_lambda=0.01):
        """Initialize DualConvLIF layer
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size for convolution
            stride (int): Stride for convolution
            padding (int): Padding for convolution
            tau_fast (float): Fast membrane time constant in seconds
            tau_slow (float): Slow membrane time constant in seconds
            dt (float): Simulation time step in seconds
            threshold (float): Firing threshold
            reset_mechanism (str): Reset mechanism after spike - 'subtract' or 'zero'
            variance_lambda (float): Decay constant for variance tracking
        """
        super().__init__()
        
        # Ensure parameters are proper types
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create two separate convolutional pathways (fast and slow)
        self.conv_fast = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        
        self.conv_slow = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        
        # Create DualLIF neurons
        # Note: we'll initialize with dummy neuron count and update during forward pass
        # because we don't know spatial dimensions yet
        self.neuron_count = 1
        self.dual_lif = DualLIFNeuron(
            num_neurons=self.neuron_count,
            tau_fast=float(tau_fast),
            tau_slow=float(tau_slow),
            dt=float(dt),
            threshold=float(threshold),
            reset_mechanism=reset_mechanism,
            variance_lambda=float(variance_lambda)
        )
        
        # Hebbian learning parameters
        self.enable_learning = False  # Will be enabled during training
        self.learning_rate = 0.01
        
        # For tracking layer output size
        self.output_height = None
        self.output_width = None
    
    def forward(self, x):
        """Forward pass through DualConvLIF layer
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width]
            
        Returns:
            torch.Tensor: Output spikes of shape [batch_size, out_channels, height', width']
        """
        # Apply convolutions for fast and slow pathways
        # Each outputs: [batch_size, out_channels, height', width']
        fast_out = self.conv_fast(x)
        slow_out = self.conv_slow(x)
        
        # Store output spatial dimensions for reference
        batch_size, channels, height, width = fast_out.shape
        self.output_height = height
        self.output_width = width
        
        # Total number of neurons (channels * height * width)
        num_neurons = channels * height * width
        
        # Reshape to [batch_size, num_neurons] for LIF processing
        fast_flat = fast_out.reshape(batch_size, num_neurons)
        slow_flat = slow_out.reshape(batch_size, num_neurons)
        
        # Combine inputs for dual LIF processing
        # Using element-wise averaging for combined input
        combined_input = (fast_flat + slow_flat) / 2.0
        
        # Update neuron count if needed
        if num_neurons != self.neuron_count:
            self.neuron_count = num_neurons
            # Reinitialize DualLIF neuron with correct neuron count
            tau_fast = self.dual_lif.tau_fast
            tau_slow = self.dual_lif.tau_slow
            dt = self.dual_lif.dt
            threshold = self.dual_lif.threshold
            reset_mech = self.dual_lif.reset_mechanism
            var_lambda = self.dual_lif.lam
            self.dual_lif = DualLIFNeuron(
                num_neurons, tau_fast, tau_slow, dt, threshold, reset_mech, var_lambda
            )
        
        # Process through DualLIF neurons
        spikes_fast, spikes_slow, spikes_merged = self.dual_lif(combined_input)
        
        # Apply Hebbian learning if enabled (during training)
        if self.training and self.enable_learning:
            self._apply_hebbian_update(combined_input, spikes_fast, spikes_slow)
        
        # Reshape back to [batch_size, channels, height, width]
        spikes = spikes_merged.reshape(batch_size, channels, height, width)
        
        return spikes
    
    def _apply_hebbian_update(self, inputs, spikes_fast, spikes_slow):
        """Apply local Hebbian learning updates to weights
        
        Args:
            inputs (torch.Tensor): Input tensor [batch_size, num_neurons]
            spikes_fast (torch.Tensor): Fast pathway spikes [batch_size, num_neurons]
            spikes_slow (torch.Tensor): Slow pathway spikes [batch_size, num_neurons]
        """
        # Simple Hebbian update: strengthen weights where both pre and post are active
        # We'll use batch-averaged activity
        with torch.no_grad():
            batch_size = inputs.shape[0]
            # Calculate pre-post correlation across batch samples
            prepost_fast = torch.mm(inputs.t(), spikes_fast) / batch_size
            prepost_slow = torch.mm(inputs.t(), spikes_slow) / batch_size
            
            # Apply updates to weights
            # Note: in a real implementation, we would need to map these updates
            # back to the convolutional weights properly
            delta_fast = self.learning_rate * prepost_fast
            delta_slow = self.learning_rate * prepost_slow
            
            # For now, just update the internal weights of the DualLIF neuron
            self.dual_lif.weight_fast += delta_fast
            self.dual_lif.weight_slow += delta_slow
    
    def reset_state(self):
        """Reset neuron states"""
        self.dual_lif.reset_state()
    
    def enable_hebbian_learning(self, enable=True, learning_rate=None):
        """Enable or disable Hebbian learning
        
        Args:
            enable (bool): Whether to enable Hebbian learning
            learning_rate (float, optional): Learning rate for Hebbian updates
        """
        self.enable_learning = enable
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
