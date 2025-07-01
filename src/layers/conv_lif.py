import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from src.layers.lif_neuron import LIFNeuron

class ConvLIF(nn.Module):
    """
    Convolutional layer with LIF neurons for baseline algorithms.
    
    Wrapper that applies a 2D convolution followed by LIF dynamics.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 tau_fast=5e-3, tau_slow=None, dt=1e-3, threshold=0.5,
                 reset_mechanism="subtract"):
        """Initialize ConvLIF layer
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size for convolution
            stride (int): Stride for convolution
            padding (int): Padding for convolution
            tau_fast (float): Membrane time constant in seconds
            tau_slow (float): Optional slow time constant (not used in single compartment)
            dt (float): Simulation time step in seconds
            threshold (float): Firing threshold
            reset_mechanism (str): Reset mechanism after spike - 'subtract' or 'zero'
        """
        super().__init__()
        
        # Ensure parameters are proper types
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create convolutional layer (without bias by default for SNNs)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        
        # Create LIF neurons
        # Note: we'll initialize with dummy neuron count and update during forward pass
        # because we don't know spatial dimensions yet
        self.neuron_count = 1
        self.lif = LIFNeuron(
            num_neurons=self.neuron_count,
            tau_mem=float(tau_fast),  # Use tau_fast as the membrane time constant
            dt=float(dt),
            threshold=float(threshold),
            reset_mechanism=reset_mechanism
        )
        
        # For tracking layer output size
        self.output_height = None
        self.output_width = None
    
    def forward(self, x):
        """Forward pass through ConvLIF layer
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width]
            
        Returns:
            torch.Tensor: Output spikes of shape [batch_size, out_channels, height', width']
        """
        # Apply convolution: [batch_size, out_channels, height', width']
        conv_out = self.conv(x)
        
        # Store output spatial dimensions for reference
        batch_size, channels, height, width = conv_out.shape
        self.output_height = height
        self.output_width = width
        
        # Total number of neurons (channels * height * width)
        num_neurons = channels * height * width
        
        # Reshape to [batch_size, num_neurons] for LIF processing
        # We need to properly flatten the spatial and channel dimensions
        conv_flat = conv_out.reshape(batch_size, num_neurons)
        
        # Update neuron count if needed
        if num_neurons != self.neuron_count:
            self.neuron_count = num_neurons
            # Reinitialize LIF neuron with correct neuron count
            tau = self.lif.tau_mem
            dt = self.lif.dt
            threshold = self.lif.threshold
            reset_mech = self.lif.reset_mechanism
            self.lif = LIFNeuron(num_neurons, tau, dt, threshold, reset_mech)
        
        # Process through LIF neurons
        spikes_flat = self.lif(conv_flat)
        
        # Reshape back to [batch_size, channels, height, width]
        spikes = spikes_flat.reshape(batch_size, channels, height, width)
        
        return spikes
    
    def reset_state(self):
        """Reset neuron states"""
        self.lif.reset_state()
