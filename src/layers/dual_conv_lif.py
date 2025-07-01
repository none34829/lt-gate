import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from src.layers.dual_lif_neuron import DualLIFNeuron

class DualConvLIF(nn.Module):
    """
    Dual-pathway convolutional layer with DualLIF neurons for the LT-Gate algorithm.
    
    Implements separate fast and slow convolutional pathways connected to
    a dual-compartment LIF neuron model.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 tau_fast=5e-3, tau_slow=100e-3, dt=1e-3, threshold=0.5,
                 reset_mechanism="subtract", variance_lambda=0.01,
                 eta_h=1e-3):
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
            eta_h (float): Learning rate for Hebbian updates
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
        self.eta_h = eta_h  # Learning rate for Hebbian updates
        self.mse = nn.MSELoss(reduction='mean')
        
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
        recon_loss = 0.0
        if self.training and self.enable_learning:
            recon_loss = self._apply_hebbian_update(x, spikes_fast, spikes_slow)
        
        # Reshape back to [batch_size, channels, height, width]
        spikes = spikes_merged.reshape(batch_size, channels, height, width)
        
        return spikes, recon_loss
    
    @torch.no_grad()
    def _apply_hebbian_update(self, input_tensor, spikes_fast, spikes_slow):
        """Apply Hebbian learning update to weights.
        
        Args:
            input_tensor (torch.Tensor): Input to the layer
            spikes_fast (torch.Tensor): Fast neuron spikes
            spikes_slow (torch.Tensor): Slow neuron spikes
            
        Returns:
            float: Reconstruction loss for this update
        """
        if not self.enable_learning:
            return 0.0
        
        # Compute reconstruction loss
        recon_loss = 0.0
        
        # Fast pathway update and loss
        dw_fast = torch.mm(spikes_fast.t(), input_tensor) * self.eta_h
        self.conv_fast.weight.data += dw_fast.view_as(self.conv_fast.weight)
        recon_fast = torch.mm(spikes_fast, self.conv_fast.weight.view(-1, self.conv_fast.in_channels))
        recon_loss += self.mse(recon_fast, input_tensor)
        
        # Slow pathway update and loss
        dw_slow = torch.mm(spikes_slow.t(), input_tensor) * self.eta_h
        self.conv_slow.weight.data += dw_slow.view_as(self.conv_slow.weight)
        recon_slow = torch.mm(spikes_slow, self.conv_slow.weight.view(-1, self.conv_slow.in_channels))
        recon_loss += self.mse(recon_slow, input_tensor)
        
        return recon_loss / input_tensor.shape[0]
    
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
