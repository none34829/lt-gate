import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

class LIFNeuron(nn.Module):
    """Single-compartment Leaky Integrate-and-Fire neuron."""
    def __init__(self, tau: float = 0.01):
        super().__init__()
        self.tau = tau
        self.v = None  # membrane potential

    def reset_state(self):
        """Reset membrane potential."""
        self.v = None

    def forward(self, x):
        batch_size = x.shape[0]
        if self.v is None:
            self.v = torch.zeros_like(x)
        
        # Update membrane potential
        self.v = self.v + (-self.v + x) * self.tau
        
        # Spike generation (threshold = 1.0)
        spike = (self.v >= 1.0).float()
        self.v = self.v * (1 - spike)  # Reset where spiked
        
        return spike

class SingleConv(nn.Module):
    """Convolutional layer with LIF neurons."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, tau: float = 0.01):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=0)
        self.lif = LIFNeuron(tau=tau)
        
        # Initialize weights
        nn.init.xavier_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def reset_state(self):
        """Reset neuron states."""
        self.lif.reset_state()

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor of shape [B, C, H, W]
        Returns:
            Spike tensor of shape [B, out_ch, H', W']
        """
        z = self.conv(x)
        spikes = self.lif(z)
        return spikes
