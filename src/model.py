import torch
import torch.nn as nn
import yaml

# Import our custom layers - new implementation
from src.layers.lif_neuron import LIFNeuron
from src.layers.dual_lif_neuron import DualLIFNeuron
from src.layers.conv_lif import ConvLIF
from src.layers.dual_conv_lif import DualConvLIF

# HLOP projector (we'll implement this properly later)
class HLOPProjector(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weights and masks
        self.register_parameter('weights', nn.Parameter(torch.randn(out_features, in_features) * 0.01))
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def forward(self, x):
        # Apply masking
        masked_weights = self.weights * self.mask
        # Apply projection
        return torch.matmul(x, masked_weights.t())

# For Lava-DL compatibility
try:
    import lava.lib.dl.slayer as slayer
    from lava.lib.dl import slayer2loihi
    HAS_LAVA = True
except ImportError:
    HAS_LAVA = False

class SNNBackbone(nn.Module):
    """Spiking Neural Network backbone architecture that supports
    multiple algorithm variants (LT-Gate, HLOP, DSD-SNN).
    
    The backbone structure follows section 2.1 specifications:
    - Conv0: 1→32 channels, 3x3 kernel
    - Conv1: 32→32 channels, 3x3 kernel
    - Conv2: 32→64 channels, 3x3 kernel
    - Flatten
    - FC: 36864→256 dense dual-τ
    - Readout: 256→10 output units
    """
    
    def __init__(self, config):
        """Initialize SNNBackbone with configuration
        
        Args:
            config (dict): Configuration dictionary with algorithm and parameters
        """
        super().__init__()
        
        # Store configuration
        self.config = config
        self.algorithm = config['alg']  # 'ltgate', 'hlop', or 'dsd'
        
        # Extract time constants from config
        self.tau_fast = float(config.get('tau_fast', 5e-3))
        self.tau_slow = float(config.get('tau_slow', 100e-3))
        self.dt = float(config.get('dt', 1e-3))
        
        # Select appropriate layer types based on algorithm
        if self.algorithm == 'ltgate':
            # LT-Gate uses dual compartment neurons
            ConvLayerType = DualConvLIF
            self.is_dual = True
        else:
            # HLOP and DSD use single compartment neurons
            ConvLayerType = ConvLIF
            self.is_dual = False
        
        # Build network layers
        self.conv0 = ConvLayerType(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            tau_fast=self.tau_fast,
            tau_slow=self.tau_slow,
            dt=self.dt
        )
        
        self.conv1 = ConvLayerType(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            tau_fast=self.tau_fast,
            tau_slow=self.tau_slow,
            dt=self.dt
        )
        
        self.conv2 = ConvLayerType(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            tau_fast=self.tau_fast,
            tau_slow=self.tau_slow,
            dt=self.dt
        )
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Calculate flattened size (assuming 28x28 input)
        # After 3 convs with padding=1 and stride=1, spatial dims remain 28x28
        # Adjust if using different input dimensions or stride > 1
        self.flat_size = 64 * 28 * 28
        
        # Fully connected and readout layers
        if self.is_dual:
            # Dual compartment neurons for LT-Gate
            self.fc = DualLIFNeuron(
                num_neurons=256,
                tau_fast=self.tau_fast,
                tau_slow=self.tau_slow,
                dt=self.dt
            )
            self.readout = DualLIFNeuron(
                num_neurons=10,
                tau_fast=self.tau_fast,
                tau_slow=self.tau_slow,
                dt=self.dt
            )
            # FC layer weights
            self.fc_weights = nn.Parameter(torch.randn(256, self.flat_size) * 0.01)
            # Readout weights
            self.readout_weights = nn.Parameter(torch.randn(10, 256) * 0.01)
        else:
            # Single compartment neurons for baselines
            self.fc = LIFNeuron(
                num_neurons=256,
                tau_mem=self.tau_fast,
                dt=self.dt
            )
            self.readout = LIFNeuron(
                num_neurons=10,
                tau_mem=self.tau_fast,
                dt=self.dt
            )
            # FC layer weights
            self.fc_weights = nn.Parameter(torch.randn(256, self.flat_size) * 0.01)
            # Readout weights
            self.readout_weights = nn.Parameter(torch.randn(10, 256) * 0.01)
    
    def forward(self, x):
        """Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Output spikes/activations of shape [batch_size, num_classes]
        """
        # Process through convolutional layers
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Flatten output for FC layer
        x = self.flatten(x)
        
        # Apply FC layer with weights
        fc_input = torch.matmul(x, self.fc_weights.t())
        
        # Process through FC layer
        if self.is_dual:
            _, _, fc_out = self.fc(fc_input)
        else:
            fc_out = self.fc(fc_input)
        
        # Apply readout layer with weights
        readout_input = torch.matmul(fc_out, self.readout_weights.t())
        
        # Process through readout layer
        if self.is_dual:
            _, _, spikes = self.readout(readout_input)
        else:
            spikes = self.readout(readout_input)
        
        return spikes
    
    def reset_state(self):
        """Reset all neuron states in the network"""
        self.conv0.reset_state()
        self.conv1.reset_state()
        self.conv2.reset_state()
        self.fc.reset_state()
        self.readout.reset_state()
        
    def enable_learning(self, enable=True):
        """Enable or disable learning in applicable layers
        
        Args:
            enable (bool): Whether to enable learning
        """
        # For LT-Gate, enable Hebbian learning in conv layers
        if self.is_dual and hasattr(self.conv0, 'enable_hebbian_learning'):
            self.conv0.enable_hebbian_learning(enable)
            self.conv1.enable_hebbian_learning(enable)
            self.conv2.enable_hebbian_learning(enable)

def build_backbone(cfg):
    """Build a neural network backbone based on config"""
    return SNNBackbone(cfg)

def compile_for_loihi(net, cfg):
    """Compile the network for Loihi-2 neuromorphic hardware
    
    Args:
        net (SNNBackbone): Network to compile
        cfg (dict): Configuration dictionary with parameters
        
    Returns:
        object: Compiled Loihi model or None if compilation failed
    """
    if not HAS_LAVA:
        print("Warning: Lava-DL not found. Cannot compile for Loihi.")
        return None
    
    # Compile the model for Loihi-2
    try:
        # Create a batch-size neutral input shape for compilation
        input_shape = (1, 1, 28, 28)  # [batch=1, channels=1, height=28, width=28]
        if isinstance(cfg, dict) and 'batch_size' in cfg:
            input_shape = (cfg['batch_size'], 1, 28, 28)
            
        loihi_net = slayer2loihi.loihi_model(net, input_shape=input_shape)
        print("Successfully compiled model for Loihi-2.")
        return loihi_net
    except Exception as e:
        print(f"Error compiling for Loihi-2: {e}")
        return None

def process_sequence(net, sequence, readout_window=20):
    """Process a complete input sequence through the network
    
    Args:
        net (SNNBackbone): Neural network to process sequence
        sequence (torch.Tensor): Input sequence of shape [batch_size, time_steps, channels, height, width]
        readout_window (int): Number of time steps at the end to count spikes for classification
        
    Returns:
        tuple: (class_decisions, spike_counts) where decisions has shape [batch_size] 
               and spike_counts has shape [batch_size, num_classes]
    """
    batch_size = sequence.shape[0]
    seq_length = sequence.shape[1]
    num_classes = 10  # Assuming 10 classes (digits 0-9)
    
    # Make sure the network state is reset before processing the sequence
    net.reset_state()
    
    # Initialize spike counters for readout
    spike_counts = torch.zeros(batch_size, num_classes, device=sequence.device)
    
    # Process each timestep
    for t in range(seq_length):
        # Forward pass for current timestep - sequence shape is [B, T, C, H, W]
        frame = sequence[:, t]  # Extract current time slice [batch, channels, height, width]
        spikes = net(frame)  # Forward pass produces [batch, num_classes]
        
        # Count spikes in readout window (last N ms)
        if t >= seq_length - readout_window:
            spike_counts += spikes
    
    # Decision based on maximum spike count
    decisions = torch.argmax(spike_counts, dim=1)
    
    return decisions, spike_counts

def load_backbone_from_config(config_path):
    """Load a model from a YAML configuration file
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        SNNBackbone: Instantiated model
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model from config
    model = build_backbone(config)
    
    return model

def reset_network_state(net):
    """Reset all state variables (membrane potentials, spike counters) in the network"""
    for module in net.modules():
        if hasattr(module, 'reset_state'):
            module.reset_state()
