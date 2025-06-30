import torch
import torch.nn as nn
import numpy as np
import logging
from collections import defaultdict

from src.layers.lt_conv import LTConv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='ltgate_diagnostics.log',
    filemode='a'
)
logger = logging.getLogger('ltgate_diagnostics')

class LTGateTrainer:
    """
    LT-Gate algorithm orchestrator that coordinates learning across all LT-Gate layers.
    
    This class handles the training process for the LT-Gate algorithm, maintaining
    presynaptic traces and applying local weight updates to all LTConv layers in the model.
    """
    def __init__(self, model, cfg):
        """
        Initialize the LT-Gate trainer.
        
        Args:
            model (nn.Module): The neural network model containing LTConv layers
            cfg (dict): Configuration dictionary with hyperparameters
        """
        self.model = model
        self.eta = cfg.get('eta', 0.01)  # Default learning rate if not specified
        
        # Gamma strategy configuration
        self.gamma_strategy = cfg.get('gamma_strategy', 'var')  # 'var' or 'fixed'
        self.gamma_fixed_value = cfg.get('gamma_fixed_value', 0.5)  # Used if strategy is 'fixed'
        
        # Edge-case diagnostics configuration
        self.max_weight = cfg.get('max_weight', 5.0)  # Maximum allowed weight value (Loihi constraint)
        self.min_weight = cfg.get('min_weight', -5.0)  # Minimum allowed weight value (Loihi constraint)
        self.gamma_stuck_threshold = cfg.get('gamma_stuck_threshold', 0.95)  # Threshold to consider γ stuck
        self.tracking_interval = cfg.get('tracking_interval', 100)  # Steps between diagnostics checks
        self.enable_diagnostics = cfg.get('enable_diagnostics', True)  # Enable edge-case diagnostics
        self.enable_weight_clipping = cfg.get('enable_weight_clipping', True)  # Enable weight clipping
        self.weight_monitoring = cfg.get('weight_monitoring', True)  # Enable weight monitoring
        self.loihi_constraints = cfg.get('loihi_constraints', True)  # Check Loihi hardware constraints
        
        # Initialize diagnostic tracking structures
        self.step_count = 0
        self.gamma_history = {}  # Track γ values over time to detect stuck gates
        self.weight_stats = defaultdict(list)  # Track weight statistics
        self.weight_violations = defaultdict(int)  # Count weight constraint violations
        
        # Initialize presynaptic traces for each layer
        self.traces = {}
        self.ltconv_layers = {}
        
        for name, m in model.named_modules():
            if isinstance(m, LTConv):
                # Initialize trace with appropriate shape matching the input to this layer
                self.traces[name] = torch.zeros_like(m.conv_fast.weight[0])
                
                # Set the learning rate for this layer
                m.eta_w = self.eta
                
                # Store reference to LTConv layers for diagnostics
                self.ltconv_layers[name] = m
                
                # Configure gamma strategy for this layer
                if self.gamma_strategy == 'fixed':
                    m.gate.force_gamma(self.gamma_fixed_value)
                
                # Initialize gamma history for this layer
                self.gamma_history[name] = torch.zeros_like(m.gate.var_f)
        
        # Log diagnostic setup
        if self.enable_diagnostics:
            logger.info("Initialized LT-Gate diagnostics with settings:")
            logger.info(f"  - Weight clipping: {self.enable_weight_clipping} (min={self.min_weight}, max={self.max_weight})")
            logger.info(f"  - Gamma strategy: {self.gamma_strategy}")
            if self.gamma_strategy == 'fixed':
                logger.info(f"  - Fixed gamma value: {self.gamma_fixed_value}")
            logger.info(f"  - Gamma stuck threshold: {self.gamma_stuck_threshold}")
            logger.info(f"  - Tracking interval: {self.tracking_interval} steps")
            logger.info(f"  - Loihi constraints checking: {self.loihi_constraints}")
        
        # Other training parameters
        self.pre_decay = cfg.get('pre_decay', 0.8)  # Decay factor for presynaptic traces
    
    def step(self, seq):
        """
        Process a sequence batch and apply LT-Gate updates.
        
        Args:
            seq (torch.Tensor): Input sequence of shape [time_steps, batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Output spikes from the final time step
        """
        device = seq.device
        time_steps, batch_size = seq.shape[0], seq.shape[1]
        
        # Reset model state at the beginning of each sequence
        self._reset_state()
        
        # Track final output for return
        final_output = None
        
        # Process each time step in the sequence
        for t in range(time_steps):
            x_t = seq[t]  # Current input frame [batch_size, channels, height, width]
            
            # Update presynaptic traces before forward pass
            self._update_traces(x_t)
            
            # Forward pass through the model
            output = self.model(x_t)  # [batch_size, out_channels, height, width]
            final_output = output
            
            # Apply local plasticity updates to LT-Gate layers
            self._apply_lt_gate_updates()
        
        # Increment step counter for diagnostics
        self.step_count += 1
        
        # Run edge-case diagnostics at specified intervals
        if self.enable_diagnostics and self.step_count % self.tracking_interval == 0:
            self._run_diagnostics()
        
        return final_output
    
    def _run_diagnostics(self):
        """
        Run diagnostics to detect edge cases in LT-Gate training.
        
        This method checks for:
        1. Stuck gating variables (γ)
        2. Exploding or vanishing weights
        3. Loihi hardware mapping constraint violations
        """
        # Check all LTConv layers
        for name, layer in self.ltconv_layers.items():
            # 1. Gamma (gating variable) diagnostics
            self._check_gamma_status(name, layer)
            
            # 2. Weight statistics and constraints
            self._check_weight_constraints(name, layer)
            
            # 3. Loihi mapping constraints
            if self.loihi_constraints:
                self._check_loihi_constraints(name, layer)
                
        # Log summary for this step
        logger.info(f"Step {self.step_count} - Diagnostic summary:")
        for name in self.ltconv_layers.keys():
            if name in self.weight_violations and self.weight_violations[name] > 0:
                logger.info(f"  Layer {name}: {self.weight_violations[name]} weight constraint violations")
    
    def _check_gamma_status(self, name, layer):
        """
        Check for stuck gamma values in the gates.
        
        Args:
            name (str): Layer name
            layer (LTConv): LTConv layer to check
        """
        # Get current gamma values
        current_gamma = layer.gate.selector()
        
        # Calculate moving average to detect stuck values
        avg_gamma = 0.9 * self.gamma_history[name] + 0.1 * current_gamma
        self.gamma_history[name] = avg_gamma
        
        # Check for stuck values (near 0 or 1 for extended periods)
        stuck_high = (avg_gamma > self.gamma_stuck_threshold).float().mean().item()
        stuck_low = (avg_gamma < (1 - self.gamma_stuck_threshold)).float().mean().item()
        
        # Log warning if high percentage of gates are stuck
        if stuck_high > 0.8 or stuck_low > 0.8:
            dominant_branch = "fast" if stuck_high > stuck_low else "slow"
            logger.warning(f"Layer {name}: {max(stuck_high, stuck_low)*100:.1f}% of gates stuck at {dominant_branch} branch")
            logger.info(f"  - Gamma stats: mean={avg_gamma.mean().item():.3f}, min={avg_gamma.min().item():.3f}, max={avg_gamma.max().item():.3f}")
    
    def _check_weight_constraints(self, name, layer):
        """
        Check weight constraints and apply clipping if needed.
        
        Args:
            name (str): Layer name
            layer (LTConv): LTConv layer to check
        """
        # Check fast pathway weights
        fast_w = layer.conv_fast.weight.data
        
        # Record statistics
        if self.weight_monitoring:
            self.weight_stats[f"{name}_fast"].append({
                'step': self.step_count,
                'mean': fast_w.mean().item(),
                'std': fast_w.std().item(),
                'min': fast_w.min().item(),
                'max': fast_w.max().item()
            })
        
        # Count violations
        violations_fast = ((fast_w > self.max_weight) | (fast_w < self.min_weight)).sum().item()
        if violations_fast > 0:
            self.weight_violations[name] += violations_fast
            logger.warning(f"Layer {name} fast path: {violations_fast} weights outside [{self.min_weight}, {self.max_weight}] range")
        
        # Apply clipping if enabled
        if self.enable_weight_clipping:
            fast_w.clamp_(self.min_weight, self.max_weight)
        
        # Check slow pathway weights
        slow_w = layer.conv_slow.weight.data
        
        # Record statistics
        if self.weight_monitoring:
            self.weight_stats[f"{name}_slow"].append({
                'step': self.step_count,
                'mean': slow_w.mean().item(),
                'std': slow_w.std().item(),
                'min': slow_w.min().item(),
                'max': slow_w.max().item()
            })
        
        # Count violations
        violations_slow = ((slow_w > self.max_weight) | (slow_w < self.min_weight)).sum().item()
        if violations_slow > 0:
            self.weight_violations[name] += violations_slow
            logger.warning(f"Layer {name} slow path: {violations_slow} weights outside [{self.min_weight}, {self.max_weight}] range")
        
        # Apply clipping if enabled
        if self.enable_weight_clipping:
            slow_w.clamp_(self.min_weight, self.max_weight)
    
    def _check_loihi_constraints(self, name, layer):
        """
        Check Loihi hardware mapping constraints.
        
        Loihi-2 has specific constraints on weights and neuron parameters.
        
        Args:
            name (str): Layer name
            layer (LTConv): LTConv layer to check
        """
        # Check weight precision (Loihi uses 8-bit weights)
        precision_loss = False
        
        # Check if weights would lose significant precision when quantized to 8-bit
        fast_w = layer.conv_fast.weight.data
        slow_w = layer.conv_slow.weight.data
        
        # Quantize to simulate Loihi precision constraints
        weight_scale = max(abs(fast_w.min().item()), abs(fast_w.max().item()), 
                          abs(slow_w.min().item()), abs(slow_w.max().item()))
        
        if weight_scale > 0:
            # Calculate 8-bit quantization factor
            quant_factor = 127.0 / weight_scale
            
            # Quantize and dequantize to simulate precision loss
            fast_w_quant = torch.round(fast_w * quant_factor) / quant_factor
            slow_w_quant = torch.round(slow_w * quant_factor) / quant_factor
            
            # Calculate quantization error
            fast_error = (fast_w - fast_w_quant).abs().mean().item()
            slow_error = (slow_w - slow_w_quant).abs().mean().item()
            
            # Log warning if quantization error is significant
            if fast_error > 0.01 or slow_error > 0.01:
                logger.warning(f"Layer {name} has significant quantization error: fast={fast_error:.4f}, slow={slow_error:.4f}")
                precision_loss = True
        
        # Check neuron dynamics constraints
        # Loihi has specific constraints on time constants and thresholds
        if layer.cell.tau_fast < 1e-3 or layer.cell.tau_slow < 1e-3:
            logger.warning(f"Layer {name} has time constants below Loihi minimum (1ms)")
        
        # Threshold should be within reasonable range for Loihi
        if layer.cell.threshold > 1.0 or layer.cell.threshold < 0.01:
            logger.warning(f"Layer {name} threshold {layer.cell.threshold:.4f} may not be suitable for Loihi")
        
        # Log summary for this layer
        if precision_loss:
            logger.info(f"Layer {name} may encounter precision loss when mapped to Loihi hardware")

    
    def _update_traces(self, x_t):
        """
        Update presynaptic traces for all layers based on the current input.
        
        Args:
            x_t (torch.Tensor): Current input frame [batch_size, channels, height, width]
        """
        pre_flat = x_t.flatten(start_dim=2)  # [batch_size, channels, flattened_spatial]
        
        # For input layer, the trace is directly from the input
        self.current_input = pre_flat
        
        # Decay existing traces and add new activity
        for name in self.traces.keys():
            # Decay existing trace
            self.traces[name] = self.traces[name] * self.pre_decay
            
            # For the input layer, update with current input
            # Note: for deeper layers, this is more complex and requires tracking
            # activity through the network. For simplicity, we're just using the
            # input for now, but in a full implementation we would track per-layer.
            if name.endswith('conv1'):
                # Assuming first conv layer gets direct input
                batch_avg = pre_flat.mean(0)  # Average across batch
                self.traces[name] = self.traces[name] + batch_avg.flatten()
    
    def _apply_lt_gate_updates(self):
        """
        Apply LT-Gate updates to all LTConv layers in the model.
        """
        for name, m in self.model.named_modules():
            if isinstance(m, LTConv):
                # Get presynaptic trace for this layer
                pre_trace = self.current_input  # Simplified: using input for all layers
                
                # Get the binary spike outputs from fast and slow neurons
                # These are stored in the LTConv's cell during the forward pass
                post_spikes_f = m.cell.u_f >= m.cell.threshold  # Fast neuron spikes
                post_spikes_s = m.cell.u_s >= m.cell.threshold  # Slow neuron spikes
                
                # Apply local update using LT-Gate rule
                m.local_update(pre_trace, post_spikes_f, post_spikes_s)
    
    def _reset_state(self):
        """
        Reset the state of all neurons in the model and clear traces.
        """
        # Reset neuron states in all layers
        for _, m in self.model.named_modules():
            if hasattr(m, 'reset_state'):
                m.reset_state()
        
        # Reset traces to zero
        for name in self.traces.keys():
            self.traces[name].zero_()
