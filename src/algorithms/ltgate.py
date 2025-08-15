import torch  # type: ignore
import torch.nn as nn  # type: ignore
import numpy as np
import logging
from collections import defaultdict

from ..layers.lt_conv import LTConv
from ..layers.dual_conv_lif import DualConvLIF

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
            if isinstance(m, (LTConv, DualConvLIF)):
                # Initialize trace with appropriate shape for input to this layer
                # For conv layers, we need to track input activity
                if hasattr(m, 'conv_fast'):
                    # Initialize trace with the same shape as a single input to this conv layer
                    # This will be [in_channels, kernel_size, kernel_size]
                    self.traces[name] = torch.zeros_like(m.conv_fast.weight[0])
                else:
                    # Fallback for other layer types
                    self.traces[name] = torch.zeros(1)
                
                # Set the learning rate for this layer
                m.eta_w = self.eta
                
                # Store reference to LTConv layers for diagnostics
                self.ltconv_layers[name] = m
                
                # Configure gamma strategy for this layer (only for LTConv)
                if isinstance(m, LTConv) and self.gamma_strategy == 'fixed':
                    m.gate.force_gamma(self.gamma_fixed_value)
                
                # Initialize gamma history for this layer (only for LTConv)
                if isinstance(m, LTConv):
                    self.gamma_history[name] = torch.zeros_like(m.gate.var_f)
                else:
                    # For DualConvLIF, we don't track gamma history
                    self.gamma_history[name] = torch.zeros(1)  # Dummy tensor
        
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
    
    def to(self, device):
        """Move trainer traces to the specified device"""
        for k in self.traces:
            self.traces[k] = self.traces[k].to(device)
        for k in self.gamma_history:
            self.gamma_history[k] = self.gamma_history[k].to(device)
        return self
    
    @torch.no_grad()
    def step(self, seq, targets=None):
        """
        Process a sequence batch and apply LT-Gate updates.
        
        Args:
            seq (torch.Tensor): Input sequence of shape [time_steps, batch_size, channels, height, width]
            targets (torch.Tensor, optional): Target labels for supervised learning
            
        Returns:
            tuple: (final scores, average loss)
        """
        # [B,T,...] -> [T,B,...] if needed
        if seq.size(0) != 200:
            seq = seq.transpose(0, 1)
        T, B = seq.shape[0], seq.shape[1]

        self._reset_state()

        final_scores = None  # accumulate linear scores over time
        feat_sum = None      # accumulate fc features over time
        total_loss = 0.0

        for t in range(T):
            x_t = seq[t]
            self._update_traces(x_t)

            # ask model for spikes/scores/features
            spikes_t, scores_t, fc_out_t = self.model(x_t, return_features=True)

            final_scores = scores_t if final_scores is None else (final_scores + scores_t)
            feat_sum     = fc_out_t if feat_sum is None else (feat_sum + fc_out_t)

            total_loss += self._apply_lt_gate_updates()

        # --- supervised readout update (safe one-hot + delta rule) ---
        if targets is not None:
            # ensure proper dtype/device/shape
            targets = targets.detach().to(final_scores.device).to(torch.long).view(-1)
            num_classes = final_scores.size(-1)

            # guard: invalid labels → skip update instead of crashing
            if targets.numel() == 0 or targets.min().item() < 0 or targets.max().item() >= num_classes:
                print(f"[LTGateTrainer] Bad targets: dtype={targets.dtype}, "
                      f"min={targets.min().item() if targets.numel() else 'NA'}, "
                      f"max={targets.max().item() if targets.numel() else 'NA'}, "
                      f"num_classes={num_classes}. Skipping supervised update.")
            else:
                y = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()

                # Use the actual pre-readout features (fc_out accumulated over time)
                # feat_sum is already the accumulated fc_out from the time loop
                W = self.model.readout_weights  # [10, 256]

                # Simple delta rule update using the accumulated features
                pred = torch.softmax(final_scores, dim=-1)  # [B, 10]
                err  = (pred - y)                           # [B, 10]

                lr_ro = 1e-3
                lam   = 1e-4
                # Update: W += lr * error * features
                gradW = err.T @ feat_sum / targets.size(0) + lam * W  # [10, 256]
                with torch.no_grad():
                    W -= lr_ro * gradW

                # optional scalar loss for logging
                total_loss += torch.nn.functional.cross_entropy(final_scores, targets).item()

        # diagnostics cadence
        self.step_count += 1
        if self.enable_diagnostics and self.step_count % self.tracking_interval == 0:
            self._run_diagnostics()

        avg_loss = total_loss / max(T, 1)
        return final_scores, avg_loss
    
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
        
        # Log spike statistics every 10 steps
        if self.step_count % 10 == 0:
            self._log_spike_statistics()
    
    def _check_gamma_status(self, name, layer):
        """
        Check for stuck gamma values in the gates.
        
        Args:
            name (str): Layer name
            layer (LTConv or DualConvLIF): Layer to check
        """
        # Only check gamma for LTConv layers
        if not isinstance(layer, LTConv):
            return
            
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
            layer (LTConv or DualConvLIF): Layer to check
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
        
        # Check neuron dynamics constraints (only for LTConv)
        if isinstance(layer, LTConv):
            # Loihi has specific constraints on time constants and thresholds
            if layer.cell.tau_fast < 1e-3 or layer.cell.tau_slow < 1e-3:
                logger.warning(f"Layer {name} has time constants below Loihi minimum (1ms)")
            
            # Threshold should be within reasonable range for Loihi
            if layer.cell.threshold > 1.0 or layer.cell.threshold < 0.01:
                logger.warning(f"Layer {name} threshold {layer.cell.threshold:.4f} may not be suitable for Loihi")
        
        # Log summary for this layer
        if precision_loss:
            logger.info(f"Layer {name} may encounter precision loss when mapped to Loihi hardware")

    def _log_spike_statistics(self):
        """
        Log spike statistics for all layers to monitor learning progress.
        """
        logger.info("=== Spike Statistics ===")
        for name, layer in self.ltconv_layers.items():
            if hasattr(layer, 'last_spikes_fast') and hasattr(layer, 'last_spikes_slow'):
                # Calculate spike rates
                s_f_rate = layer.last_spikes_fast.float().mean().item()
                s_s_rate = layer.last_spikes_slow.float().mean().item()
                
                # Calculate weight norms
                if hasattr(layer, 'conv_fast') and hasattr(layer, 'conv_slow'):
                    w_f_norm = layer.conv_fast.weight.norm().item()
                    w_s_norm = layer.conv_slow.weight.norm().item()
                    logger.info(f"  {name}: fast_rate={s_f_rate:.4f}, slow_rate={s_s_rate:.4f}, |W_f|={w_f_norm:.4f}, |W_s|={w_s_norm:.4f}")
                else:
                    logger.info(f"  {name}: fast_rate={s_f_rate:.4f}, slow_rate={s_s_rate:.4f}")
            else:
                logger.info(f"  {name}: no spike data available")
        logger.info("======================")

    
    def _update_traces(self, x_t):
        """
        Update presynaptic traces for all layers based on the current input.
        
        Args:
            x_t (torch.Tensor): Current input frame [batch_size, channels, height, width]
        """
        # Store current input for potential use
        self.current_input = x_t
        
        # Decay existing traces and add new activity
        for name in self.traces.keys():
            # Decay existing trace
            self.traces[name] = self.traces[name] * self.pre_decay
            
            # For conv layers, we'll use a simplified approach
            # In a full implementation, we would track activity through the network
            if name.startswith('conv'):  # All conv layers
                # Average across batch and spatial dimensions to get per-channel activity
                # Shape: [batch_size, channels, height, width] -> [channels]
                channel_activity = x_t.mean(dim=(0, 2, 3))  # Average over batch, height, width
                
                # Reshape to match the trace shape [in_channels, kernel_size, kernel_size]
                trace_shape = self.traces[name].shape
                if len(trace_shape) == 3:  # [in_channels, kernel_size, kernel_size]
                    # Expand channel activity to match kernel size
                    expanded_activity = channel_activity.unsqueeze(1).unsqueeze(2).expand_as(self.traces[name])
                    self.traces[name] = self.traces[name] + expanded_activity
                else:
                    # Fallback for unexpected shapes
                    self.traces[name] = self.traces[name] + channel_activity.mean()

    def _apply_lt_gate_updates(self):
        """
        Apply local plasticity updates to all LT-Gate layers.
        
        Returns:
            float: Total reconstruction loss for this timestep
        """
        total_loss = 0.0
        mse = nn.MSELoss(reduction='mean')
        
        # Update all LT-Gate layers
        for name, layer in self.ltconv_layers.items():
            # Get current traces for this layer
            traces = self.traces[name]
            
            # Get stored spikes from the layer (stored during forward pass)
            if hasattr(layer, 'last_spikes_fast') and hasattr(layer, 'last_spikes_slow'):
                s_f = layer.last_spikes_fast  # [B, C*H*W]
                s_s = layer.last_spikes_slow  # [B, C*H*W]
                
                # Apply local plasticity update with real spikes
                recon_loss = layer.local_update(traces, s_f, s_s)
                total_loss += recon_loss
            else:
                # Fallback for layers without spike storage
                logger.warning(f"Layer {name} missing spike storage - skipping update")
                continue
        
        return total_loss



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
