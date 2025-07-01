import torch  # type: ignore

class VarianceGate(torch.nn.Module):
    """
    Tracks variance in both fast and slow neuron outputs and selects the quieter branch.
    
    The variance gate uses an exponential moving average to track the variance of each
    neuron's activity in both fast and slow compartments. It then generates a selector
    mask indicating which branch (fast or slow) should be preferred for each neuron.
    
    γ_i = 1 if fast branch is selected (fast has lower variance)
    γ_i = 0 if slow branch is selected (slow has lower variance)
    """
    def __init__(self, out_feats, lam=1e-3, eps=1e-8):
        """
        Initialize the variance gate.
        
        Args:
            out_feats (int): Number of output features/neurons to track
            lam (float): Decay factor for exponential moving average
            eps (float): Small constant for numerical stability
        """
        super().__init__()
        # Initialize variance trackers with small non-zero values
        self.register_buffer('var_f', torch.ones(out_feats) * 1e-4)
        self.register_buffer('var_s', torch.ones(out_feats) * 1e-4)
        self.lam = lam
        self.eps = eps
        self._forced_gamma = None
    
    def update(self, s_f, s_s):
        """
        Update variance estimates based on current spikes.
        
        Args:
            s_f (torch.Tensor): Fast neuron spikes [batch_size, out_feats]
            s_s (torch.Tensor): Slow neuron spikes [batch_size, out_feats]
        """
        # Convert binary spikes to float and calculate variance across batch
        dv_f = ((s_f.float() - s_f.float().mean(0)) ** 2).mean(0)
        dv_s = ((s_s.float() - s_s.float().mean(0)) ** 2).mean(0)
        
        # Update exponential moving average of variance
        self.var_f.mul_(1 - self.lam).add_(self.lam * dv_f)
        self.var_s.mul_(1 - self.lam).add_(self.lam * dv_s)
    
    def force_gamma(self, val: float) -> None:
        """Force a fixed gamma value for ablation studies.
        
        Args:
            val: Fixed gamma value between 0 and 1
        """
        assert 0 <= val <= 1, "Gamma must be between 0 and 1"
        self._forced_gamma = val
    
    def selector(self) -> torch.Tensor:
        """Generate selection mask based on variance comparison.
        
        Returns:
            torch.Tensor: Binary mask γ of shape [out_feats], where
                          γ_i = 1 if var_f <= var_s (select fast branch)
                          γ_i = 0 if var_f > var_s (select slow branch)
        """
        if self._forced_gamma is not None:
            return torch.full_like(self.var_f, self._forced_gamma)
        return (self.var_f <= self.var_s).float()
