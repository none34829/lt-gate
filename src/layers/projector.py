import torch  # type: ignore

class OneVectorProjector(torch.nn.Module):
    """
    Fast implementation of a one-vector orthogonal projector for local weight updates.
    
    This module implements the projection operator P = vv^T / ||v||^2 and provides
    functionality to orthogonalize raw weight updates using I - P (the orthogonal
    complement of the span of v).
    
    The projector also implements Oja's rule to update the principal vector v
    based on Hebbian updates.
    """
    def __init__(self, out_feats, eta_v=1e-3):
        """
        Initialize the projector.
        
        Args:
            out_feats (int): Number of output features/neurons
            eta_v (float): Learning rate for updating the principal vector v
        """
        super().__init__()
        # Initialize principal vector with small random values
        self.v = torch.nn.Parameter(torch.randn(out_feats) * 1e-3,
                                   requires_grad=False)
        self.eta_v = eta_v
    
    def orthogonalize(self, raw_update):
        """
        Project raw weight update to be orthogonal to v.
        
        Computes (I - P) * raw_update, where P = vv^T / ||v||^2
        
        Args:
            raw_update (torch.Tensor): Raw weight update of shape [out_feats, in_feats]
            
        Returns:
            torch.Tensor: Orthogonalized update of same shape
        """
        # Compute projection matrix P = vv^T / ||v||^2
        proj = torch.ger(self.v, self.v) / (self.v.norm() ** 2 + 1e-8)
        # Apply (I - P) * raw_update
        return raw_update - proj @ raw_update
    
    def update_v(self, post_spike, pre_trace):
        """
        Update principal vector v using Hebbian update with Oja's rule.
        
        v ← v + η_v * [(post_spike * pre_trace) - ((post_spike * pre_trace)^T * v) * v]
        
        Args:
            post_spike (torch.Tensor): Post-synaptic spikes [out_feats]
            pre_trace (torch.Tensor): Pre-synaptic trace [in_feats]
        """
        # Compute Hebbian update (outer product of post-synaptic and pre-synaptic activity)
        hebb = torch.einsum('o,i->oi', post_spike, pre_trace).mean(dim=1)
        
        # Oja's correction term to maintain vector norm
        correction = torch.dot(hebb, self.v) * self.v
        
        # Update principal vector v
        self.v.data += self.eta_v * (hebb - correction)
