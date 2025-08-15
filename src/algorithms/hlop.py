import torch  # type: ignore
from ..layers.single_lif import SingleConv

class HLOPTrainer:
    """
    HLOP trainer that implements Hebbian learning for main weights
    while subspace projections are handled in HLOPConv layers.
    """
    def __init__(self, net, cfg):
        self.net = net
        self.eta_w = cfg['eta_w']

    def to(self, device):
        """Move trainer to device (for compatibility with LTGateTrainer)"""
        self.net = self.net.to(device)
        return self

    def step(self, seq):
        for t in range(seq.size(0)):
            spikes = self.net(seq[t])
            # local Hebbian update main weights (skip subspace W)
            with torch.no_grad():
                for m in self.net.modules():
                    if isinstance(m, SingleConv):
                        Δ = self.eta_w * torch.einsum('bo,bi->oi',
                                                     spikes.float(),
                                                     seq[t].flatten(1))
                        m.weight.data += Δ.view_as(m.weight)
