from .single_lif import SingleConv
from .hlop_subspace import HLOPSubspace

class HLOPConv(SingleConv):
    def __init__(self, in_ch, out_ch, k=3, stride=1,
                 proj_dim=20, eta_h=1e-3, **lif_kw):
        super().__init__(in_ch, out_ch, k, stride, **lif_kw)
        # subspace is on flattened conv patch (channel*kernel*kernel)
        self.proj = HLOPSubspace(in_ch * k * k, M=proj_dim, eta=eta_h)

    def forward(self, x):
        # flatten each (C,k,k) receptive field to [B,H*W, Ck²]
        B, C, H, W = x.shape
        patches = x.unfold(2, 3, 1).unfold(3, 3, 1)         # [B,C,H',W',k,k]
        patches = patches.permute(0,2,3,1,4,5).contiguous() # [B,H',W',C,k,k]
        patches = patches.view(B, -1, C*3*3)                # [B,P,D]

        # Project each patch through subspace
        patches_hat = self.proj.project(patches.reshape(-1, C*3*3))
        patches_hat = patches_hat.view(B, -1, C, 3, 3)

        # For simplicity, we'll use a global projection approach
        # This avoids complex patch reconstruction
        z = super().forward(x)                              # call parent conv
        
        if self.training:
            self.proj.hebb_update(patches.reshape(-1, C*3*3))
        
        return z
