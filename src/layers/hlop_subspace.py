import torch  # type: ignore
import torch.nn as nn  # type: ignore

class HLOPSubspace(nn.Module):
    """
    M auxiliary neurons learn the K principal directions of presynaptic activity
    via Oja's rule. Projection is x̂ = x - Wᵀ(Wx)
    """
    def __init__(self, in_feats: int, M: int = 20, eta: float = 1e-3):
        super().__init__()
        self.M = M
        self.eta = eta
        # W shape [M, D]. requires_grad=False to prevent autograd tracking
        self.W = nn.Parameter(0.01 * torch.randn(M, in_feats), requires_grad=False)

    @torch.no_grad()
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape [B, D]; returns x̂ (orthogonal component).
        Project x onto the orthogonal complement of the subspace spanned by W.
        Uses QR decomposition for robust orthogonalization.
        """
        # Get orthonormal basis for W using QR decomposition
        Q, _ = torch.linalg.qr(self.W.t())  # [D, M]
        
        # Project x onto orthogonal complement of W's column space
        # x - QQᵀx gives the orthogonal component
        x_proj = x - (x @ Q) @ Q.t()
        
        return x_proj

    @torch.no_grad()
    def hebb_update(self, x: torch.Tensor):
        """
        Oja rule ΔW = η (y xᵀ – yyᵀ W).
        Use batch mean to reduce variance.
        """
        y = torch.matmul(x, self.W.t())                    # [B,M]
        yx = torch.einsum('bm,bd->md', y, x) / x.size(0)   # [M,D]
        yy = torch.einsum('bm,bn->mn', y, y) / x.size(0)   # [M,M]
        self.W += self.eta * (yx - yy @ self.W)
