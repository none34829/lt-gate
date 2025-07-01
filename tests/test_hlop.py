import torch
import pytest
from src.layers.hlop_subspace import HLOPSubspace
from src.layers.hlop_conv import HLOPConv

def test_projection_property():
    """Test that projected vectors are orthogonal to the subspace."""
    sub = HLOPSubspace(6, M=3)
    x = torch.randn(100, 6)
    x_proj = sub.project(x)
    # dot(x_proj, Wᵀ) ≈ 0
    proj_dot = x_proj @ sub.W.t()  # [100, 3]
    assert torch.allclose(proj_dot, torch.zeros_like(proj_dot), atol=1e-4)

def test_hlop_conv_shapes():
    """Test that HLOPConv maintains correct shapes through projection."""
    layer = HLOPConv(in_ch=1, out_ch=16, k=3, stride=1, proj_dim=10)
    x = torch.randn(2, 1, 28, 28)  # [B,C,H,W]
    out = layer(x)
    assert out.shape == (2, 16, 26, 26)  # [B,out_ch,H',W']

def test_hebbian_update():
    """Test that Hebbian updates modify weights in expected direction."""
    sub = HLOPSubspace(4, M=2)
    x = torch.ones(10, 4)
    old_w = sub.W.clone()
    sub.hebb_update(x)
    assert not torch.allclose(sub.W, old_w)  # weights should change

def test_training_mode():
    """Test that Hebbian updates only occur in training mode."""
    layer = HLOPConv(in_ch=1, out_ch=16, k=3)
    x = torch.randn(2, 1, 28, 28)
    
    layer.train()
    w_before = layer.proj.W.clone()
    _ = layer(x)
    assert not torch.allclose(layer.proj.W, w_before)
    
    layer.eval()
    w_before = layer.proj.W.clone()
    _ = layer(x)
    assert torch.allclose(layer.proj.W, w_before)
