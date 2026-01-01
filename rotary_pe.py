import torch.nn as nn
import torch

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        # RoPE is applied per-head, so use head_dim not full dim
        idx = torch.arange(0, head_dim // 2)
        thetas = torch.pow(10000, -2 * (idx) / head_dim)
        # essentially every dimension pair gets its own micro-rotationary value (base rotation factor)
        self.register_buffer("thetas", thetas)

    #(batch, n_heads, seq_len, head_dim) - (applied after reshaping to separate heads)
    #rope decays over distance, the inner product gets smaller over a longer distance,
    def forward(self, x):
        batch, n_heads, seq_len, head_dim = x.shape
        positions = torch.arange(0, seq_len, device=x.device, dtype=x.dtype) #(seq_len,)
        m_theta = positions.unsqueeze(-1) * self.thetas #(seq_len,1) x (head_dim/2,) -> (seq_len, head_dim/2)
        cos, sin = torch.cos(m_theta), torch.sin(m_theta) # (seq_len, head_dim/2)

        # x has shape (batch, n_heads, seq_len, head_dim)
        x_dim = (x[..., 0::2] * cos) - (x[..., 1::2] * sin) # (batch, n_heads, seq_len, head_dim/2)
        y_dim = (x[..., 1::2] * cos) + (x[..., 0::2] * sin) # (batch, n_heads, seq_len, head_dim/2)

        #inplace ops messes up autograd (cant track properly), gradients are needed as rope flows through other layers even though it doesnt have params itself
        #torch stack inserts a dimension, dim=-1 creates new axis at last position
        #after stacking, becomes (batch, n_heads, seq_len, head_dim/2, 2), then flatten to (batch, n_heads, seq_len, head_dim)
        return torch.stack((x_dim, y_dim), dim=-1).flatten(-2)