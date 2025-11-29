import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from layer_norm import LayerNorm

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, p_dropout, d_type, device):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, p_dropout, d_type, device)
        self.layernorm_1 = LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(p=p_dropout)
        )
        self.layernorm_2 = LayerNorm(d_model)

    #could be residual or initial token embedding + positional encoding
    def forward(self, x, pad_mask):
        self_attention = self.attention(q=x, k=x, v=x, padding_mask = pad_mask)
        normalized_attn = self.layernorm_1(x + self_attention)
        feedforward = self.ffn(normalized_attn)
        output = self.layernorm_2(normalized_attn + feedforward)
        return output
