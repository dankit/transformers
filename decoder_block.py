import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from layer_norm import LayerNorm

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, p_dropout, d_type, device):
        super().__init__()
        #self attention and cross attention learn different weights
        self.self_attention = MultiHeadAttention(d_model, num_heads, p_dropout, d_type, device, mask=True)
        self.layernorm_1 = LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, p_dropout, d_type, device, mask=False)
        self.layernorm_2 = LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(p=p_dropout)
        )
        self.layernorm_3 = LayerNorm(d_model)

    def forward(self, decoder_input, trg_pad_mask, encoder_input=None, src_pad_mask=None):
        #input = batch x seq_len x dim
        masked_self_attention = self.self_attention(q=decoder_input, k=decoder_input, v=decoder_input, padding_mask = trg_pad_mask)
        normalized_attn = self.layernorm_1(decoder_input + masked_self_attention)

        #can be none for decoder-only transformer
        if encoder_input is not None:
            cross_attention = self.cross_attention(q=normalized_attn, k=encoder_input, v=encoder_input, padding_mask = src_pad_mask)
            normalized_attn = self.layernorm_2(normalized_attn + cross_attention)

        feedforward = self.ffn(normalized_attn)
        output = self.layernorm_3(normalized_attn + feedforward)
        return output