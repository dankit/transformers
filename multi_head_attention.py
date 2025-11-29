import torch.nn as nn
import torch
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, p_dropout, d_type, device, mask=False):
        super().__init__()
        self.num_heads = num_heads
        self.d_type = d_type
        self.device = device
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=p_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.masked = mask
    
    def forward(self, q, k, v, padding_mask=None):
        batch_size, seq_len, dim = q.shape
        kv_seq_len = k.shape[1]
        Q, K, V = self.W_q(q), self.W_k(k), self.W_v(v)

        #split into (batch, seq_len, n_heads, dim), then shape to (batch , n_heads, seq_len, dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, -1).transpose(1,2)
        K = K.view(batch_size, kv_seq_len, self.num_heads, -1).transpose(1,2)
        V = V.view(batch_size, kv_seq_len, self.num_heads, -1).transpose(1,2)

        #transpose so we can do [seq_len x dim] x [dim x kv_seq_len]
        K_t = K.transpose(-1, -2)

        #QK shape = (batch x n_heads x seq_len x kv_seq_len) - full token attention matrix for each head
        QK = (Q @ K_t) / math.sqrt(Q.shape[-1])
        
        #used for ignoring padding tokens, padding mask is batch x kv_seq_len
        if padding_mask is not None:
            #shape to (batch len x 1 x 1 x kv_seq_len)
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            #padded tokens are 0 in the pad_mask
            QK = QK.masked_fill(padding_mask == 0, float('-inf'))

        #used in decoder masked self attention, returns lower triangle part of matrix, zeros out other elements
        if self.masked:
            #initially [seq_len x kv_seq_len] -> (1 x 1 x seq_len x kv_seq_len)
            mask = torch.tril(torch.ones(seq_len, kv_seq_len, device=self.device)).bool().unsqueeze(0).unsqueeze(0)
            QK = QK.masked_fill(~mask, float('-inf')) #negate to flip false -> true, set values to -inf

        #I believe modern implementations dropout here instead of on the output, prevents relying heavily on specific attention patterns
        QK = self.softmax(QK)
        output = QK @ V
        #output shape = (batch x n_heads x seq_len x dim)
        #shape back to (batch x seq len x dim)
        output = output.transpose(1,2).view(batch_size, seq_len, -1)
        #final linear projection
        output = self.W_o(output)
        return self.dropout(output)


'''
Personal notes:

torch.tril() by itself does NOT create attention mask
softmax is e^i / summation(e^n), tril puts diaganol values to 0. e^0 = 1 , so would return 1 / denominator, 
need smaller number, -10000 would be closer to 0/denominator

broadcasting looks at dimensions from right to left. 
we manually unsqueeze to not rely on implicit inference. it doesn't actually copy the tensor, shares same memory as original tensor.

unsqueeze at dim, means that 1 is inserted exactly at dim. all subsequent dimensions are shifted right.
if doing unsqueeze(2,3,4) at dim=1, then it becomes (2,1,3,4).


during cross attention:
q = trg_len x d_model
k^t = d_model x src_len
qk = trg_len x src_len
qk * v = [trg_len x src_len] x [src_len x d_model] = trg_len x d_model
final output = trg_len x d_model

Padding was added at the end, and in this case we might be able to add the mask to a buffer like we did on positional encodings
'''