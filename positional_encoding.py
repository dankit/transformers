import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, p_dropout, d_type, device):
        super().__init__()
        self.dropout = nn.Dropout(p=p_dropout)

        #position = position within seq_len, i= dimension within the embedding/vector
        positions = torch.arange(max_seq_len, device=device).unsqueeze(1) #(max_seq_len,1), has to be applied column wise
        i = (torch.floor_divide(torch.arange(d_model, device=device), 2)).float().unsqueeze(0) #(1xd_model), has to be applied row wise
        #round values down, initial tensor is [0, 1, 2, ... , 510, 511] -> [0, 0, 1, ..., 255, 255], each value maps to the dimension now (i)
        #tensor of [10000.] of shape (1,) added to (1xd_model) -> broadcasts (1,) -> (1x512)
        base, exp_divisor = torch.tensor(10000, dtype=d_type, device=device), torch.tensor(d_model, dtype=d_type, device=device)
        divisor = torch.pow(base, ((2 * i) / exp_divisor))
        #(max_seq_len,1) / (1,d_model) -> broadcast to (max_seq_len x d_model)
        args = positions / divisor
        pe = torch.zeros((max_seq_len, d_model), dtype=d_type, device=device)
        pe[:, 0::2] = torch.sin(args[:, 0::2])
        pe[:, 1::2] = torch.cos(args[:, 1::2])
        self.register_buffer('pe', pe.unsqueeze(0)) # unsqueeze so its broadcastable to all batches
        #pe is (1, max_seq_len, d_model)

    def forward(self, x):
        #x shape: (batch_size, seq_len, d_model) where seq_len <= max_seq_len
        #self.pe[:, :x.size(1), :] slices to (1, seq_len, d_model), then broadcasts to batch_size
        #each example in a batch might be padded to a fraction of the max_len
        #if max_len is 2048, and we have 4 examples, each example might be padded to 512
        return self.dropout(x + self.pe[:, :x.size(1), :])

'''
Mainly beginner pytorch/linear algebra notes as a refresher. 

row wise = groups all data for one entry together
column wise = process data vertically, all data for one attribute/feature together (positions)

-not all tensor operations follow matmul rules. AKA element-wise operations (+, -, *, pow)
-e.g. torch.pow(), it will broadcast the 1x1 array to match exponent of 512x1
-it does the corresponding i,j index for a matrix


matmul follows its own rules, matmul is NOT the same as element wise multiplication
matrix multiplication is for representing linear transformations - e.g. rotating, scaling, projecting vector into new space

when doing A @ x, (A is matrix, x is vector) for every element in Ax (vector), every component of the new vector is a mixture of all of
the components in the old vector x.

if x is input vectors with features 'height, weight, age' and A is the weight matrix,
Ax calculates weighted sum for all inputs for each neuron. Neuron depends on all info passed in from previous layer, which is why its needed

if we only did element wise multiplication, we would only be scaling values. by multiplying and adding all the values, we are transforming.

Essentially, matmul is for having an aggregated representation of all the inputs. the neuron aggregates information from all inputs,
and learns a new, compound feature.



broadcasting always goes right to left.
so comparing (30,15) to (30,) will fail, as 15 =/= 30.
however (30, 15) to (15,) will work, as 15 == 15, and then it will pad a 1 becoming (1,15)


PE(pos, 2i) = sin(pos/10000^(2i / d_model) )
PE(pos, 2i+1) = cos(pos/10000^(2i / d_model) )
pos = token position, i = dimension of token
so technically, i = d_model / 2, a dimension indexes a pair of dimensions

encodings are dependent on position index and seq length, not the embeddings themselves

pytorch buffer :
non-trainable tensor that does not require gradients
essentially a state that is non-trainable (fixed), and needs to be associated with module
'''