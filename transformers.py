import torch.nn as nn
import torch
import math
from positional_encoding import PositionalEncoding
from encoder_block import EncoderBlock
from decoder_block import DecoderBlock

'''I believe with BERT, after pre-training a task-specific head (linear layer) can be added to convert it to a cross-encoder with further finetuning
For embeddings, pool the tokens'''
class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len, p_dropout, d_type, device):
        super().__init__()
        self.d_model = d_model
        self.d_type = d_type
        self.device = device
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, dtype=d_type)
        self.positional_encodings = PositionalEncoding(d_model, max_seq_len, p_dropout, d_type, device)
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, p_dropout, d_type, device) for _ in range(num_layers)])

    def forward(self, x, padding_mask=None):
        x = self.embeddings(x) * math.sqrt(self.d_model)
        x = self.positional_encodings(x)
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x #(batch, seq, dim) (each output is a collection of embeddings for each token at this point) 

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len, p_dropout, d_type, device):
        super().__init__()
        self.d_model = d_model
        self.d_type = d_type
        self.device = device
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, dtype=d_type)
        self.positional_encodings = PositionalEncoding(d_model, max_seq_len, p_dropout, d_type, device)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, p_dropout, d_type, device) for _ in range(num_layers)])
    
    def forward(self, x, padding_mask=None):
        x = self.embeddings(x) * math.sqrt(self.d_model)
        x = self.positional_encodings(x)
        for layer in self.layers:
            x = layer(x, padding_mask) #(batch, seq, dim)
        #(seq, dim) x (dim, vocab_size) = (seq, vocab_size)
        logits = x @ self.embeddings.weight.T
        return logits

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len, p_dropout, d_type, device):
        super().__init__()
        self.d_model = d_model
        self.d_type = d_type
        self.device = device
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, dtype=d_type)
        self.positional_encodings = PositionalEncoding(d_model, max_seq_len, p_dropout, d_type, device)
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, num_heads, p_dropout, d_type, device) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, num_heads, p_dropout, d_type, device) for _ in range(num_layers)])
    
    def forward(self, src_seq, trg_seq, src_padding_mask, trg_padding_mask):
        scale_factor = math.sqrt(self.d_model)

        src_seq = self.embeddings(src_seq) * scale_factor
        src_seq = self.positional_encodings(src_seq)
        for layer in self.encoder_layers:
            src_seq = layer(src_seq, src_padding_mask) #(batch, seq, dim)

        trg_seq = self.embeddings(trg_seq) * scale_factor
        trg_seq = self.positional_encodings(trg_seq)
        for layer in self.decoder_layers:
            trg_seq = layer(trg_seq, trg_padding_mask, src_seq, src_padding_mask)
        #(seq, dim) x (dim, vocab_size) = (seq, vocab_size) -> distribution of probabilities (softmaxed)
        #i think doing inner product here is like doing similarity with embeddings, higher score = closer in meaning
        logits = trg_seq @ self.embeddings.weight.T
        return logits

'''
dropout is used to avoid overfitting

using modulelist, as nn.sequential does not allow multiple arguments

encoder creates a representation or final "embedding" which is why its for encoding information. Attends to all tokens

embeddings are multiplied by sqrt(d_model) to boost magnitude  otherwise positional encodings would be disproportionately large

use softmax(outputs) + return torch.topk on inference, 
for training return raw logits as crossentropyloss expects logits, does softmax internally

padding happens externally
1) tokenizer outputs variable length sequences
2) collate function in dataloader (combines samples into a batch) creates both padded token input, and padding mask

The only place tokens interact is within the attention layer
FFN doesn't mix tokens, it creates a position wise transformation per token (looks at tokens individually).
This is why pad token is not masked in pe, ffn

encoder and decoder run on different inputs (diff token streams), not same one
decoder is fed a special token at the start, <t_start>, doesnt start empty
at the first step, token_0 (<t_start>), attends to itself -> generates next token
at second step, token_1 (t_start, generated_token), t_start attends to itself, generated_token attends to t_start and itself
first decoder uses masked self attention, then uses cross-attention with encoder for the keys/values
during inference, an external loop appends the output of the decoder to the trg_seq, and re-calls. it does not rerun the encoder step.
encoder output is cached, then reruns decoder until <eos>

Depending on task, e.g. translation, source and target language have separate vocab/embeddings
'''