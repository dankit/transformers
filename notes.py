'''
words -> tokenizer (bpe) -> represent numerically -> embed tokens.
Each token is represented by a number, anywhere from 1 to vocab size

each token is passed through a single layer nn (embedding layer), into an embedding vector
gpt-2 used 768, gpt-3 used 12,288

after embedding layer, add with positional encoding

then pass through transformer
Encoding layer:
Multi head attention -> add residual + layer norm
feed forward -> add residual + layer norm

if encoder/decoder architecture, feed into decoder


Decoder layer: auto-regressive.
embed outputs -> positional encoding -> masked multihead attention (for training, cant see next tokens) -> add & norm
multi head self attention with residual from encoder layer (if using encoder/decoder) -> add residual from decoder layer (not encoder), norm
FFN, add norm -> linear -> softmax -> output probabilities


In essence, for encoder/decoder
take x1,..,xn -> map to z1, ... , zn -> decoder takes z and outputs y1, ..., yn one symbol at a time autoregressively


Encoder stack x 6
two sublayers in each layer
-multi-head self attention
-FFN
-output of each sublayer is LayerNorm(x + Sublayer(x)), x is the residual, sublayer(x) is result when x is passed through
To facilitate residual connections, all sub-layers in model and embedding layer are outputs of dimension = 512. [x1, ... , x512]


Decoder stack x 6
three sublayers
-masked head self attention - can only attend to positions before current position i.
-multi head self attention over output of encoder stack
-FFN
-also uses layer norm/residuals
-final output is linear + softmax (probability distribution)

***params*** 6x encoder/decoder layers

*residuals are needed to prevent vanishing gradient problem*, prevents gradient from becoming too small.
remember, numbers between 0-1 times each other get smaller. numbers greater than 1 times each other get bigger (exploding gradient)
Residuals also allow for each layer to "refine" outputs of previous layers. Rather than completely transforming it, it builds ontop of it.
In essence, this preserves information with adding x, to the transformation of layer(x).
In which now, each transformation is a delta to the original x, rather than a complete new representation that has to be learned.
By preventing vanishing gradients, essentially prevents forgetting of information.

Attention:
Dimension of query and key = d_q == d_k
dimension of value = d_v (separate)
Uses scaled dot product, Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    need to tranpose as query and key matrix are stored differently with a shape of [num_tokens x d]
    query [num_tokens x d] = each row is a query for a token, key [num_tokens x d] = each row is a key for a token
    doing [num_tokens x d] x [num_tokens x d] doesnt work as inner dimensions dont match, transposing fixes the geometry, [num_tokens x d]  x [d x num_tokens]
    after doing QK, this gives the attention score weight (how much attention to give)
Compute dot product of all queries with keys, divide by sqrt(d_k), apply softmax to obtain weights on values
Q,K,V are all matrices
for large values of d_k, its thought that dot product grows large in magnitude pushing softmax functions into small gradients. to prevent, scale by dividing by sqrt(d_k)

Multi-head attention:
Instead of using one single attention head with d_model-dimensional key,value,query, they instead linearly project Q,K,V h times.
Results in different matrixes of d_k, d_k, d_v dimensions, h times for each matrix.
Advantage of MHA is that it allows to attend to more fine grained representations, for different objectives
    one could be for positional patterns (a -> b, 1 -> 2), one for verbs, one for adjectives, etc. (as an example)
Consolidating into one head/matrix makes it so that information is lost, becomes more "noisy"

At the end, concat all attention heads, multiply by W^O 

projections for attention head i:
Wq = d_model x d_k
Wk = d_model x d_k
Wv = d_model x d_v


W^o = hd_v x d_model, h = # heads, d_v = output size of each head, hd_v = concatentated output of all heads into one matrix, project back to d_model

***params*** in paper, used h = 8 parallel attention heads. d_k = d_v = d_model/h = 64 dimensions. Total computational cost is still the same as one fully dimensional head


Attention usage has three different patterns:
encoder/decoder attention layer: queries come from previous decoder layer, memory key and values come from output of encoder.
    -allows every position in decoder to attend to all positions in input sequence

encoder self attention layer: all Q,K,V come from same place, the output of previous layer in the encoder.
    -each position in encoder can attend to all positions in the previous layer of the encoder.

decoder self attention layer: self attention layers in decoder allow each position in decoder to attend to all positions in decoder up to and including that position.
    -Need to prevent leftward information flow to preserve auto-regressive property, implemented in scaled dot product attention by masking out (-inf value) all positions > i

*during training, model is given entire sequence, which is why masking is needed*


Activation function:
each fully connected feed forward layer has two linear transformations, with a ReLU activation in between

Embedding & softmax:
Use learned embeddings to convert the input tokens and output tokens to vectors of dimension d_model.
Learned linear transformation (layer) and softmax to convert decoder output to predicted next word probabilities.
Use same weight matrix between two embedding layers and the pre-softmax linear transformation.
***In embedding layer, multiply weights by sqrt(d_model)***


Positional encoding:
Each dimension in positional encoding corresponds to sinusoid (sine wave)
PE(pos, 2i) = sin(pos/10000^(2i / d_model) )
PE(pos, 2i+1) = cos(pos/10000^(2i / d_model) )
pos = token position, i = dimension of token
so technically, i = d_model / 2, i indexes a pair of dimensions 
*** Learned and fixed positional embeddings had identical results. ***
Hypothesized model can learn to attend by relative position, as for a given offset k, PE_pos+k is a linear function of PE_pos using trig identity rules for sums of angles
Add positional encodings to embeddings of encoder/decoder.


i determines the wavelength/frequency of the graph for a given dimension.
 Essentially, each dimension gets its own graph (with both a sine and cosine wave), shared across all tokens, determined by i
 changing pos, is like taking a different snapshot of all the cosine/sine graphs at a given time, in which each cosine/sine graph has a diff frequency depending on i.
 changing pos is the same as moving along the x axis.

 low i = values change rapidly between adjacent positions, good for capturing local information (words next to each other). start vs end not as noticable compared to adjacent words.
 large i = values change slowly, good at capturing global position in long sequences (start vs end word), words next to each other dont change much, but start vs end is noticable
 i contains both local and global information, combining these granularities into one vector.
 pos leverages this vector to determine the token's unique fingerprint within the positional encoding "table".

cosine is needed because to calculate relative position/distance between any two positions
to calculate sin(pos+k) = sin(p)cos(k) + cos(p)sin(k)
             cos(pos+k) = cos(p)cos(k) - sin(p)sin(k)
             this is a linear function
also allows for pos = 0 to have meaningful info, cos(0) = 1 (strong signal), sin(0) = 0 (no signal)
if only sine was used, would be indistinguishable from no positional encoding
this is all important for the learned attention matrix

cosine/sine also allows for uniqueness, can tell which phase the wave is in as sine/cosine are phase shifted.
less ambiguity on positions, imagine a sine graph like /\  /\    -> can hit same spot twice within a cycle, cosine lets you know if wave is going up or down at a certain value.
                                                         \/  \/     e.g. if sine is at 0, if cosine is going down sine is going up. if cosine is going up, sine is going down.
sine and cosine are always between -1 and 1 keeping values normalized

wavelength and frequency are inversely proportional. 
wavelength = distance between two consecutive points of same phase (think distance between two peaks)
frequency = number of waves that pass a point per second. (how long between two peaks)


Regularization during training:
dropout to output of each sublayer before added to the sublayer input and normalized. rate of P_drop = .1.
also apply dropout to the sums of the embeddings w/ positional encodings in both encoder and decoder stacks.



params:
d_model 512 (input/output of sublayers)
heads = 8
d_q, d_k, d_v = 64
encoder/decoder layers = 6
feed forward layer = 2048
'''