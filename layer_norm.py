import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))
        self.epsilon = 1e-5

    def forward(self, x):
        #shape batch_size x seq_len x dim
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, dim=-1, keepdim=True)
        y = (x - mean) / (torch.sqrt(variance + self.epsilon))
        y = (y * self.gamma) + self.beta
        return y



'''
variance = avg of squared differences from mean, uses squared units.  
standard deviation (sigma) = sqrt(variance) -> same units as original data, more interpretable

Layernorm:
step 1: normalization ->
calculate mean and variance
normalize by subtracting mean, divide by sqrt of variance. makes data have a mean of 0 and variance of 1

step 2: scale & shift-> 
gamma = scaling factor (controls how much each feature influences output)
beta = shift factor
recovers the correct distribution as we don't always want it to be a mean of 0 and variance of 1


Important because its stabilizes values, takes less long to converge/learn optimal values.
epsilon is added for numerical stability, prevents divison by 0

Too large values, or too small values = instability
consistent value range = more consistent updates

subtracting by mean = mean of 0
dividing by standard deviation (sqrt of variance) = variance of 1
std deviation = how much each value deviates from mean on avg,
so dividing by the deviation means that all values are scaled to the deviation
obvious point: if std deviation is 2.49, dividing by 2.49 puts all values on scale of 1 unit (2.49) instead.
We can also use z-score with the standard deviation to see the spread of data.


Intuitive example:
input x
token 0: [1,2,3,4] mean = 2.5, var=1.25
token 1: [10,20,30,40] mean=25, var=12.5

after normalization (mean=0, variance=1)
token 0: [-1.342, -.447, .447, 1.342]
token 1: [-1.342, -.447, .447, 1.342] (the same)

then multiply by gamma, add beta (learnable params)
output = learned distribution, scaled and shifted per dimension

A good counter point is what if we only did mean=0, no variance normalization? then,
token 0: [-1.5, -0.5, 0.5, 1.5]
token 1: [-15, -5, 5, 15] -> token 1 is 10x larger 
we have only shifted the distribution around the mean, but values are still unscaled. 

token 1 would dominate in attention and gradients would be unstable. gamma/beta needs to handle different scales too (bad)

so by dividing by sqrt(variance), all tokens live within a similar numerical range regardless of magnitude.
The tokens dont live within a set bounds per say, e.g. [-1,1], but converts them to a smaller/similar scale.
'''