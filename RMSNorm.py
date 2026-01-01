import torch.nn as nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = 1e-6 # prevents divisor from going to 0 by adding a very small number
        self.gamma = nn.Parameter(torch.ones(dim))
    
    #come back later, use f.rms_norm (does everything in one kernel)
    def forward(self, x):
        #use rsqrt instead, fused op
        rms = (x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt())
        return (x  / rms ) * self.gamma

'''
RMS formula: RMSNorm(x) = ( x / sqrt(RMS(x)) ) * gamma
                RMS(x) = ((x1^2 + x2^2 + x3^2 ... xn^2) / n) ** 0.5   (RMS = squareroot of the average of the squared values in each dimension)
epsilon is a hyperparameter, constant value (not learned). prevents division by zero in rare cases that the sum of squares is very close to zero. allows for numerical stability.
gamma is learnable parameter,allows network to adjust scale of normalized activations if needed. Can scale up or down for optimal performance in subsequent layers. usually intialized as vector of ones.

main difference between layernorm and RMSnorm is that layernorm uses both mean/variance, rmsnorm only uses RMS (root mean square), omits mean-centering step.

weaknesses of rms:
no zero mean - relies on model to handle bias/shifts
some info loss/slightly less expressive
less stable in small setups


rsqrt is done in one operation, fused kernel for rms does it in one kernel
rsqrt does this through "fast inverse sqrt", some very minor error but good enough to use for approximating for speed.
rsqrt is less accurate than sqrt + div+

Doesn't need beta, because we removed mean shift, so no need for recovering original distribution through shift? (only scale)
'''