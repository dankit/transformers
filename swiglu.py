import torch.nn as nn
import torch.nn.functional as F
#original Swiglu paper omits bias terms
class SwigluFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = int((2/3) * 4 * config.dim) #throws an error if not int (float division)
        self.w_gate = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w_value = nn.Linear(config.dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, config.dim, bias=False)
    
    #Since we are using b=1, swish is the same as Sigmoid linear unit (silu)
    #The "gate" is the silu function (x * sigmoid(x)), values are from self.v
    #Silu is what is creating the nonlinearity in the activation. Without it, function would be linear and reduce learning capability.
    def forward(self, x):
        return self.out(F.silu(self.w_gate(x)) * self.w_value(x))