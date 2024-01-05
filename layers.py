import torch as tr
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Reduce, Rearrange

class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPBlock, self).__init__()
        
        self.dense_1 = nn.Linear(input_dim, hidden_dim)
        self.dense_2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, inputs):
        
        x = self.dense_1(inputs)
        x = F.gelu(x)
        x = self.dense_2(x)
        return x


class MixerBlock(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super(MixerBlock, self).__init__()
        
        self.layer_norm = nn.LayerNorm(input_dim)
        self.token_mixing = MLPBlock(input_dim[-2], hidden_dim)
        self.channel_mixing = MLPBlock(input_dim[-1], hidden_dim)
        self.transpose = Rearrange('b h w -> b w h')
    
    def forward(self, inputs):
        
        x = self.layer_norm(inputs)
        x = self.transpose(x)
        x = self.token_mixing(x)
        x = self.transpose(x)
        x = x + inputs
        y = self.layer_norm(x)
        y = self.channel_mixing(y)
        return x + y