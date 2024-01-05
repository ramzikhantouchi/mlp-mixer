import torch as tr
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Reduce, Rearrange
from layers import MLPBlock, MixerBlock

class MLPMixer(nn.Module):
    
    def __init__(self, image_dim, patch_size, embedding_dim, hidden_dim, num_blocks, num_classes, dropout = 0.2, channels = None):
        super(MLPMixer, self).__init__()
        
        assert (image_dim[0] % patch_size) == 0 and (image_dim[1] % patch_size) == 0, f"The image dimensions must be divisible by the patch size ({patch_size})"
        self.num_patches_h = image_dim[0] // patch_size
        self.num_patches_w = image_dim[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        if channels == None:
            self.patches = Rearrange('b (p1 h) (p2 w) -> b (p1 p2) (h w)', p1 = self.num_patches_h, p2 = self.num_patches_w)
            self.patch_encoding = nn.Linear(patch_size**2, self.embedding_dim)
        else:
            self.patches = Rearrange('b (p1 h) (p2 w) c -> b (p1 p2) (h w c)', p1 = self.num_patches_h, p2 = self.num_patches_w)
            self.patch_encoding = nn.Linear((patch_size**2)*channels, self.embedding_dim)
        self.mixer_blocks = nn.Sequential(*[MixerBlock((self.num_patches,self.embedding_dim), self.hidden_dim) for _ in range(num_blocks)
            
        ])
        self.dropout = nn.Dropout(dropout)
        self.global_avg_pooling = Reduce('b h w -> b w', 'mean')
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)
    
    def forward(self, inputs):
        
        x = self.patches(inputs)
        x = self.patch_encoding(x)
        x = self.mixer_blocks(x)
        x = self.dropout(x)
        x = self.global_avg_pooling(x)
        x = self.classifier(x)
        return x