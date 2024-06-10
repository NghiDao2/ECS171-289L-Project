import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from mamba_ssm.modules.mamba_simple import Mamba, Block
from collections import namedtuple
import numpy as np

Prediction = namedtuple("Prediction", ("denoised"))

class MambaDenoiser(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        n_channels: int = 4
    ):
        super().__init__()
        assert n_layers % 8 == 0, f"Number of layers must be division by 8"

        
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.n_channels = n_channels
        self.patch_size = patch_size
        
        patch_dim = self.n_channels * self.patch_size * self.patch_size
        
        n_io=3
        n_f=embed_dim
        n_b=n_layers
        self.enc = nn.Sequential(nn.Conv2d(n_io + 1, n_f, 1), nn.ReLU(), nn.Conv2d(n_f, n_f, 1, bias=False), nn.PixelUnshuffle(patch_size))
        self.mid = nn.ModuleList(Block(n_f * patch_size**2, Mamba) for _ in range(n_b))
        self.dec = nn.Sequential(nn.Conv2d(n_f * patch_size**2 * 3, n_f * patch_size**2, 1), nn.ReLU(), nn.PixelShuffle(patch_size), nn.Conv2d(n_f, n_io, 1))


    def transpose_xy(self, *args):
        # swap x/y axes of an N[XY]C tensor
        return [a.view(a.shape[0], int(a.shape[1]**0.5), int(a.shape[1]**0.5), a.shape[2]).transpose(1, 2).reshape(a.shape) for a in args]

    def flip_s(self, *args):
        # reverse sequence axis of an NSE tensor
        return [a.flip(1) for a in args]

    def forward(self, x, noise_level):
        
        x = self.enc(torch.cat([x, noise_level.expand(x[:, :1].shape)], 1))
        y = x.flatten(2).transpose(-2, -1)
        z = None
        for i, mid in enumerate(self.mid):
            y, z = mid(y, z)
            
            y, z = self.transpose_xy(y, z)
            if (i + 1) % 4 == 0:
                y, z = self.flip_s(y, z)
                
        y, z = y.transpose(-2, -1).view(x.shape), z.transpose(-2, -1).view(x.shape)
        out = self.dec(torch.cat([x, y, z], 1))
        
        return Prediction(out)
