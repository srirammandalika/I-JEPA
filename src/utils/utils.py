# src/utils/utils.py

import torch
import torch.nn as nn
from math import pi, sin, cos

class PatchEmbed(nn.Module):
    """
    Turn an image into a sequence of patch embeddings.
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.grid = img_size // patch_size
        self.n_patches = self.grid * self.grid

    def forward(self, x):
        # x: [B, C, H, W] -> conv -> [B, D, H/ps, W/ps]
        x = self.conv(x)
        # flatten patches: [B, D, N]
        x = x.flatten(2)
        # [B, N, D]
        return x.transpose(1, 2)

def get_sinusoid_pos_emb(n_positions, dim):
    """
    Create a fixed sinusoidal positional embedding [n_positions, dim].
    """
    pe = torch.zeros(n_positions, dim)
    position = torch.arange(0, n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
