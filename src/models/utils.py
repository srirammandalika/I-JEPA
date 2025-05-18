# utils.py
import torch
import torch.nn as nn
import math
from einops import rearrange


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Create 2D sin-cos positional embeddings as in MAE.
    Returns: [grid_size*grid_size, embed_dim]
    """
    # use standard implementation
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h)
    grid = torch.stack(grid, dim=0)  # 2, Wh, Ww
    grid = grid.reshape(2, 1, grid_size, grid_size)

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0].reshape(-1))
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1].reshape(-1))
    emb = torch.cat([emb_h, emb_w], dim=1)
    if cls_token:
        emb = torch.cat([torch.zeros([1, embed_dim]), emb], dim=0)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim must be even
    pos: [M], position
    Returns: [M, embed_dim]
    """
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    omega = torch.arange(half, dtype=torch.float32) / half
n    omega = 1.0 / (10000 ** omega)
    pos = pos.unsqueeze(1)  # M,1
    out = pos * omega.unsqueeze(0)  # M, half
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # M, embed_dim
    return emb

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.n_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, Gh, Gw]
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by returning boolean masks.
    x: [B, N, D]
    mask_ratio: float in [0,1]
    Returns:
      x_masked: x with masked tokens zeroed
      mask: boolean mask [B, N] where True means masked
      mask_indices: list of masked indices per sample
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))
    mask = torch.ones([B, N], dtype=torch.bool, device=x.device)
    mask_indices = []
    for i in range(B):
        perm = torch.randperm(N, device=x.device)
        keep = perm[:len_keep]
        m = torch.ones(N, dtype=torch.bool, device=x.device)
        m[keep] = False
        mask[i] = m
        mask_indices.append(m.nonzero(as_tuple=False).flatten().tolist())
    x_masked = x.clone()
    x_masked[mask] = 0
    return x_masked, mask, mask_indices
