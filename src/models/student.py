# src/models/student.py

import torch
import torch.nn as nn
import random
from src.utils.utils import PatchEmbed, get_sinusoid_pos_emb

class IJEPAStudent(nn.Module):
    """
    Student network for distilled I-JEPA.
    - Mask up to `mask_ratio` of patches.
    - Encode visible patches via Transformer encoder.
    - Predict masked-patch embeddings via a small Transformer predictor.
    - Decode predicted embeddings back to pixel patches.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 64,
        enc_depth: int = 6,
        pred_depth: int = 3,
        num_heads: int = 8,
        mask_ratio: float = 0.25,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans

        # 1) Patch embedding + positional embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.N = self.patch_embed.n_patches
        pos_emb = get_sinusoid_pos_emb(self.N, embed_dim)  # [N, D]
        self.register_buffer("pos_emb", pos_emb)

        # 2) Context encoder (student)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.ctx_enc = nn.TransformerEncoder(enc_layer, num_layers=enc_depth)

        # 3) Predictor head
        pred_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.pred = nn.TransformerEncoder(pred_layer, num_layers=pred_depth)

        # 4) Learned mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 5) Pixel-decoder: D → (in_chans × patch_size × patch_size)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, in_chans * patch_size * patch_size)
        )

        # 6) Masking hyperparameter
        self.mask_ratio = mask_ratio

    def sample_mask(self, batch_size: int):
        """
        Sample lists of masked patch indices, K = int(mask_ratio * N) per sample.
        """
        K = int(self.mask_ratio * self.N)
        return [random.sample(range(self.N), K) for _ in range(batch_size)]

    def decode_patches(self, emb: torch.Tensor):
        """
        Decode predicted embeddings into RGB patches.
        emb: [K, D] → returns [K, in_chans, patch_size, patch_size]
        """
        K, D = emb.shape
        out = self.decoder(emb)  # [K, in_chans * p * p]
        return out.view(K, self.in_chans, self.patch_size, self.patch_size)

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        Returns:
          all_preds: list of [K, D] predicted embeddings per sample
          mask_idxs: list of masked patch indices per sample
        """
        B = x.size(0)

        # 1) Embed patches + add positional embedding
        x_patches = self.patch_embed(x)                   # [B, N, D]
        x_ctx     = x_patches + self.pos_emb.unsqueeze(0) # [B, N, D]

        # 2) Sample and apply mask
        mask_idxs = self.sample_mask(B)
        mask      = torch.zeros(B, self.N, dtype=torch.bool, device=x_ctx.device)
        for b in range(B):
            mask[b, mask_idxs[b]] = True
        x_ctx = x_ctx.masked_fill(mask.unsqueeze(-1), 0.0)  # zero-out masked

        # 3) Encode context
        ctx_feats = self.ctx_enc(x_ctx)                    # [B, N, D]

        # 4) Predict masked embeddings
        all_preds = []
        for b in range(B):
            seq = ctx_feats[b].unsqueeze(0).clone()        # [1, N, D]
            # inject mask token + pos_emb at masked positions
            tok = self.mask_token + self.pos_emb[mask_idxs[b]]  # [K, D]
            seq[0, mask_idxs[b]] = tok
            out = self.pred(seq)                           # [1, N, D]
            preds_b = out[0, mask_idxs[b], :]              # [K, D]
            all_preds.append(preds_b)

        return all_preds, mask_idxs
