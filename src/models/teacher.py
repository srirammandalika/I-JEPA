# src/models/teacher.py

import torch
import torch.nn as nn
import copy
from src.utils.utils import PatchEmbed, get_sinusoid_pos_emb
from src.models.student import IJEPAStudent


class JEPEATeacher(nn.Module):
    """
    EMA Teacher wrapper for distilled I-JEPA.
    Maintains an exponential moving average copy of the student's context encoder.
    Provides frozen target embeddings for distillation.
    """
    def __init__(self, student: IJEPAStudent, ema_m: float = 0.996):
        super().__init__()
        # Copy student's patch embedding & positional embedding
        self.patch_embed = copy.deepcopy(student.patch_embed)
        self.pos_emb     = student.pos_emb
        # EMA context encoder
        self.teacher_enc = copy.deepcopy(student.ctx_enc)
        # Freeze teacher parameters
        for p in self.teacher_enc.parameters():
            p.requires_grad = False
        # EMA momentum
        self.ema_m = ema_m

    @torch.no_grad()
    def update(self, student: IJEPAStudent):
        """
        Update teacher encoder parameters as EMA of student context encoder.
        """
        for p_t, p_s in zip(self.teacher_enc.parameters(), student.ctx_enc.parameters()):
            p_t.data.mul_(self.ema_m).add_(p_s.data, alpha=1.0 - self.ema_m)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        Compute teacher embeddings for full image patches.
        Returns:
          teacher_feats: Tensor [B, N, D]
        """
        # x: [B, C, H, W]
        # 1) patch embed + add pos
        x_patches = self.patch_embed(x)               # [B, N, D]
        x_emb     = x_patches + self.pos_emb.unsqueeze(0)  # [B, N, D]
        # 2) pass through frozen teacher encoder
        teacher_feats = self.teacher_enc(x_emb)       # [B, N, D]
        return teacher_feats
