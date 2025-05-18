# train_planner.py

import os
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.models.teacher import JEPEATeacher
from src.data.cifar10 import get_cifar10_dataloaders

class ContextualPlanner(nn.Module):
    """
    System 2: Contextual Transformer Planner.
    Masks a small fraction of patch embeddings (from teacher),
    refines via deep Transformer, and predicts masked embeddings.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        enc_depth: int = 12,
        pred_depth: int = 4,
        num_heads: int = 8,
        mask_ratio: float = 0.15,
        pos_emb: torch.Tensor = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        # positional embeddings (copied from teacher)
        assert pos_emb is not None, "Need positional embeddings"
        # pos_emb: [N, D]
        self.register_buffer("pos_emb", pos_emb)

        # deep context encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.ctx_enc = nn.TransformerEncoder(enc_layer, num_layers=enc_depth)

        # predictor head
        pred_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.pred = nn.TransformerEncoder(pred_layer, num_layers=pred_depth)

        # learned mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # number of patches
        self.N = pos_emb.shape[0]

    def sample_mask(self, batch_size: int):
        """
        Sample per-sample lists of masked patch indices.
        Mask exactly K = int(mask_ratio * N) patches per image.
        """
        K = int(self.mask_ratio * self.N)
        return [torch.randperm(self.N)[:K].tolist() for _ in range(batch_size)]

    def forward(self, teacher_feats: torch.Tensor):
        """
        teacher_feats: [B, N, D] embeddings from JEPEATeacher
        Returns:
          preds_list: list of [K, D] tensors per sample
          mask_idxs: list of masked indices per sample
        """
        B, N, D = teacher_feats.shape
        # 1) add positional embedding
        x = teacher_feats + self.pos_emb.unsqueeze(0)  # [B, N, D]

        # 2) sample and mask
        mask_idxs = self.sample_mask(B)
        x_ctx = x.clone()
        for b in range(B):
            for idx in mask_idxs[b]:
                x_ctx[b, idx] = 0

        # 3) deep context encoding
        ctx_feats = self.ctx_enc(x_ctx)  # [B, N, D]

        # 4) predict masked embeddings
        preds_list = []
        for b in range(B):
            seq = ctx_feats[b].unsqueeze(0)  # [1, N, D]
            # inject mask token at masked positions
            for idx in mask_idxs[b]:
                seq[0, idx] = self.mask_token + self.pos_emb[idx]
            out = self.pred(seq)  # [1, N, D]
            preds_b = out[0, mask_idxs[b], :]  # [K, D]
            preds_list.append(preds_b)

        return preds_list, mask_idxs


def parse_args():
    parser = argparse.ArgumentParser(description="Train System-2 Contextual Planner")
    parser.add_argument("--train_root", type=str, required=True,
                        help="Path to CIFAR-10 training data root")
    parser.add_argument("--val_root", type=str, default=None,
                        help="Path to CIFAR-10 validation data root (optional)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--ema_m", type=float, default=0.996,
                        help="EMA momentum for loading teacher encoder")
    parser.add_argument("--enc_depth", type=int, default=12)
    parser.add_argument("--pred_depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/teacher_enc.pth",
                        help="Path to pretrained teacher encoder weights")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data loaders
    loaders = get_cifar10_dataloaders(
        train_root=args.train_root,
        val_root=args.val_root,
        batch_size=args.batch_size,
        img_size=None,  # not used in planner
        max_per_class=None,
    )
    if isinstance(loaders, tuple):
        train_loader, val_loader = loaders
    else:
        train_loader, val_loader = loaders, None

    # Load teacher encoder
    # We construct a dummy student to get pos_emb and then load teacher_enc
    from src.models.student import IJEPAStudent
    student_dummy = IJEPAStudent()
    teacher = JEPEATeacher(student_dummy, ema_m=args.ema_m)
    teacher.teacher_enc.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    teacher = teacher.to(device).eval()

    # Initialize planner
    planner = ContextualPlanner(
        embed_dim=student_dummy.patch_embed.conv.out_channels,
        enc_depth=args.enc_depth,
        pred_depth=args.pred_depth,
        num_heads=args.num_heads,
        mask_ratio=args.mask_ratio,
        pos_emb=teacher.pos_emb,
    ).to(device)

    optimizer = AdamW(planner.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        planner.train()
        total_loss = 0.0
        total_samples = 0

        for imgs, _ in train_loader:
            imgs = imgs.to(device)

            # 1) get teacher embeddings
            with torch.no_grad():
                teacher_feats = teacher(imgs)  # [B, N, D]

            # 2) planner predictions
            preds_list, mask_idxs = planner(teacher_feats)

            # 3) compute loss
            loss = 0.0
            count = 0
            for b, preds in enumerate(preds_list):
                targets = teacher_feats[b, mask_idxs[b], :]  # [K, D]
                loss += F.mse_loss(preds, targets, reduction="sum")
                count += preds.numel()
            loss = loss / count

            # 4) backward & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch:02d} | Planner Loss: {avg_loss:.6f}")

        # Optional validation
        if val_loader is not None:
            planner.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs = imgs.to(device)
                    teacher_feats = teacher(imgs)
                    preds_list, mask_idxs = planner(teacher_feats)
                    loss_v = 0.0
                    count_v = 0
                    for b, preds in enumerate(preds_list):
                        targets = teacher_feats[b, mask_idxs[b], :]
                        loss_v += F.mse_loss(preds, targets, reduction="sum")
                        count_v += preds.numel()
                    val_loss += (loss_v / count_v).item() * imgs.size(0)
                    val_samples += imgs.size(0)
            print(f"           Validation Loss: {val_loss/val_samples:.6f}")

    # Save planner
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(planner.state_dict(), "checkpoints/planner.pth")
    print("Planner training complete. Model saved to checkpoints/planner.pth")


if __name__ == "__main__":
    main()
