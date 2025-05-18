# src/training/train_dynamics.py

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import transforms
from src.models.teacher import JEPEATeacher
from src.data.cifar10 import get_cifar10_dataloaders

class LatentDynamics(nn.Module):
    """
    Predict next-step patch embeddings given current embeddings and an action embedding.
    """
    def __init__(self, embed_dim, action_dim=16, hidden_dim=128):
        super().__init__()
        # Action embedding MLP
        self.act_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        # Dynamics MLP
        self.dyn_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, o, u):
        # o: [B, N, D], u: [B, action_dim]
        u_emb = self.act_mlp(u)  # [B, D]
        u_emb = u_emb.unsqueeze(1).expand(-1, o.size(1), -1)  # [B, N, D]
        inp = torch.cat([o, u_emb], dim=-1)  # [B, N, 2D]
        o_next = self.dyn_mlp(inp)  # [B, N, D]
        return o_next

class ValueNet(nn.Module):
    """
    Estimate distance to goal embedding.
    """
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, o):
        # o: [B, N, D]
        o_pool = o.mean(dim=1)  # [B, D]
        v = self.net(o_pool).squeeze(-1)  # [B]
        return v

def parse_args():
    parser = argparse.ArgumentParser(description="Train latent dynamics and value net")
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--checkpoint_teacher", type=str, default="checkpoints/teacher_enc.pth")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # CIFAR-10 loader
    train_loader = get_cifar10_dataloaders(
        train_root=args.train_root,
        batch_size=args.batch_size,
        img_size=32,
        max_per_class=None,
        num_workers=4
    )

    # Initialize teacher
    from src.models.student import IJEPAStudent
    student_dummy = IJEPAStudent()
    teacher = JEPEATeacher(student_dummy)
    teacher.teacher_enc.load_state_dict(
        torch.load(args.checkpoint_teacher, map_location="cpu")
    )
    teacher.to(device).eval()

    # Models
    embed_dim = student_dummy.patch_embed.conv.out_channels
    dyn = LatentDynamics(embed_dim, action_dim=args.action_dim, hidden_dim=args.hidden_dim).to(device)
    val_net = ValueNet(embed_dim, hidden_dim=args.hidden_dim).to(device)

    optimizer = AdamW(
        list(dyn.parameters()) + list(val_net.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # Sample actions uniformly in [-1,1]
    def sample_actions(B, dim):
        return (torch.rand(B, dim) * 2 - 1).to(device)

    # Color jitter with fixed strengths
    color_jitter = transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
    )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        dyn.train(); val_net.train()
        total_dyn_loss = 0.0
        total_val_loss = 0.0
        count = 0

        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            # Teacher embeddings at time t
            with torch.no_grad():
                o_t = teacher(imgs)  # [B, N, D]

            # Sample action and generate next image
            u = sample_actions(imgs.size(0), args.action_dim)  # [B, action_dim]
            imgs_aug = color_jitter(imgs)
            with torch.no_grad():
                o_tp1 = teacher(imgs_aug)  # [B, N, D]

            # Predict next embeddings
            o_pred = dyn(o_t, u)  # [B, N, D]
            dyn_loss = F.mse_loss(o_pred, o_tp1)

            # Value net: regress pooled distance
            v_pred = val_net(o_t)  # [B]
            dist_true = torch.norm(
                o_tp1.mean(dim=1) - o_t.mean(dim=1), dim=1
            )  # [B]
            val_loss = F.mse_loss(v_pred, dist_true)

            # Backprop
            loss = dyn_loss + val_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_dyn_loss += dyn_loss.item() * imgs.size(0)
            total_val_loss += val_loss.item() * imgs.size(0)
            count += imgs.size(0)

        print(
            f"Epoch {epoch:02d} | "
            f"Dynamics Loss: {total_dyn_loss/count:.6f} | "
            f"Value Loss:    {total_val_loss/count:.6f}"
        )

    # Save checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(dyn.state_dict(), "checkpoints/dynamics.pth")
    torch.save(val_net.state_dict(), "checkpoints/value_net.pth")
    print("Training complete. Models saved to checkpoints/")

if __name__ == "__main__":
    main()
