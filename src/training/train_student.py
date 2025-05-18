# src/training/train_student.py

import os
import argparse
import math
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.models.student import IJEPAStudent
from src.models.teacher import JEPEATeacher
from src.data.cifar10 import get_cifar10_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description="Train distilled I-JEPA student with pixel + cosine-loss distillation")
    p.add_argument("--train_root",    type=str,   required=True, help="Path to CIFAR-10 train folder")
    p.add_argument("--val_root",      type=str,   default=None, help="Path to CIFAR-10 val folder (optional)")
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=0.05)
    p.add_argument("--ema_m",         type=float, default=0.996, help="EMA momentum")
    p.add_argument("--mask_ratio",    type=float, default=0.25,  help="Final fraction of patches to mask")
    p.add_argument("--lambda_pixel",  type=float, default=0.1,   help="Weight for pixel loss")
    p.add_argument("--max_per_class", type=int,   default=None,  help="Limit images per class")
    p.add_argument("--img_size",      type=int,   default=32)
    p.add_argument("--patch_size",    type=int,   default=4)
    p.add_argument("--embed_dim",     type=int,   default=64)
    p.add_argument("--enc_depth",     type=int,   default=6)
    p.add_argument("--pred_depth",    type=int,   default=3)
    p.add_argument("--num_heads",     type=int,   default=8)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--device",        type=str,   default=None, help="mps/cuda/cpu; auto if unset")
    return p.parse_args()


def main():
    args = parse_args()

    # Device selection
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
        img_size=args.img_size,
        max_per_class=args.max_per_class,
        num_workers=args.num_workers,
    )
    if isinstance(loaders, tuple):
        train_loader, val_loader = loaders
    else:
        train_loader, val_loader = loaders, None

    # Models
    student = IJEPAStudent(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        enc_depth=args.enc_depth,
        pred_depth=args.pred_depth,
        num_heads=args.num_heads,
        mask_ratio=args.mask_ratio,  # this is final mask_ratio
    ).to(device)

    teacher = JEPEATeacher(student, ema_m=args.ema_m).to(device)

    optimizer = AdamW(
        list(student.patch_embed.parameters()) +
        list(student.ctx_enc.parameters()) +
        list(student.pred.parameters()) +
        list(student.decoder.parameters()) +
        [student.mask_token],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning-rate scheduler: warmup (5 epochs) + cosine decay
    total_steps = args.epochs * len(train_loader)
    warmup_steps = 5 * len(train_loader)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Constants for mask scheduling
    init_mask = 0.1
    final_mask = args.mask_ratio

    # Training loop
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        # Mask-ratio scheduling: ramp from init_mask → final_mask over epochs
        student.mask_ratio = init_mask + (final_mask - init_mask) * (epoch / args.epochs)

        student.train()
        total_loss = 0.0
        total_samples = 0
        p = args.patch_size
        grid = args.img_size // p

        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            B = imgs.size(0)

            # 1) Teacher targets
            with torch.no_grad():
                teacher_feats = teacher(imgs)  # [B, N, D]

            # 2) Student predictions
            preds_list, mask_idxs = student(imgs)

            # 3) Compute joint loss
            loss_sum = 0.0
            patch_count = 0
            for b, preds in enumerate(preds_list):
                # 3a) Cosine-similarity distillation
                tgt = teacher_feats[b, mask_idxs[b], :]  # [K, D]
                l_cos = (1.0 - F.cosine_similarity(preds, tgt, dim=-1)).sum()

                # 3b) Pixel reconstruction
                decoded = student.decode_patches(preds)  # [K,3,p,p]
                img_b = imgs[b]  # [3,H,W]
                gt_patches = []
                for idx in mask_idxs[b]:
                    i, j = divmod(idx, grid)
                    y0, y1 = i*p, (i+1)*p
                    x0, x1 = j*p, (j+1)*p
                    gt_patches.append(img_b[:, y0:y1, x0:x1])
                gt_patches = torch.stack(gt_patches, dim=0)  # [K,3,p,p]
                l_pix = F.mse_loss(decoded, gt_patches, reduction='sum')

                loss_sum += l_cos + args.lambda_pixel * l_pix
                patch_count += preds.numel()

            loss = loss_sum / patch_count

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            teacher.update(student)
            scheduler.step()
            global_step += 1

            total_loss += loss.item() * B
            total_samples += B

        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch:03d} | Combined Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.3e}")

        # Optional validation
        if val_loader:
            student.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs = imgs.to(device)
                    teacher_feats = teacher(imgs)
                    preds_list, mask_idxs = student(imgs)

                    v_sum = 0.0
                    v_count = 0
                    for b, preds in enumerate(preds_list):
                        tgt = teacher_feats[b, mask_idxs[b], :]
                        l_cos = (1.0 - F.cosine_similarity(preds, tgt, dim=-1)).sum()
                        decoded = student.decode_patches(preds)
                        img_b = imgs[b]
                        gt_patches = []
                        for idx in mask_idxs[b]:
                            i, j = divmod(idx, grid)
                            y0, y1 = i*p, (i+1)*p
                            x0, x1 = j*p, (j+1)*p
                            gt_patches.append(img_b[:, y0:y1, x0:x1])
                        gt_patches = torch.stack(gt_patches, dim=0)
                        l_pix = F.mse_loss(decoded, gt_patches, reduction='sum')
                        v_sum += l_cos + args.lambda_pixel * l_pix
                        v_count += preds.numel()
                    val_loss += (v_sum / v_count).item() * imgs.size(0)
                    val_samples += imgs.size(0)
            print(f"         | Val Loss: {val_loss/val_samples:.6f}")

    # Save checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(student.state_dict(),                  "checkpoints/student.pth")
    torch.save(teacher.teacher_enc.state_dict(),      "checkpoints/teacher_enc.pth")
    print("Done. Models saved in ./checkpoints/")


if __name__ == "__main__":
    main()
