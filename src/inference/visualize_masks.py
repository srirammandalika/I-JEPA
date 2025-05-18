# src/inference/visualize_masks.py

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.models.student import IJEPAStudent
from src.data.cifar10 import get_cifar10_dataloaders

def main():
    # Device
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load student (with patch decoder)
    student = IJEPAStudent().to(device)
    student.load_state_dict(
        torch.load("checkpoints/student.pth", map_location=device),
        strict=True
    )
    student.eval()

    # Data loader: small batch for visualization
    train_loader = get_cifar10_dataloaders(
        train_root="/Users/srirammandalika/Downloads/Minor/CIFAR-10 data/cifar10/train/",
        batch_size=4,
        img_size=32,
        max_per_class=10
    )
    images, _ = next(iter(train_loader))
    images = images.to(device)

    # 1) Run student to get preds_list and mask_idxs
    with torch.no_grad():
        preds_list, mask_idxs = student(images)

    # 2) Decode embeddings into pixel patches
    decoded_patches = []
    for emb in preds_list:
        # emb: [K, D]
        patches = student.decode_patches(emb)         # [K, C, p, p]
        # detach before numpy, then to CPU, then to numpy HWC
        patches = patches.detach().cpu().permute(0, 2, 3, 1).numpy()
        decoded_patches.append(patches)

    # 3) Stitch decoded patches into canvas
    B, C, H, W = images.shape
    p = student.patch_size
    grid = int(student.N ** 0.5)
    pred_recons = []
    for b in range(B):
        canvas = images[b].permute(1, 2, 0).cpu().numpy().copy()
        for idx, patch in zip(mask_idxs[b], decoded_patches[b]):
            i, j = divmod(idx, grid)
            y0, y1 = i * p, (i + 1) * p
            x0, x1 = j * p, (j + 1) * p
            canvas[y0:y1, x0:x1] = patch
        pred_recons.append(canvas)

    # 4) Plot 4-column figure
    fig, axes = plt.subplots(B, 4, figsize=(12, 3 * B))
    for b in range(B):
        orig = images[b].permute(1, 2, 0).cpu().numpy()

        # Context masked
        masked = orig.copy()
        for idx in mask_idxs[b]:
            i, j = divmod(idx, grid)
            masked[i * p:(i + 1) * p, j * p:(j + 1) * p] = 0.5

        # Plot columns
        axes[b, 0].imshow(orig)
        axes[b, 0].set_title("Original")
        axes[b, 0].axis("off")

        axes[b, 1].imshow(masked)
        axes[b, 1].set_title("Context Masked")
        axes[b, 1].axis("off")

        axes[b, 2].imshow(pred_recons[b])
        axes[b, 2].set_title("Predicted Reconstruction")
        axes[b, 2].axis("off")

        # Mask positions outline
        axes[b, 3].imshow(orig)
        for idx in mask_idxs[b]:
            i, j = divmod(idx, grid)
            y0, y1 = i * p, (i + 1) * p
            x0, x1 = j * p, (j + 1) * p
            rect = Rectangle((x0, y0), p, p, fill=False, linewidth=2, edgecolor='lime')
            axes[b, 3].add_patch(rect)
        axes[b, 3].set_title("Mask Positions")
        axes[b, 3].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
