# src/data/cifar10.py

import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class CIFAR10Folder(Dataset):
    """
    CIFAR-10 Dataset loader from a folder structure:
      root/
        class0/
          0.png, 1.jpg, ...
        class1/
          0.png, 1.jpg, ...
        ...
    Optionally limits the number of images per class.
    Only files with extensions .png, .jpg, .jpeg are considered.
    """
    IMG_EXTS = ('.png', '.jpg', '.jpeg')

    def __init__(
        self,
        root: str,
        transform=None,
        max_per_class: int = None,
    ):
        super().__init__()
        self.root = root
        self.transform = transform

        # gather all files then filter by extension
        all_paths = []
        for cls in os.listdir(root):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(self.IMG_EXTS):
                    all_paths.append(os.path.join(cls_dir, fname))

        if not all_paths:
            raise RuntimeError(f"No images found in {root} with extensions {self.IMG_EXTS}")

        # derive class names
        classes = sorted({os.path.basename(os.path.dirname(p)) for p in all_paths})
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        # group by class and optionally limit per class
        grouped = {cls: [] for cls in classes}
        for p in all_paths:
            cls = os.path.basename(os.path.dirname(p))
            grouped[cls].append(p)

        samples = []
        for cls, img_list in grouped.items():
            img_list.sort()
            if max_per_class:
                img_list = img_list[: max_per_class]
            label = self.class_to_idx[cls]
            samples.extend([(p, label) for p in img_list])

        self.samples = samples
        print(f"Loaded {len(self.samples)} images from {len(classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_cifar10_dataloaders(
    train_root: str,
    val_root: str = None,
    batch_size: int = 64,
    img_size: int = 32,
    max_per_class: int = None,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Returns: train_loader, val_loader (if val_root provided)
    """
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])

    train_ds = CIFAR10Folder(
        root=train_root,
        transform=transform,
        max_per_class=max_per_class,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if val_root:
        val_ds = CIFAR10Folder(
            root=val_root,
            transform=transform,
            max_per_class=max_per_class,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader

    return train_loader
