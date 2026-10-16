"""Dataset loading and preprocessing for known and OOD image sets."""
from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
datasets.CIFAR100.url = (
    "https://data.brainchip.com/dataset-mirror/cifar100/"
    "cifar-100-python.tar.gz"
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_transforms(image_size: int = 128):
    """Return train-time augmentation and deterministic evaluation transforms.

    CIFAR images are 32x32. They are resized for the ImageNet-pretrained ResNet-18.
    Augmentation is applied only to the training split.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def make_loaders(
    root: str = "data",
    image_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 2,
    val_fraction: float = 0.1,
    calibration_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Create deterministic train/validation/calibration splits from CIFAR-10 train data."""
    import numpy as np

    train_transform, eval_transform = make_transforms(image_size)
    train_dataset_aug = datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_transform
    )
    train_dataset_eval = datasets.CIFAR10(
        root=root, train=True, download=True, transform=eval_transform
    )

    n = len(train_dataset_aug)
    indices = np.random.default_rng(seed).permutation(n).tolist()
    val_n = max(1, int(n * val_fraction))
    calibration_n = max(1, int(n * calibration_fraction))
    if val_n + calibration_n >= n:
        raise ValueError("val_fraction + calibration_fraction must be less than 1")
    train_indices = indices[: n - val_n - calibration_n]
    val_indices = indices[n - val_n - calibration_n : n - calibration_n]
    calibration_indices = indices[n - calibration_n :]

    train_subset = Subset(train_dataset_aug, train_indices)
    val_subset = Subset(train_dataset_eval, val_indices)
    calibration_subset = Subset(train_dataset_eval, calibration_indices)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    train_loader = DataLoader(train_subset, shuffle=True, **common)
    val_loader = DataLoader(val_subset, shuffle=False, **common)
    calibration_loader = DataLoader(calibration_subset, shuffle=False, **common)
    return train_loader, val_loader, calibration_loader, list(train_dataset_aug.classes)


def make_eval_dataset(name: str, root: str = "data", image_size: int = 128):
    """Load a test split for the known set or one of the excluded OOD datasets."""
    _, eval_transform = make_transforms(image_size)
    key = name.lower()
    if key in {"cifar10", "known"}:
        return datasets.CIFAR10(
            root=root, train=False, download=True, transform=eval_transform
        )
    if key in {"cifar100", "ood_cifar100"}:
        return datasets.CIFAR100(
            root=root, train=False, download=True, transform=eval_transform
        )
    if key in {"svhn", "ood_svhn"}:
        return datasets.SVHN(
            root=root, split="test", download=True, transform=eval_transform
        )
    raise ValueError(f"Unsupported dataset '{name}'. Choose cifar10, cifar100, or svhn.")
