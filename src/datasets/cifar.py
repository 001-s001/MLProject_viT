from typing import Any

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def build_cifar_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, list[str]]:
    name = str(config.get("name", "cifar10")).lower()
    data_dir = str(config.get("data_dir", "./data"))
    batch_size = int(config.get("batch_size", 128))
    num_workers = int(config.get("num_workers", 2))
    val_ratio = float(config.get("val_ratio", 0.1))
    image_size = int(config.get("image_size", 32))

    mean, std = _stats(name)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    dataset_cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    train_full = dataset_cls(root=data_dir, train=True, download=True, transform=train_transform)
    val_full = dataset_cls(root=data_dir, train=True, download=True, transform=eval_transform)
    class_names = list(train_full.classes)

    val_size = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    train_indices, val_indices = random_split(range(len(train_full)), [train_size, val_size], generator=generator)

    train_set = torch.utils.data.Subset(train_full, train_indices.indices)
    val_set = torch.utils.data.Subset(val_full, val_indices.indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, class_names


def _stats(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if name == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    if name == "cifar100":
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    raise ValueError("dataset.name must be cifar10 or cifar100")

