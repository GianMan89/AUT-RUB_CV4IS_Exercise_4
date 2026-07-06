from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class CIFAR10Bundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    raw_test_dataset: datasets.CIFAR10
    test_indices: list[int]
    class_names: list[str]
    eval_transform: transforms.Compose


def make_cifar10_bundle(
    data_root,
    train_samples=15000,
    val_samples=2000,
    test_samples=2000,
    batch_size=128,
    seed=42,
    num_workers=0,
) -> CIFAR10Bundle:
    data_root = Path(data_root)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_augmented = datasets.CIFAR10(
        root=data_root, train=True, transform=train_transform, download=False
    )
    train_evaluation = datasets.CIFAR10(
        root=data_root, train=True, transform=eval_transform, download=False
    )
    test_evaluation = datasets.CIFAR10(
        root=data_root, train=False, transform=eval_transform, download=False
    )
    raw_test = datasets.CIFAR10(
        root=data_root, train=False, transform=None, download=False
    )

    rng = np.random.default_rng(seed)
    train_permutation = rng.permutation(len(train_augmented))
    required_train = train_samples + val_samples
    if required_train > len(train_permutation):
        raise ValueError("Requested train/validation subset is too large.")

    train_indices = train_permutation[:train_samples].tolist()
    val_indices = train_permutation[train_samples:required_train].tolist()

    test_permutation = rng.permutation(len(test_evaluation))
    if test_samples > len(test_permutation):
        raise ValueError("Requested test subset is too large.")
    test_indices = test_permutation[:test_samples].tolist()

    train_loader = DataLoader(
        Subset(train_augmented, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(train_evaluation, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        Subset(test_evaluation, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return CIFAR10Bundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        raw_test_dataset=raw_test,
        test_indices=test_indices,
        class_names=list(train_augmented.classes),
        eval_transform=eval_transform,
    )
