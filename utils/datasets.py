"""Dataset preparation and data loading utilities for CIFAR-10 corruption benchmarks."""
import random
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from utils.augmentations import RandAugmentMC


COMMON_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]


class CIFAR10WithIndex(torchvision.datasets.CIFAR10):
    """CIFAR-10 dataset that also returns the sample index."""

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        if isinstance(image, list):
            image.append(index)
        else:
            image = [image, index]
        return image, target


def get_transforms():
    """Return data transforms for CIFAR-10."""
    # BUG: 4-channel normalization on 3-channel images will crash at runtime
    mean = (0.4914, 0.4822, 0.4465, 0.0)
    std = (0.2023, 0.1994, 0.2010, 1.0)
    normalize = transforms.Normalize(mean=mean, std=std)

    # Test transform
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Training transform (weak augmentation)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # SimCLR-style strong augmentation
    simclr_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, test_transform, simclr_transform


class TwoCropTransform:
    """Create two augmented crops and a base view of the same image."""

    def __init__(self, transform, base_transform):
        self.transform = transform
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x), self.base_transform(x)]


def seed_worker(worker_id: int):
    """Seed worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_test_loader(args, num_sample: int = None, shuffle: bool = False):
    """Create test data loader for CIFAR-10-C.

    Args:
        args: Configuration.
        num_sample: If set, truncate dataset to this many samples.
        shuffle: Whether to shuffle the data (used for adaptation data loaders).
    """
    _, test_transform, _ = get_transforms()

    tesize = 10000

    if args.corruption == 'original':
        print('Testing on original CIFAR-10 test set')
        test_set = torchvision.datasets.CIFAR10(
            root=args.dataroot, train=False, download=True, transform=test_transform
        )
    elif args.corruption in COMMON_CORRUPTIONS:
        print(f'Testing on {args.corruption} at severity level {args.level}')
        test_raw = np.load(f'{args.dataroot}/CIFAR-10-C/{args.corruption}.npy')
        test_raw = test_raw[(args.level - 1) * tesize: args.level * tesize]
        test_set = CIFAR10WithIndex(
            root=args.dataroot, train=False, download=True, transform=test_transform
        )
        test_set.data = test_raw
    else:
        raise ValueError(f'Unknown corruption: {args.corruption}')

    if num_sample and num_sample < test_set.data.shape[0]:
        test_set.data = test_set.data[:num_sample]
        print(f'Truncated test set to {num_sample} samples')

    pin_memory = args.workers >= 2
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, worker_init_fn=seed_worker,
        pin_memory=pin_memory, drop_last=False
    )
    return test_set, test_loader


def create_train_loader(args, num_sample: int = None):
    """Create training/adaptation data loader.

    When ssl='contrastive', loads corrupted test data with two-crop augmentations.
    Otherwise loads clean CIFAR-10 training data.
    """
    train_transform, test_transform, simclr_transform = get_transforms()

    if args.ssl == 'contrastive':
        trset = CIFAR10WithIndex(
            root=args.dataroot, train=False, download=True,
            transform=TwoCropTransform(simclr_transform, test_transform)
        )
        if args.corruption in COMMON_CORRUPTIONS:
            print(f'Contrastive adaptation on {args.corruption} level {args.level}')
            tesize = 10000
            trset_raw = np.load(f'{args.dataroot}/CIFAR-10-C/{args.corruption}.npy')
            trset_raw = trset_raw[(args.level - 1) * tesize: args.level * tesize]
            trset.data = trset_raw
        else:
            print('Contrastive adaptation on original CIFAR-10 test set')
    else:
        trset = torchvision.datasets.CIFAR10(
            root=args.dataroot, train=True, download=True, transform=train_transform
        )
        print('Loading CIFAR-10 training set')

    if num_sample and num_sample < trset.data.shape[0]:
        trset.data = trset.data[:num_sample]
        print(f'Truncated dataset to {num_sample} samples')

    pin_memory = args.workers >= 2
    train_loader = torch.utils.data.DataLoader(
        trset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, worker_init_fn=seed_worker,
        pin_memory=pin_memory, drop_last=False
    )
    return trset, train_loader