"""
Dataset classes for FSDNet project.
Handles loading of real and fake images with proper preprocessing.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import logging
from typing import Tuple, List, Optional, Dict
import numpy as np
import random

logger = logging.getLogger(__name__)

class FSDNetDataset(Dataset):
    """
    Dataset class for FSDNet project.
    Loads real and fake images with proper preprocessing.
    """
    
    def __init__(self, 
                 real_dir: str,
                 fake_dir: str,
                 image_size: int = 224,
                 transform: Optional[transforms.Compose] = None,
                 split: str = "train"):
        """
        Initialize FSDNet dataset.
        
        Args:
            real_dir: Directory containing real images
            fake_dir: Directory containing fake images
            image_size: Size to resize images to
            transform: Optional transforms to apply
            split: Dataset split ('train', 'val', 'test')
        """
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.image_size = image_size
        self.split = split
        
        # Support optional split subdirectories: real_dir/{train,val,test}, fake_dir/{train,val,test}
        real_dir_split = os.path.join(real_dir, self.split)
        fake_dir_split = os.path.join(fake_dir, self.split)
        real_src = real_dir_split if os.path.isdir(real_dir_split) else real_dir
        fake_src = fake_dir_split if os.path.isdir(fake_dir_split) else fake_dir

        # Get image paths
        self.real_paths = self._get_image_paths(real_src)
        self.fake_paths = self._get_image_paths(fake_src)
        
        # Create labels (0 for real, 1 for fake)
        self.images = self.real_paths + self.fake_paths
        self.labels = [0] * len(self.real_paths) + [1] * len(self.fake_paths)
        
        # Setup transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
        
        logger.info(f"FSDNet {split} dataset: {len(self.real_paths)} real, {len(self.fake_paths)} fake images")
    
    def _get_image_paths(self, directory: str) -> List[str]:
        """Get list of image file paths from directory."""
        if not os.path.isdir(directory):
            return []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = []
        for root, _, files in os.walk(directory):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_exts:
                    paths.append(os.path.join(root, f))
        return paths
    
    def _get_default_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int):
        path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class CombinedImageVideoDataset(Dataset):
    """
    Combined dataset for images and video-extracted frames.
    Expects directory layout:
      - image reals/fakes from provided dirs (optionally with split subdirs)
      - video frames under video_frames_dir/{train,val,test}/{real,fake}/...
    """
    def __init__(self,
                 image_real_dir: str,
                 image_fake_dir: str,
                 video_frames_dir: Optional[str],
                 split: str,
                 image_size: int,
                 transform: Optional[transforms.Compose] = None):
        self.split = split
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.paths: List[str] = []
        self.labels: List[int] = []

        # Images
        def collect(dir_root: str, label: int):
            if not dir_root:
                return
            dir_split = os.path.join(dir_root, split)
            src = dir_split if os.path.isdir(dir_split) else dir_root
            for p in FSDNetDataset._get_image_paths(self, src):
                self.paths.append(p)
                self.labels.append(label)
        collect(image_real_dir, 0)
        collect(image_fake_dir, 1)

        # Video frames
        if video_frames_dir:
            real_frames = os.path.join(video_frames_dir, split, 'real')
            fake_frames = os.path.join(video_frames_dir, split, 'fake')
            for p in FSDNetDataset._get_image_paths(self, real_frames):
                self.paths.append(p)
                self.labels.append(0)
            for p in FSDNetDataset._get_image_paths(self, fake_frames):
                self.paths.append(p)
                self.labels.append(1)

        logger.info(f"Combined {split} dataset: total={len(self.paths)} (real={self.labels.count(0)}, fake={self.labels.count(1)})")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label


def _stratified_split_indices(labels: List[int], splits: Tuple[float, float, float]) -> Tuple[List[int], List[int], List[int]]:
    idxs = list(range(len(labels)))
    real_idxs = [i for i in idxs if labels[i] == 0]
    fake_idxs = [i for i in idxs if labels[i] == 1]
    def split_group(group: List[int], ratios: Tuple[float, float, float]):
        random.shuffle(group)
        n = len(group)
        n_train = int(ratios[0] * n)
        n_val = int(ratios[1] * n)
        train = group[:n_train]
        val = group[n_train:n_train+n_val]
        test = group[n_train+n_val:]
        return train, val, test
    train_r, val_r, test_r = split_group(real_idxs, splits)
    train_f, val_f, test_f = split_group(fake_idxs, splits)
    train = train_r + train_f
    val = val_r + val_f
    test = test_r + test_f
    random.shuffle(train); random.shuffle(val); random.shuffle(test)
    return train, val, test


def create_combined_data_loaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders that combine images and optional video frames, with a 70/15/15 stratified split
    applied across the combined pool when explicit split dirs are absent.
    """
    use_frames = getattr(config.data, 'use_video_frames', False)
    frames_dir = config.data.video_frames_dir if use_frames else None

    transform = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # If both images and frames are already split into subfolders, build loaders directly
    has_explicit_splits = (
        os.path.isdir(os.path.join(config.data.real_images_dir, 'train')) and
        os.path.isdir(os.path.join(config.data.fake_images_dir, 'train')) and
        (not frames_dir or os.path.isdir(os.path.join(frames_dir, 'train')))
    )

    if has_explicit_splits:
        train_dataset = CombinedImageVideoDataset(
            config.data.real_images_dir,
            config.data.fake_images_dir,
            frames_dir,
            split='train',
            image_size=config.data.image_size,
            transform=transform
        )
        val_dataset = CombinedImageVideoDataset(
            config.data.real_images_dir,
            config.data.fake_images_dir,
            frames_dir,
            split='val',
            image_size=config.data.image_size,
            transform=transform
        )
        test_dataset = CombinedImageVideoDataset(
            config.data.real_images_dir,
            config.data.fake_images_dir,
            frames_dir,
            split='test',
            image_size=config.data.image_size,
            transform=transform
        )
    else:
        # Build a single combined pool and split stratified
        pool = CombinedImageVideoDataset(
            config.data.real_images_dir,
            config.data.fake_images_dir,
            frames_dir,
            split='train',  # ignored; paths resolve internally
            image_size=config.data.image_size,
            transform=transform
        )
        train_idx, val_idx, test_idx = _stratified_split_indices(pool.labels, (config.data.train_split, config.data.val_split, config.data.test_split))

        class Subset(Dataset):
            def __init__(self, base: CombinedImageVideoDataset, indices: List[int]):
                self.base = base
                self.indices = indices
                # Expose labels for efficient sampling without image I/O
                self.labels = [base.labels[i] for i in indices]
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, i):
                return self.base[self.indices[i]]

        train_dataset = Subset(pool, train_idx)
        val_dataset = Subset(pool, val_idx)
        test_dataset = Subset(pool, test_idx)

    def make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        if getattr(config.training, 'sampler_balance', False) and shuffle:
            # WeightedRandomSampler to balance classes without touching image I/O
            if hasattr(ds, 'labels'):
                labels_np = np.array(ds.labels)
            else:
                # Fallback (avoid heavy transforms): use a small probe then default uniform
                labels_np = np.zeros(len(ds), dtype=np.int64)
            class_sample_count = np.bincount(labels_np, minlength=2)
            class_weights = 1.0 / np.maximum(class_sample_count, 1)
            sample_weights = class_weights[labels_np]
            sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(ds), replacement=True)
            return DataLoader(
                ds,
                batch_size=config.data.batch_size,
                sampler=sampler,
                num_workers=config.data.num_workers,
                pin_memory=True,
                persistent_workers=getattr(config.data, 'persistent_workers', False),
                prefetch_factor=getattr(config.data, 'prefetch_factor', None)
            )
        else:
            return DataLoader(
                ds,
                batch_size=config.data.batch_size,
                shuffle=shuffle,
                num_workers=config.data.num_workers,
                pin_memory=True,
                persistent_workers=getattr(config.data, 'persistent_workers', False),
                prefetch_factor=getattr(config.data, 'prefetch_factor', None)
            )

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)
    test_loader = make_loader(test_dataset, shuffle=False)

    return train_loader, val_loader, test_loader


def create_data_loaders(config, split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
    """
    Existing loader for image-only flow; retained for backward-compatibility.
    """
    train_ratio, val_ratio, test_ratio = split_ratios

    has_explicit_splits = (
        os.path.isdir(os.path.join(config.data.real_images_dir, 'train')) and
        os.path.isdir(os.path.join(config.data.fake_images_dir, 'train')) and
        os.path.isdir(os.path.join(config.data.real_images_dir, 'val')) and
        os.path.isdir(os.path.join(config.data.fake_images_dir, 'val')) and
        os.path.isdir(os.path.join(config.data.real_images_dir, 'test')) and
        os.path.isdir(os.path.join(config.data.fake_images_dir, 'test')) and
        os.path.isdir(os.path.join(config.data.fake_images_dir, 'test'))
    )

    if has_explicit_splits:
        train_dataset = FSDNetDataset(
            real_dir=config.data.real_images_dir,
            fake_dir=config.data.fake_images_dir,
            image_size=config.data.image_size,
            split="train"
        )
        val_dataset = FSDNetDataset(
            real_dir=config.data.real_images_dir,
            fake_dir=config.data.fake_images_dir,
            image_size=config.data.image_size,
            split="val"
        )
        test_dataset = FSDNetDataset(
            real_dir=config.data.real_images_dir,
            fake_dir=config.data.fake_images_dir,
            image_size=config.data.image_size,
            split="test"
        )
    else:
        full_dataset = FSDNetDataset(
            real_dir=config.data.real_images_dir,
            fake_dir=config.data.fake_images_dir,
            image_size=config.data.image_size,
            split="train"  # placeholder label; random split below
        )

        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=getattr(config.data, 'persistent_workers', False),
        prefetch_factor=getattr(config.data, 'prefetch_factor', None)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=getattr(config.data, 'persistent_workers', False),
        prefetch_factor=getattr(config.data, 'prefetch_factor', None)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=getattr(config.data, 'persistent_workers', False),
        prefetch_factor=getattr(config.data, 'prefetch_factor', None)
    )

    return train_loader, val_loader, test_loader
