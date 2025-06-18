import os
from PIL import Image
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Dataset(Dataset):
    def __init__(self, image_paths: List[str], mask_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Binarize the mask
        mask = (mask > 0).float()

        return image, mask


class DataModule:
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        input_size: Tuple[int, int] = (256, 256),
        batch_size: int = 8,
        num_workers: int = 2,
        val_split: float = 0.2,
        seed: int = 42
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
        ])

    def setup(self):
        image_paths = sorted([os.path.join(self.image_dir, fname) for fname in os.listdir(self.image_dir)])
        mask_paths = sorted([os.path.join(self.mask_dir, fname) for fname in os.listdir(self.mask_dir)])

        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            image_paths, mask_paths, test_size=self.val_split, random_state=self.seed
        )

        self.train_dataset = Dataset(train_imgs, train_masks, transform=self.transform)
        self.val_dataset = Dataset(val_imgs, val_masks, transform=self.transform)

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers)
        return train_loader, val_loader

    def get_config(self) -> dict:
        return {
            "input_size": self.input_size,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "val_split": self.val_split,
            "seed": self.seed
        }
