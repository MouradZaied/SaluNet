import numpy as np
import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from torchvision.transforms import TrivialAugmentWide

from salunet.data.samplers import RepeatedAugmentationSampler


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        batch_size: int = 180,
        num_workers: int = 12,
        train_crop_size: int = 224,
        val_resize_size: int = 256,
        random_erase_prob: float = 0.5,
        repeated_aug_reps: int = 1,
        pin_memory_device: str | None = None,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_crop_size = train_crop_size
        self.val_resize_size = val_resize_size
        self.random_erase_prob = random_erase_prob
        self.repeated_aug_reps = repeated_aug_reps
        self.pin_memory_device = pin_memory_device
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def setup(self, stage=None):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.train_crop_size),
            transforms.RandomHorizontalFlip(),
            TrivialAugmentWide(num_magnitude_bins=31),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
            transforms.RandomErasing(p=self.random_erase_prob),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(self.val_resize_size),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        self.train_dataset = ImageFolder(self.train_dir, transform=train_transform)
        self.val_dataset = ImageFolder(self.val_dir, transform=val_transform)

    def _loader_kwargs(self):
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 4
        if self.pin_memory_device:
            kwargs["pin_memory_device"] = self.pin_memory_device
        return kwargs

    def train_dataloader(self):
        sampler = RepeatedAugmentationSampler(self.train_dataset, num_repeats=self.repeated_aug_reps, shuffle=True)
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            worker_init_fn=worker_init_fn,
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **self._loader_kwargs(),
        )
