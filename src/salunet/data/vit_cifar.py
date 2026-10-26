"""CIFAR data recipe from the supplied Transformer training pipeline."""
import torch
from torchvision import transforms
from salunet.data.cifar import CIFARDataModule
from salunet.transforms.vit_autoaugment import ViTCIFAR10Policy

class ViTCIFARDataModule(CIFARDataModule):
    def __init__(self, val_batch_size=1024, **kwargs):
        super().__init__(**kwargs)
        self.val_batch_size = val_batch_size

    def _train_transforms(self):
        operations = [transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip()]
        if self.use_auto_augment:
            operations.append(ViTCIFAR10Policy())
        elif self.use_rand_augment:
            operations.append(transforms.RandAugment(num_ops=2, magnitude=9))
        operations.extend([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        return transforms.Compose(operations)

    def _loader_kwargs(self):
        return dict(num_workers=self.num_workers, pin_memory=torch.cuda.is_available())

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
            batch_size=self.batch_size, shuffle=True, drop_last=False,
            **self._loader_kwargs())

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
            batch_size=self.val_batch_size, shuffle=False, **self._loader_kwargs())
