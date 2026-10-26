import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl


class Cutout:
    """Improved Cutout from the original CIFAR notebook."""

    def __init__(self, n_holes: int = 1, length: int = 16, p: float = 0.5):
        self.n_holes = int(n_holes)
        self.length = int(length)
        self.p = float(p)

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)
        for _ in range(self.n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0
        return img * mask.expand_as(img)


class CIFARDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str = "cifar100",
        data_dir: str = "./data",
        batch_size: int = 512,
        num_workers: int = 8,
        use_auto_augment: bool = True,
        use_rand_augment: bool = False,
        use_cutout: bool = True,
        cutout_n_holes: int = 1,
        cutout_length: int = 16,
        cutout_p: float = 0.5,
        fallback_aug: bool = False,
        pin_memory_device: str | None = None,
    ):
        super().__init__()
        self.dataset = dataset.lower()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_auto_augment = use_auto_augment
        self.use_rand_augment = use_rand_augment
        self.use_cutout = use_cutout
        self.cutout_n_holes = cutout_n_holes
        self.cutout_length = cutout_length
        self.cutout_p = cutout_p
        self.fallback_aug = fallback_aug
        self.pin_memory_device = pin_memory_device
        if self.dataset == "cifar10":
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2470, 0.2435, 0.2616)
            self.dataset_cls = torchvision.datasets.CIFAR10
        elif self.dataset == "cifar100":
            self.mean = (0.5071, 0.4867, 0.4408)
            self.std = (0.2675, 0.2565, 0.2761)
            self.dataset_cls = torchvision.datasets.CIFAR100
        else:
            raise ValueError(f"Unsupported CIFAR dataset: {dataset}")

    def setup(self, stage=None):
        self.train_dataset = self.dataset_cls(root=self.data_dir, train=True, download=True, transform=self._train_transforms())
        self.val_dataset = self.dataset_cls(root=self.data_dir, train=False, download=True, transform=self._val_transforms())

    def _train_transforms(self):
        t = [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        if self.use_auto_augment:
            t.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
        elif self.use_rand_augment:
            t.append(transforms.RandAugment(num_ops=2, magnitude=9))
        elif self.fallback_aug:
            # Fallback branch preserved from the original notebook.
            t.extend(
                [
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
                ]
            )
        t += [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        if self.use_cutout:
            t.append(Cutout(n_holes=self.cutout_n_holes, length=self.cutout_length, p=self.cutout_p))
        return transforms.Compose(t)

    def _val_transforms(self):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])

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
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            **self._loader_kwargs(),
        )
