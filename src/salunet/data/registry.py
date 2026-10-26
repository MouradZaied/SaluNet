from salunet.data.cifar import CIFARDataModule
from salunet.data.imagenet import ImageNetDataModule


def build_datamodule(cfg: dict):
    name = cfg.get("name", cfg.get("dataset", "")).lower()
    if name in {"cifar10", "cifar100"} and cfg.get("recipe") == "vit_cifar":
        from salunet.data.vit_cifar import ViTCIFARDataModule
        return ViTCIFARDataModule(
            dataset=name, data_dir=cfg.get("data_dir", "./data"),
            batch_size=cfg.get("batch_size", 128), num_workers=cfg.get("num_workers", 4),
            val_batch_size=cfg.get("val_batch_size", 1024),
            use_auto_augment=cfg.get("use_auto_augment", True),
            use_rand_augment=cfg.get("use_rand_augment", False), use_cutout=False,
        )
    if name in {"cifar10", "cifar100"}:
        return CIFARDataModule(
            dataset=name,
            data_dir=cfg.get("data_dir", "./data"),
            batch_size=cfg.get("batch_size", 512),
            num_workers=cfg.get("num_workers", 8),
            use_auto_augment=cfg.get("use_auto_augment", True),
            use_rand_augment=cfg.get("use_rand_augment", False),
            use_cutout=cfg.get("use_cutout", True),
            cutout_n_holes=cfg.get("cutout_n_holes", 1),
            cutout_length=cfg.get("cutout_length", 16),
            cutout_p=cfg.get("cutout_p", 0.5),
            fallback_aug=cfg.get("fallback_aug", False),
            pin_memory_device=cfg.get("pin_memory_device"),
        )
    if name in {"imagenet", "imagenet1k"}:
        return ImageNetDataModule(
            train_dir=cfg["train_dir"],
            val_dir=cfg["val_dir"],
            batch_size=cfg.get("batch_size", 180),
            num_workers=cfg.get("num_workers", 12),
            train_crop_size=cfg.get("train_crop_size", 224),
            val_resize_size=cfg.get("val_resize_size", 256),
            random_erase_prob=cfg.get("random_erase_prob", 0.5),
            repeated_aug_reps=cfg.get("repeated_aug_reps", 1),
            pin_memory_device=cfg.get("pin_memory_device"),
        )
    raise ValueError(f"Unknown datamodule/dataset: {name}")
