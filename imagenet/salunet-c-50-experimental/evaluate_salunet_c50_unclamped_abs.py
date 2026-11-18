#!/usr/bin/env python3
"""Evaluate the experimental unclamped abs-positive SaluNet-C-50 variant.

Supports best_main.pth, best_ema.pth, epoch-boundary last_resume.pth,
and legacy model_state_dict/state_dict exports. The architecture matches
train_salunet_c50_unclamped_abs.py. Use --weights main or ema to select
weights from a full resume checkpoint; auto prefers EMA in that format.
Full resume files contain Python/NumPy RNG objects and require the explicit
--trusted-checkpoint option. Only use that option for files you trust.
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SALU(nn.Module):
    """Historical SALU with positive log-parameterized coefficients."""

    def __init__(self, a_init=1.0, b_init=0.1, name=""):
        super().__init__()
        self.loga = nn.Parameter(torch.tensor(math.log(a_init)))
        self.logb = nn.Parameter(torch.tensor(math.log(b_init)))
        self.name = name

    def forward(self, x):
        # Identical to the training implementation: SALU is evaluated in FP32.
        x = x.float()
        a = torch.exp(self.loga)
        b = torch.exp(self.logb)
        denom = torch.sqrt(1.0 + a * b * x.square())
        return (a * x) / denom


class PositiveGateSALU(nn.Module):
    """SALU inside SWALU, with direct parameters mapped through abs + eps."""

    def __init__(self, a=1.0, b=1.0, eps=1e-6, name=""):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.b = nn.Parameter(torch.tensor(float(b)))
        self.eps = eps
        self.name = name

    def forward(self, x):
        x = x.float()
        a = self.a.abs() + self.eps
        b = self.b.abs() + self.eps
        return (a * x) / torch.sqrt(1.0 + a * b * x.square())

    def get_params(self):
        return {
            f"swalu/{self.name}/a": (self.a.abs() + self.eps).item(),
            f"swalu/{self.name}/b": (self.b.abs() + self.eps).item(),
        }


class SWALU(nn.Module):
    """Experimental unclamped abs-positive SWALU; preserve the reference FP32 output."""

    def __init__(self, a=1.0, b=1.0, name=""):
        super().__init__()
        self.salu = PositiveGateSALU(a=a, b=b, name=name)
        self.name = name

    def forward(self, x):
        return 0.5 * x * (1.0 + self.salu(x))

    def get_params(self):
        return self.salu.get_params()




def build_salunet_c50():
    """Reconstruct the exact 53-SALU/17-SWALU ResNet-50 architecture."""
    model = models.resnet50(weights=None)

    # Active initialization block in the supplied 150-epoch training script.
    special_bn = {
        1: {"a": 5.5, "b": 1e-5},
        5: {"a": 2.0, "b": 0.1},
        15: {"a": 1.5, "b": 0.01},
        28: {"a": 0.8, "b": 0.001},
        47: {"a": 1.0, "b": 0.001},
    }

    bn_index = 0
    salu_count = 0

    def replace_bn(module):
        nonlocal bn_index, salu_count
        for child_name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                bn_index += 1
                init = special_bn.get(bn_index, {"a": 1.0, "b": 0.1})
                replacement_name = (
                    f"salu_downsample_{bn_index}"
                    if bn_index in special_bn
                    else f"salu_bn_{bn_index}"
                )
                setattr(
                    module,
                    child_name,
                    SALU(
                        a_init=init["a"],
                        b_init=init["b"],
                        name=replacement_name,
                    ),
                )
                salu_count += 1
            else:
                replace_bn(child)

    replace_bn(model)

    gate_count = 0

    def replace_relu(module):
        nonlocal gate_count
        for child_name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                gate_count += 1
                setattr(
                    module,
                    child_name,
                    SWALU(name=f"swalu_{gate_count}"),
                )
            else:
                replace_relu(child)

    replace_relu(model)
    if (salu_count, gate_count) != (53, 17):
        raise RuntimeError(
            f"Unexpected architecture: {salu_count} SALU and "
            f"{gate_count} unclamped abs-positive SWALU; expected 53 and 17."
        )
    print(
        f"Architecture: {salu_count} BN -> SALU, "
        f"{gate_count} ReLU -> unclamped abs-positive SWALU"
    )
    return model


def clean_state_dict(state_dict):
    """Accept plain, DDP, and torch.compile-prefixed checkpoints."""
    cleaned = {}
    prefixes = ("module.", "_orig_mod.")
    for original_key, value in state_dict.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        cleaned[key] = value
    return cleaned


def select_state_dict(checkpoint, weights="auto"):
    """Select weights without silently substituting main for EMA, or vice versa."""
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise TypeError("Checkpoint must be a nonempty dictionary")

    # Full epoch-boundary resume format produced by the experimental trainer.
    if "model" in checkpoint and "ema" in checkpoint:
        if weights == "selected":
            raise ValueError("A resume file has no selected export; use --weights main or ema")
        kind = "ema" if weights == "auto" else weights
        key = "ema" if kind == "ema" else "model"
        state_dict = checkpoint[key]
        epoch = checkpoint.get("next_epoch", "unknown")
        score = checkpoint.get(f"best_top1_{kind}")
        score_label = "Best recorded Top-1 (not necessarily these current weights)"
    else:
        if "model_state_dict" in checkpoint:
            key = "model_state_dict"
            state_dict = checkpoint[key]
        elif "state_dict" in checkpoint:
            key = "state_dict"
            state_dict = checkpoint[key]
        elif all(torch.is_tensor(value) for value in checkpoint.values()):
            key = "plain_state_dict"
            state_dict = checkpoint
        else:
            raise KeyError("No supported weights found in checkpoint")
        kind = checkpoint.get("kind", "unknown") if key != "plain_state_dict" else "unknown"
        if weights in ("main", "ema") and kind != weights:
            raise ValueError(
                f"Requested {weights} but this export has kind={kind!r}. "
                "Use --weights selected for a legacy export with unspecified kind."
            )
        epoch = checkpoint.get("epoch", "unknown")
        score = checkpoint.get("top1", checkpoint.get("best_top1"))
        score_label = "Saved Top-1"

    if not isinstance(state_dict, dict) or not state_dict:
        raise TypeError(f"{key} must be a nonempty state dictionary")
    if not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state_dict.items()):
        raise TypeError(f"{key} contains invalid parameter entries")
    return state_dict, key, kind, epoch, score, score_label


def load_checkpoint(model, checkpoint_path, weights="auto", trusted_checkpoint=False):
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=not trusted_checkpoint
        )
    except TypeError as exc:
        if not trusted_checkpoint:
            raise RuntimeError(
                "This PyTorch version does not support weights_only loading. "
                "Upgrade PyTorch or use --trusted-checkpoint for your own file."
            ) from exc
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict, key, kind, epoch, score, score_label = select_state_dict(checkpoint, weights)
    model.load_state_dict(clean_state_dict(state_dict), strict=True)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Loaded state: {key}; weights kind: {kind}")
    print(f"Completed epoch: {epoch}")
    if score is not None:
        print(f"{score_label}: {float(score):.3f}%")


def interpolation_from_name(name):
    return {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "lanczos": InterpolationMode.LANCZOS,
    }[name]


@torch.inference_mode()
def evaluate(model, loader, device, use_amp):
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0

    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(
            device=device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            logits = model(images)

        top5 = logits.topk(5, dim=1, largest=True, sorted=True).indices
        correct = top5.eq(targets[:, None])
        correct1 += correct[:, :1].sum().item()
        correct5 += correct.sum().item()
        total += targets.numel()

        if batch_index % 50 == 0 or batch_index == len(loader):
            print(
                f"[{batch_index:4d}/{len(loader):4d}] "
                f"Top-1={100.0 * correct1 / total:.3f}%  "
                f"Top-5={100.0 * correct5 / total:.3f}%"
            )

    if total == 0:
        raise ValueError("Evaluation loader contains no images")
    return 100.0 * correct1 / total, 100.0 * correct5 / total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--weights", choices=("auto", "selected", "main", "ema"), default="auto",
        help="auto uses an export's weights, or EMA from a full resume file.",
    )
    parser.add_argument(
        "--trusted-checkpoint", action="store_true",
        help="Allow Python/NumPy objects in a trusted full resume checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="ImageNet val directory containing the 1000 class folders.",
    )
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=180)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument(
        "--interpolation",
        choices=("bilinear", "bicubic", "lanczos"),
        default="bilinear",
        help="Online resize interpolation; the training code used bilinear.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.data_root.is_dir():
        raise NotADirectoryError(args.data_root)
    if args.crop_size > args.resize_size:
        raise ValueError("crop-size must not exceed resize-size")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    transform = transforms.Compose(
        [
            transforms.Resize(
                args.resize_size,
                interpolation=interpolation_from_name(args.interpolation),
            ),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = ImageFolder(args.data_root, transform=transform)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(**loader_kwargs)

    print(f"Validation images: {len(dataset)}")
    print(f"Autocast: {'FP16' if not args.no_amp and device.type == 'cuda' else 'disabled (FP32)'}")
    print(
        f"Evaluation: resize={args.resize_size}, crop={args.crop_size}, "
        f"interpolation={args.interpolation}, device={device}"
    )

    model = build_salunet_c50()
    load_checkpoint(model, args.checkpoint, args.weights, args.trusted_checkpoint)
    model = model.to(device=device, memory_format=torch.channels_last)

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch version")
        model = torch.compile(model, mode=args.compile_mode)

    top1, top5 = evaluate(model, loader, device, use_amp=not args.no_amp)
    print("\nFinal ImageNet-1K validation result")
    print(f"Top-1: {top1:.3f}%")
    print(f"Top-5: {top5:.3f}%")


if __name__ == "__main__":
    main()
