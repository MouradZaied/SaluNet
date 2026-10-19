#!/usr/bin/env python3
"""Evaluate a SaluNet checkpoint on CIFAR-10/100.

Run this script from the repository root.  The YAML supplies the full training
recipe; --normalization and --activation explicitly select the architecture
that must match the checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    """Find the repository root containing src/salunet."""
    for candidate in (start, *start.parents):
        if (candidate / "src" / "salunet").is_dir():
            return candidate
    raise RuntimeError(
        "Unable to find the SaluNet repository root. Expected a "
        "src/salunet directory in the script directory or one of its parents."
    )


ROOT = find_repo_root(SCRIPT_DIR)
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from salunet.data.registry import build_datamodule
from salunet.lightning.classification_module import ClassificationModule
from salunet.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training YAML used by the run.")
    parser.add_argument("--checkpoint", required=True, help=".pth or .ckpt checkpoint.")
    parser.add_argument("--normalization", choices=("salu", "bn", "ln", "rmsnorm", "dyt"), required=True)
    parser.add_argument("--activation", choices=("relu", "gelu", "swalu", "galu"), required=True)
    parser.add_argument(
        "--weights",
        choices=("auto", "selected", "raw", "ema"),
        default="auto",
        help="Weights stored in the checkpoint (default: auto).",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def choose_state_dict(checkpoint: dict, choice: str) -> tuple[dict, str]:
    if choice == "raw":
        candidates = ("raw_model_state_dict", "model_state_dict", "state_dict")
    elif choice == "ema":
        candidates = ("ema_model_state_dict", "model_state_dict")
    elif choice == "selected":
        candidates = ("model_state_dict",)
    else:
        # best_mp_model.pth and best_ema_model.pth both expose the weights to
        # evaluate as model_state_dict.  last.ckpt exposes state_dict.
        candidates = ("model_state_dict", "state_dict", "raw_model_state_dict")

    for key in candidates:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value, key
    raise KeyError(
        f"No usable state dict for --weights={choice}. "
        f"Available keys: {sorted(checkpoint.keys())}"
    )


def clean_key(key: str) -> str:
    # Full Lightning checkpoints prefix the network with 'model.'. Compiled
    # checkpoints can additionally contain one or more '_orig_mod.' prefixes.
    if key.startswith("model."):
        key = key[len("model.") :]
    while key.startswith("_orig_mod."):
        key = key[len("_orig_mod.") :]
    return key.replace("._orig_mod.", ".")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)


    cfg.setdefault("signal_control", {})["type"] = args.normalization
    cfg.setdefault("activation", {})["type"] = args.activation
    # Evaluation does not need torch.compile and avoiding it makes checkpoint
    # key handling independent of PyTorch's OptimizedModule wrapper.
    cfg.setdefault("model", {})["compile_model"] = False
    if args.batch_size is not None:
        cfg.setdefault("data", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg.setdefault("data", {})["num_workers"] = args.num_workers

    module = ClassificationModule(cfg)
    model = module.model

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("The checkpoint must contain a dictionary.")
    state_dict, source_key = choose_state_dict(payload, args.weights)
    state_dict = {clean_key(k): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint/architecture mismatch. Verify --normalization and "
            "--activation, and use the YAML from the same run.\n\n"
            f"Original error:\n{exc}"
        ) from exc

    device = torch.device(args.device)
    model.to(device).eval()

    dm = build_datamodule(cfg["data"])
    dm.setup("validate")
    loader = dm.val_dataloader()

    loss_sum = 0.0
    correct1 = 0
    correct5 = 0
    sample_count = 0
    use_amp = device.type == "cuda"

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = F.cross_entropy(logits, targets, reduction="sum")

            topk = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
            loss_sum += loss.item()
            correct1 += topk[:, 0].eq(targets).sum().item()
            correct5 += topk.eq(targets[:, None]).any(dim=1).sum().item()
            sample_count += targets.numel()

    checkpoint_type = payload.get("checkpoint_type", "unknown")
    monitor = payload.get("monitor")
    monitor_value = payload.get("monitor_value")
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Checkpoint type  : {checkpoint_type}")
    print(f"Loaded state     : {source_key}")
    print(f"Architecture     : {args.normalization.upper()} + {args.activation.upper()}")
    if monitor is not None and monitor_value is not None:
        print(f"Saved monitor    : {monitor} = {float(monitor_value):.4f}")
    print(f"Samples          : {sample_count}")
    print(f"Validation loss  : {loss_sum / sample_count:.4f}")
    print(f"Validation top-1 : {100.0 * correct1 / sample_count:.2f}%")
    print(f"Validation top-5 : {100.0 * correct5 / sample_count:.2f}%")


if __name__ == "__main__":
    main()
