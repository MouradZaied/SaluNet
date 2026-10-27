#!/usr/bin/env python3
"""Evaluate a SALU/SWALU ViT checkpoint produced by this Derf repository.

Place this file in the same directory as ``main.py`` and run it from there.
The checkpoint is loaded into the uncompiled model first; optional
``torch.compile`` is applied only after the weights have been restored.
"""

import argparse
from collections import OrderedDict
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from timm.models import create_model

from datasets import build_dataset
from engine import evaluate
from main import get_args_parser, str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        "Evaluate SALU ViT",
        parents=[get_args_parser()],
        add_help=True,
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Checkpoint produced by main.py.",
    )
    parser.add_argument(
        "--checkpoint_key",
        default="model_ema",
        choices=["model", "model_ema"],
        help="State dictionary to evaluate (default: model_ema).",
    )
    parser.add_argument(
        "--compile_eval",
        type=str2bool,
        default=True,
        help="Compile the model after loading its weights.",
    )
    parser.add_argument(
        "--compile_mode",
        default="max-autotune-no-cudagraphs",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
    )
    parser.add_argument(
        "--resize-size",
        "--resize_size",
        dest="resize_size",
        default=None,
        type=int,
        help="Short-edge resize used for validation before the center crop.",
    )
    parser.add_argument(
        "--crop-size",
        "--crop_size",
        dest="crop_size",
        default=None,
        type=int,
        help="Validation center-crop size and ViT input resolution.",
    )
    return parser


def build_model(args):
    if "convnext" in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            ls_init_value=args.layer_scale_init_value,
            head_init_scale=args.head_init_scale,
        )
    elif "vit" in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool="avg",
            drop_path_rate=args.drop_path,
            img_size=args.input_size,
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if args.normtype == "salu":
        from salu import convert_gelu_to_swalu, convert_ln_to_salu

        model = convert_ln_to_salu(model)
        model = convert_gelu_to_swalu(model)
    elif args.normtype == "derf":
        from dynamic_erf import convert_ln_to_salu

        model = convert_ln_to_salu(model)
    elif args.normtype == "dyt":
        from dynamic_tanh import convert_ln_to_dyt

        model = convert_ln_to_dyt(model)
    elif args.normtype not in (None, "ln"):
        raise ValueError(
            f"This evaluation script does not implement normtype={args.normtype!r}."
        )

    return model


def normalize_state_dict_keys(state_dict):
    """Remove wrappers commonly introduced by DDP and torch.compile."""
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            if key.startswith("module."):
                key = key[len("module.") :]
            elif key.startswith("_orig_mod."):
                key = key[len("_orig_mod.") :]
        cleaned[key] = value
    return cleaned


def interpolate_positional_embedding(model, state_dict):
    """Interpolate checkpoint patch positions to the requested ViT grid."""
    if "pos_embed" not in state_dict or not hasattr(model, "pos_embed"):
        return state_dict

    source = state_dict["pos_embed"]
    target = model.pos_embed
    if source.shape == target.shape:
        return state_dict

    num_prefix_tokens = getattr(model, "num_prefix_tokens", 1)
    source_patch_tokens = source.shape[1] - num_prefix_tokens
    target_patch_tokens = target.shape[1] - num_prefix_tokens
    source_grid = int(math.sqrt(source_patch_tokens))
    target_grid = int(math.sqrt(target_patch_tokens))

    if source_grid**2 != source_patch_tokens or target_grid**2 != target_patch_tokens:
        raise RuntimeError(
            "Cannot infer square positional grids: "
            f"checkpoint={source_patch_tokens}, model={target_patch_tokens}."
        )

    prefix = source[:, :num_prefix_tokens]
    patches = source[:, num_prefix_tokens:]
    patches = patches.reshape(1, source_grid, source_grid, source.shape[-1])
    patches = patches.permute(0, 3, 1, 2).float()
    patches = F.interpolate(
        patches,
        size=(target_grid, target_grid),
        mode="bicubic",
        align_corners=False,
    )
    patches = patches.permute(0, 2, 3, 1).reshape(
        1, target_patch_tokens, source.shape[-1]
    )
    patches = patches.to(dtype=source.dtype)
    state_dict["pos_embed"] = torch.cat((prefix, patches), dim=1)
    print(
        "Interpolated positional embedding: "
        f"{source_grid}x{source_grid} -> {target_grid}x{target_grid}."
    )
    return state_dict


def load_checkpoint(model, checkpoint_path, checkpoint_key):
    # The checkpoint is trusted because it was generated by the user's run.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint_key not in checkpoint:
        available = ", ".join(sorted(checkpoint.keys()))
        raise KeyError(
            f"Key {checkpoint_key!r} is absent from the checkpoint. "
            f"Available keys: {available}"
        )

    state_dict = checkpoint[checkpoint_key]
    state_dict = normalize_state_dict_keys(state_dict)
    state_dict = interpolate_positional_embedding(model, state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            f"Missing keys: {incompatible.missing_keys}\n"
            f"Unexpected keys: {incompatible.unexpected_keys}"
        )

    epoch = checkpoint.get("epoch", "unknown")
    print(f"Loaded {checkpoint_key!r} from {checkpoint_path} (epoch={epoch}).")


def main():
    args = get_parser().parse_args()
    device = torch.device(args.device)

    if (args.resize_size is None) != (args.crop_size is None):
        raise ValueError("Specify --resize-size and --crop-size together.")
    if args.crop_size is not None:
        if args.resize_size < args.crop_size:
            raise ValueError("--resize-size must be >= --crop-size.")
        args.input_size = args.crop_size
        args.crop_pct = args.crop_size / args.resize_size
        print(
            f"Validation transform: resize={args.resize_size}, "
            f"center_crop={args.crop_size}, crop_pct={args.crop_pct:.6f}."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=torch.utils.data.SequentialSampler(dataset_val),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = build_model(args)
    load_checkpoint(model, args.checkpoint, args.checkpoint_key)
    model.to(device)
    model.eval()

    if args.compile_eval:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch version.")
        print(f"Compiling model with mode={args.compile_mode!r} ...")
        model = torch.compile(model, mode=args.compile_mode)

    print(
        f"Evaluating {len(dataset_val)} images with model={args.model}, "
        f"normtype={args.normtype}, checkpoint_key={args.checkpoint_key}."
    )
    stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)

    print("\nFinal ImageNet-1K validation result")
    print(f"Top-1: {stats['acc1']:.3f}%")
    print(f"Top-5: {stats['acc5']:.3f}%")


if __name__ == "__main__":
    main()
