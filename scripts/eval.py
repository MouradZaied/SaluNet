#!/usr/bin/env python3
"""Evaluate a SaluNet checkpoint on CIFAR-10/100.

Run from the repository root with the YAML and operators matching the checkpoint.
FP32 is the default; --precision fp16 enables CUDA autocast.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def find_repo_root(start: Path) -> Path:
    """Find the repository root containing src/salunet."""
    for candidate in (start, *start.parents):
        if (candidate / 'src' / 'salunet').is_dir():
            return candidate
    raise RuntimeError('Unable to find repository root containing src/salunet.')


ROOT = find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(ROOT / 'src'))

from salunet.data.registry import build_datamodule
from salunet.lightning.classification_module import ClassificationModule
from salunet.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Training YAML used by the run.')
    parser.add_argument('--checkpoint', required=True, help='.pth or .ckpt checkpoint.')
    parser.add_argument('--normalization', choices=('salu', 'bn', 'ln', 'rmsnorm', 'dyt'), required=True)
    parser.add_argument('--activation', choices=('relu', 'gelu', 'swalu', 'galu'), required=True)
    parser.add_argument('--weights', choices=('auto', 'selected', 'raw', 'ema'), default='auto')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--precision', choices=('fp32', 'fp16'), default='fp32',
                        help='fp32 disables autocast (default); fp16 enables CUDA autocast.')
    return parser.parse_args()


def choose_state_dict(checkpoint: dict, choice: str) -> tuple[dict, str]:
    if choice == 'raw':
        candidates = ('raw_model_state_dict', 'model_state_dict', 'state_dict')
    elif choice == 'ema':
        candidates = ('ema_model_state_dict', 'model_state_dict')
    elif choice == 'selected':
        candidates = ('model_state_dict',)
    else:
        candidates = ('model_state_dict', 'state_dict', 'raw_model_state_dict')
    for key in candidates:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value, key
    raise KeyError(f'No usable state dict for --weights={choice}. Available keys: {sorted(checkpoint.keys())}')


def clean_key(key: str) -> str:
    """Remove Lightning and torch.compile prefixes."""
    if key.startswith('model.'):
        key = key[len('model.'):]
    while key.startswith('_orig_mod.'):
        key = key[len('_orig_mod.'):]
    return key.replace('._orig_mod.', '.')


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.precision == 'fp16' and device.type != 'cuda':
        raise ValueError('--precision fp16 requires a CUDA device. Use fp32 on CPU.')
    cfg = load_config(args.config)
    cfg.setdefault('signal_control', {})['type'] = args.normalization
    cfg.setdefault('activation', {})['type'] = args.activation
    cfg.setdefault('model', {})['compile_model'] = False
    if args.batch_size is not None:
        cfg.setdefault('data', {})['batch_size'] = args.batch_size
    if args.num_workers is not None:
        cfg.setdefault('data', {})['num_workers'] = args.num_workers

    module = ClassificationModule(cfg)
    model = module.model
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError('The checkpoint must contain a dictionary.')
    state_dict, source_key = choose_state_dict(payload, args.weights)
    state_dict = {clean_key(k): v for k, v in state_dict.items()}
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError('Checkpoint/architecture mismatch. Verify operators and training YAML.\n'
                           f'Original error:\n{exc}') from exc
    model.to(device).eval()
    dm = build_datamodule(cfg['data'])
    dm.setup('validate')
    loader = dm.val_dataloader()
    loss_sum = 0.0
    correct1 = correct5 = sample_count = 0
    use_amp = args.precision == 'fp16'
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = F.cross_entropy(logits, targets, reduction='sum')
            topk = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
            loss_sum += loss.item()
            correct1 += topk[:, 0].eq(targets).sum().item()
            correct5 += topk.eq(targets[:, None]).any(dim=1).sum().item()
            sample_count += targets.numel()
    if sample_count == 0:
        raise ValueError('The evaluation loader contains no samples.')
    print(f'Checkpoint       : {args.checkpoint}')
    print(f'Checkpoint type  : {payload.get("checkpoint_type", "unknown")}')
    print(f'Loaded state     : {source_key}')
    print(f'Architecture     : {args.normalization.upper()} + {args.activation.upper()}')
    print(f'Precision        : {args.precision} (autocast {"enabled" if use_amp else "disabled"})')
    monitor, monitor_value = payload.get('monitor'), payload.get('monitor_value')
    if monitor is not None and monitor_value is not None:
        print(f'Saved monitor    : {monitor} = {float(monitor_value):.4f}')
    print(f'Samples          : {sample_count}')
    print(f'Validation loss  : {loss_sum / sample_count:.4f}')
    print(f'Validation top-1 : {100.0 * correct1 / sample_count:.2f}%')
    print(f'Validation top-5 : {100.0 * correct5 / sample_count:.2f}%')


if __name__ == '__main__':
    main()
