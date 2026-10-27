# SaluNet-C-50: experimental unclamped abs-positive SWALU

This variant starts from the released clamped reference script, not from the
separate refactored trainer. ImageNet performance is not validated. Published
reference scores must not be attributed to this variant.

## Changes

- Remove only the SWALU output clamp.
- Change the internal SWALU parameters from exp(log parameters) to abs(raw)+1e-6.
- Keep SALU, FP32 outputs, five critical initializations, optimizer,
  learning-rate schedule, augmentations, Beta(0.2,0.2), and RA x4 unchanged.
- Apply and correctly normalize the final incomplete accumulation window.
- Update EMA only after successful optimizer steps; match EMA tensors by name
  after unwrapping torch.compile, preserving the reference EMA decay warmup.
- Report training accuracy weighted against both mixed targets.
- Finish writing the new best checkpoint before deleting the previous one.
- Use separate experimental checkpoint and W&B output names.

## Training

Review dataset paths, GPU batch size and output paths in the script first.
Defaults retain batch 180, accumulation 5, 150 epochs, and a single CUDA GPU.

```bash
python train_salunet_c50_unclamped_abs.py
```

## Evaluation of NEW checkpoints only

```bash
python evaluate_salunet_c50_unclamped_abs.py \
  --checkpoint /path/to/new/ema_checkpoint.pth \
  --data-root /path/to/original/imagenet/val \
  --resize-size 320 --crop-size 288 --batch-size 128 --num-workers 4
```

The released historical clamped checkpoint has incompatible gate coordinates.
Keep using the reference evaluation script for that checkpoint. This variant
requires training from scratch. It is not an isolated clamp ablation, because
the SWALU parameterization and training bugs also change.

Best-checkpoint files contain selected EMA weights and training metadata, not
a complete resume state. The historical raw validation path remains unchanged.

Validation performed here: Python syntax and static comparison of standalone
SALU against the reference. No PyTorch/GPU or ImageNet execution was possible
in the preparation environment. Independent experiments are welcome; report
the code version, seed, environment, logs and evaluation settings.
