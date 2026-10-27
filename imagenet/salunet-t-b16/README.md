# SaluNet-T-B/16 on ImageNet-1K

This implementation builds on the [Derf ViT codebase](https://github.com/zlab-princeton/Derf/tree/8dd9ea3cc70370f423a80a80af6f72466bc621db/ViT).

It replaces LayerNorm with SALU and GELU with SWALU.
SWALU is not output-clamped in this implementation.

## Setup

Clone SaluNet and Derf into sibling directories:

```bash
git clone https://github.com/MouradZaied/SaluNet.git
git clone https://github.com/zlab-princeton/Derf.git

cd Derf
git checkout 8dd9ea3cc70370f423a80a80af6f72466bc621db

cp ../SaluNet/imagenet/salunet-t-b16/main.py ViT/main.py
cp ../SaluNet/imagenet/salunet-t-b16/salu.py ViT/salu.py
cp ../SaluNet/imagenet/salunet-t-b16/test.py ViT/test.py

cd ViT
```

Follow the upstream dependency installation instructions.
Run the following commands from the Derf `ViT` directory.

## Training

The reference experiment uses one GPU, batch size 128, no gradient
accumulation, and 300 epochs.

```bash
python main.py \
  --model vit_base_patch16_224 \
  --batch_size 128 \
  --lr 0.000125 \
  --model_ema true \
  --model_ema_eval true \
  --model_ema_decay 0.999 \
  --data_path /path/to/imagenet \
  --output_dir outputdir2/ \
  --normtype salu
```

Replace `/path/to/imagenet` with the directory containing `train/`
and `val/`. The upstream large-batch training command is available
in the Derf repository.

## Evaluation

The reported results use EMA weights:

| Resolution | Resize | Center crop | Top-1 (%) | Top-5 (%) |
| --- | --- | --- | --- | --- |
| 224 × 224 | 256 | 224 | 80.12 | 94.79 |
| 288 × 288 | 320 | 288 | 80.28 | 94.75 |

Download the [EMA checkpoint](https://drive.google.com/file/d/1tFNHfnQCrepLL7jry7ugfk4QpbIqX_P-/view?usp=sharing)
and save it in the Derf `ViT` directory as
`salunet-t-b16-imagenet1k-salu-swalu-ema.pth`.

### Evaluation at 224 × 224

```bash
python test.py \
  --checkpoint salunet-t-b16-imagenet1k-salu-swalu-ema.pth \
  --checkpoint_key model_ema \
  --model vit_base_patch16_224 \
  --batch_size 128 \
  --data_path /path/to/imagenet \
  --normtype salu \
  --resize-size 256 \
  --crop-size 224 \
  --compile_eval true \
  --compile_mode max-autotune-no-cudagraphs \
  --num_workers 4
```

### Evaluation at 288 × 288

```bash
python test.py \
  --checkpoint salunet-t-b16-imagenet1k-salu-swalu-ema.pth \
  --checkpoint_key model_ema \
  --model vit_base_patch16_224 \
  --batch_size 128 \
  --data_path /path/to/imagenet \
  --normtype salu \
  --resize-size 320 \
  --crop-size 288 \
  --compile_eval true \
  --compile_mode max-autotune-no-cudagraphs \
  --num_workers 4
```

## Integration status

These are standalone PyTorch scripts that depend on the upstream Derf
files. Conversion to PyTorch Lightning and integration with the unified
SaluNet-C-18 and SaluNet-T CIFAR codebase are ongoing.

## Attribution

The implementation builds on Derf. The upstream MIT license notice is
retained in [LICENSES/Derf-MIT.txt](../../LICENSES/Derf-MIT.txt).
