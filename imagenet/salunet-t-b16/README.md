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
Run the training and evaluation commands below from the Derf `ViT` directory.

## Data preprocessing

Both training and validation images are resized offline so that their
longest side is 320 pixels, preserving aspect ratio with Lanczos
interpolation, and saved as JPEG at quality 95 using Pillow's default
chroma subsampling. Jupyter checkpoint copies are excluded.

Use `scripts/resize_imagenet_train.py` from the SaluNet repository
to prepare each split separately. Point `--data_path` to the parent
directory containing the preprocessed `train/` and `val/` folders.

Training augmentations are applied online. Evaluation applies an
additional bicubic resize and center crop, as specified below.

## Training

The reference experiment trains from scratch on one shared NVIDIA A40
GPU with 24 GB available to the training process. It uses physical and
effective batch sizes of 128, no gradient accumulation, and 300 epochs.

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

Replace `/path/to/imagenet` with the directory containing the preprocessed
`train/` and `val/` folders. The upstream large-batch training command is
available in the Derf repository.

The completed reference run used two training stages. The initial stage
used a learning rate of 0.000125 and 20 warmup epochs. Training resumed
from `checkpoint-150.pth` at epoch index 151, with a learning rate of
0.0001 and no additional warmup, keeping the total training horizon
at 300 epochs. EMA decay remained 0.999.

The command above describes the initial stage; reproducing the completed
run also requires this resumed stage.

The adapted batch size and learning-rate configuration differ from the
published Derf recipe. This experiment demonstrates scaling feasibility
under hardware constraints rather than a recipe-matched operator comparison.

## Evaluation

The reported results use EMA weights:

| Resolution | Online resize (shorter side) | Center crop | Top-1 (%) | Top-5 (%) |
| --- | --- | --- | --- | --- |
| 224 × 224 | 256 | 224 | 80.12 | 94.79 |
| 288 × 288 | 320 | 288 | 80.28 | 94.74 |

Both resolutions use the same selected EMA checkpoint, without additional
training or fine-tuning. The positional embedding is interpolated
bicubically for evaluation at 288 × 288.

Each result comes from the same completed training run; no run-to-run
standard deviation is reported.

Download the [EMA checkpoint](https://drive.google.com/file/d/1tFNHfnQCrepLL7jry7ugfk4QpbIqX_P-/view?usp=sharing)
and save it in the Derf `ViT` directory as
`salunet-t-b16-imagenet1k-salu-swalu-ema.pth`.

For both commands below, `/path/to/imagenet` must contain the validation
copy preprocessed offline as described above.

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
  --num_workers 4 \
  --use_amp false
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
  --num_workers 4 \
  --use_amp false
```

## Integration status

These are standalone PyTorch scripts that depend on the upstream Derf
files. Conversion to PyTorch Lightning and integration with the unified
SaluNet-C-18 and SaluNet-T CIFAR codebase are ongoing.

## Attribution

The implementation builds on Derf. The upstream MIT license notice is
retained in [LICENSES/Derf-MIT.txt](../../../LICENSES/Derf-MIT.txt).