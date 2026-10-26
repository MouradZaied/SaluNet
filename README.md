# SaluNet

**Learnable elementwise transformations for normalization-free convolutional networks and Vision Transformers.**

SaluNet explores replacing feature normalization layers with **SALU**, a learnable elementwise transformation that does not compute batch or token feature statistics. Its gated variants, **SWALU** and **GALU**, provide learnable activation functions.

This repository contains PyTorch and PyTorch Lightning implementations, training and evaluation scripts, and experiment configurations for CIFAR-10 and CIFAR-100. The results section also summarizes the ImageNet-1K experiments reported in the manuscript.

![SaluNet teaser: PReLU dynamics on CIFAR-10, SALU geometry, and batch-size resilience on CIFAR-100.](assets/teaser.png)

**From normalization-dependent activation plasticity to geometric stabilization.**
**Left (CIFAR-10):** Mean PReLU slope across three activation layers under BN,
no normalization, and SALU. Curves and bands report the mean and one standard
deviation over three seeds; the dashed line marks $\alpha_0=0.25$.
**Center:** SALU fixes the geometric center at the origin while learning the
saturation amplitude $S$, transition width $W$, and local gain $S/W$.
The inset contrasts its polynomial derivative decay with those of $\tanh$
and $\mathrm{erf}$.
**Right (CIFAR-100):** SaluNet-C-18 remains trainable as batch size decreases
from 512 to 1, whereas the matched ResNet-18 (BN+ReLU) baseline degrades sharply
at small batch sizes.

[View the vector figure (PDF)](assets/teaser.pdf).

## Operators

For positive coefficients $a$ and $b$, SALU is defined as

$$
\phi(x;a,b)=\frac{ax}{\sqrt{1+abx^2}}.
$$

Its slope at the origin is $a$, its asymptotic magnitude is $\sqrt{a/b}$, and its characteristic transition scale is $(ab)^{-1/2}$. The coefficients are learned during training; their positivity policy and initialization depend on the experiment configuration.

The corresponding gated activations are

$$
\mathrm{SWALU}(x;a,b)=\frac{x}{2}\left(1+\phi(x;a,b)\right),
$$

$$
\mathrm{GALU}(x;a,b)=\frac{x}{2}\left(1+\phi(\psi(x);a,b)\right),
\qquad
\psi(x)=\sqrt{\frac{2}{\pi}}\left(x+0.044715x^3\right).
$$

SALU is bounded for fixed positive coefficients. SWALU and GALU, as defined above, are unbounded. Output clipping, when enabled in a particular experiment, is an additional implementation choice and is not part of these definitions.

## Model families

- **SaluNet-C:** convolutional models with SALU in place of normalization layers, optionally combined with SWALU or GALU activations.
- **SaluNet-T:** Vision Transformer experiments with configurable normalization and activation choices, including SALU and gated variants.

For the ResNet experiments, gate parameters are shared across activation calls within a residual block. The stem has its own activation module.

## Results

All accuracies are percentages. **Main** denotes the directly optimized model;
**EMA** denotes the exponential moving average model. CIFAR EMA and Main columns
report mean ± sample standard deviation of per-run best accuracies over three
independent runs.

### ResNet-18 / CIFAR

| Dataset | Stabilizer + activation | Runs | EMA Top-1 (%) | Main Top-1 (%) | Best Top-1 (%) |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | BN + ReLU | Multiple | 96.95 ± 0.09 | 96.93 ± 0.08 | 97.02 |
| CIFAR-10 | SALU + SWALU | Multiple | 97.08 ± 0.05 | 97.05 ± 0.06 | 97.24 |
| CIFAR-10 | SALU + GALU | Multiple | **97.19 ± 0.11** | **97.17 ± 0.10** | **97.35** |
| CIFAR-100 | BN + ReLU | Multiple | 80.68 ± 0.64 | 80.89 ± 0.25 | 81.26 |
| CIFAR-100 | BN + SWALU | Multiple | 81.13 ± 0.24 | 81.14 ± 0.21 | 81.33 |
| CIFAR-100 | SALU + ReLU | Multiple | 80.37 ± 0.07 | 79.97 ± 0.19 | 80.45 |
| CIFAR-100 | GN + ReLU | Multiple | 80.75 ± 0.28 | 79.95 ± 0.27 | 80.97 |
| CIFAR-100 | GN + SWALU | Multiple | 82.53 ± 0.34 | 81.69 ± 0.28 | 82.78 |
| CIFAR-100 | SALU + GALU | Multiple | 82.95 ± 0.31 | **82.37 ± 0.34** | 83.25 |
| CIFAR-100 | SALU + SWALU | Multiple | **83.02 ± 0.29** | 82.36 ± 0.31 | **83.27** |

Results reported as mean ± standard deviation summarize runs with different
random seeds. The exact number of runs and seed values are not specified here.

The deterministic CIFAR-100 result of 83.42% is reported separately as an
individual run.

ResNet-18 experiments use 300 epochs. **Best** is a separately reported best
observed record and is not necessarily the maximum of the three runs summarized
in the means. The 83.27% entry for SALU + SWALU on CIFAR-100 refers specifically
to the three-run cohort. The additional deterministic run is reported separately
below.

#### SALU + SWALU / CIFAR-100: execution settings

| Configuration | Execution mode | Runs | EMA Top-1 (%) |
|---|---|---:|---:|
| SALU + SWALU | Non-deterministic | Multiple | 83.02 ± 0.29 |
| SALU + SWALU | Deterministic | 1 | **83.42** |

- **Non-deterministic cohort:** `deterministic: false`, `cudnn_benchmark: true`.
- **Deterministic run:** `deterministic: true`, `cudnn_benchmark: false`.

The deterministic run is reported separately because its execution settings
differ from those of the three-run cohort. It is excluded from that cohort's
mean and sample standard deviation. The best EMA accuracy within the cohort is
83.27%. A single run does not establish an accuracy advantage of deterministic
execution.

For the CNN CIFAR-100 comparison, BN models use EMA decay 0.997; SALU, GN, and
RMSNorm use 0.9997. Main-model results provide a comparison independent of
this EMA smoothing choice.

### Compact ViT / CIFAR

| Dataset | Stabilizer + activation | Runs | EMA Top-1 (%) | Main Top-1 (%) | Best Top-1 (%) |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | LN + GELU | Multiple | 91.43 ± 0.31 | 91.29 ± 0.29 | 91.61 |
| CIFAR-10 | SALU + GELU | Multiple | 91.39 ± 0.21 | 91.22 ± 0.11 | 91.63 |
| CIFAR-10 | SALU + GALU | Multiple | 91.70 ± 0.17 | 91.47 ± 0.15 | 91.89 |
| CIFAR-10 | SALU + SWALU | Multiple | **91.91 ± 0.19** | **91.92 ± 0.16** | **92.11** |
| CIFAR-100 | DyT + GELU | Multiple | 65.32 ± 0.49 | 64.15 ± 0.70 | 65.85 |
| CIFAR-100 | RMSNorm + GELU | Multiple | 67.86 ± 0.36 | 66.76 ± 0.34 | 68.20 |
| CIFAR-100 | LN + GELU | Multiple | 68.66 ± 0.13 | 67.26 ± 0.25 | 68.76 |
| CIFAR-100 | SALU + GALU | Multiple | 70.29 ± 0.11 | **69.64 ± 0.29** | 70.37 |
| CIFAR-100 | SALU + SWALU | Multiple | 70.11 ± 0.48 | 69.22 ± 0.43 | 70.56 |
| CIFAR-100 | SALU + GELU | Multiple | **70.39 ± 0.41** | 69.56 ± 0.51 | **70.82** |

Results reported as mean ± standard deviation summarize runs with different
random seeds. The exact number of runs and seed values are not specified here.

The deterministic CIFAR-100 result of 83.42% is reported separately as an
individual run.

All compact ViT configurations, including baselines, are rerun in the same
implementation for **200 epochs**. **Best** is the maximum across the three
runs and both model variants. The ordinary learning rate is 0.0005 on CIFAR-10
and 0.001 on CIFAR-100. DyT uses an empirically selected initial alpha of 1.

The highest observed EMA mean uses SALU + SWALU on CIFAR-10 and SALU + GELU on
CIFAR-100. Gate benefits are therefore dataset-dependent; these experiments do
not isolate the effect of learning the gate parameters.

**Checkpoint selection:** CIFAR experiments evaluate the official test set after
each epoch and select Main and EMA maxima independently. Repeated test-set
selection can introduce optimism; these results are not held-out estimates after
validation-only checkpoint selection.

### ImageNet-1K

| Model | Training epochs | Physical / effective batch | Evaluation crop | EMA Top-1 (%) | EMA Top-5 (%) |
|---|---:|---:|---:|---:|---:|
| SaluNet-C-50 | 150, with 4N training views per epoch | 180 / 900 | 224 × 224 | 79.41 | 94.50 |
| SaluNet-C-50 | Same checkpoint | 180 / 900 | 288 × 288 | 80.16 | 94.96 |
| SaluNet-T-B/16 | 300 | 128 / 128 | 224 × 224 | 80.12 | 94.79 |
| SaluNet-T-B/16 | Same checkpoint | 128 / 128 | 288 × 288 | 80.28 | 94.75 |

- Each backbone is represented by **one completed run**, trained from scratch on
  one shared NVIDIA A40 GPU with 24 GB available to the training process. No
  run-to-run standard deviation is reported.
- SaluNet-C-50 uses five-step gradient accumulation. Its sampler shuffles all
  four copies of each training index without truncation: 150 scheduled epochs
  therefore correspond to **600N training views**, where N is the training-set
  size. This is not the same epoch budget as TorchVision's truncated RA sampler.
- SaluNet-T-B/16 uses no gradient accumulation. Its adapted effective batch and
  learning rate differ from the published Derf recipe; the result is a
  scalability experiment rather than a recipe-matched operator comparison.
- The 288 × 288 evaluations reuse the selected EMA checkpoints **without further
  training or fine-tuning**. The ViT positional embedding is interpolated for
  the higher evaluation resolution.
- Training uses an offline pre-resized ImageNet copy. The reported C-50 results
  are recomputed on the original 50,000 validation images. Evaluation resolution
  and preprocessing must be matched when comparing published results.

### Physical batch-size robustness

| Physical batch size | ResNet-18, BN + ReLU, CIFAR-100 | SaluNet-C-18, SALU + SWALU, CIFAR-100 | SaluNet-C-18, SALU + GALU, CIFAR-10 |
|---:|---:|---:|---:|
| 512 | 82.15 | 83.25 | 97.35 |
| 256 | 82.32 | 82.86 | 97.25 |
| 128 | 82.55 | 82.00 | 97.30 |
| 64 | 81.90 | 81.74 | 97.20 |
| 32 | 81.27 | 81.38 | 97.35 |
| 16 | 78.49 | 80.91 | 97.10 |
| 8 | 67.90 | 80.69 | 97.08 |
| 4 | 46.47 | 80.07 | 96.60 |
| 2 | 30.90 | 79.12 | 95.57 |
| 1 | 1.10 | 76.23 | 93.44 |

These are retained point estimates from the batch-size sweep, not three-run
means. Batch sizes of at least 32 were explored with multiple seeds; settings
below 32 use one completed run per configuration. The learning-rate policy is
adapted identically for the compared models within each dataset and batch size.
These sweep records should not be substituted for the primary aggregate results.

<details>
<summary>Additional historical ResNet-18 / CIFAR-100 comparisons</summary>

| Configuration | EMA Top-1 (%) | Main Top-1 (%) | Best EMA Top-1 (%) |
|---|---:|---:|---:|
| RMSNorm + ReLU | 80.54 ± 0.31 | 79.71 ± 0.22 | 80.87 |
| RMSNorm + SWALU | 82.39 ± 0.30 | 81.62 ± 0.23 | 82.69 |
| DyT + ReLU | 79.93 ± 0.16 | 79.12 ± 0.03 | 80.08 |
| DyT + SWALU | 81.42 ± 0.15 | 79.30 ± 0.06 | 81.58 |

These comparisons are retained from the original experiment set. Means and
sample standard deviations use three independent runs; Best is the retained
best EMA record. They are reported separately from the updated SaluNet
aggregate results.

</details>

## Installation

Clone the repository:

```bash
git clone https://github.com/MouradZaied/SaluNet.git
cd SaluNet
```

Create and activate a Python environment, then install a compatible PyTorch/torchvision pair for your hardware using the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

Install the remaining dependencies and the local package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Review `requirements.txt` and `pyproject.toml` for dependency requirements. GPU memory usage depends on the architecture, batch size, numerical precision, and compilation settings.

## Training

Run commands from the repository root. Review dataset locations, batch size, logging settings, seed, and output paths in the selected YAML file before launching.

### SaluNet-C / CIFAR-10

```bash
python scripts/train.py --config configs/experiments/salunet-c-18/cifar10/salunet-c-18_cifar10.yaml
```

### SaluNet-C / CIFAR-100

Reference configuration with deterministic execution (seed 142):

```bash
python scripts/train.py --config configs/experiments/salunet-c18/cifar100/salunet-c-18_cifar100_salu_swalu.yaml
```

Non-deterministic configuration with SALU/SWALU parameter logging:

```bash
python scripts/train.py --config configs/experiments/salunet-c18/cifar100/salunet-c-18_cifar100_salu_swalu_a_b.yaml
```

The second configuration records the evolution of parameters `a` and `b`.
It is a separate experiment and is not intended to reproduce the 83.42% result.

Additional configurations are available under `configs/experiments/`, including CIFAR-10 experiments and alternative normalization/activation combinations. Use the configuration associated with the experiment you intend to reproduce; different configurations may use different recipes.

To inspect command-line options:

```bash
python scripts/train.py --help
python scripts/eval.py --help
```


## Pretrained checkpoints

| Model | Dataset | Weights | Top-1 | Top-5 | Download |
| --- | --- | --- | --- | --- | --- |
| SaluNet-C-18 (SALU + GALU) | CIFAR-10 | EMA | 97.27% | 99.89% | [Checkpoint](https://drive.google.com/file/d/1IeYOrmdkGwm-cAaz2XKzfz4XhQ1dcSsp/view?usp=sharing) |
| SaluNet-C-18 (SALU + SWALU) | CIFAR-100 | EMA | 83.42% | 96.25% | [Checkpoint](https://drive.google.com/file/d/11_BCqoLbV6SWW0Rdkjf68vjm06JVC8rj/view?usp=drive_link) |

Both released checkpoints were verified in FP32 on the respective
10,000-image test sets.

For CIFAR-10, the best observed accuracy was 97.35%; its checkpoint is
no longer available. The released checkpoint achieved 97.27% in a separate run.

For CIFAR-100, FP16 evaluation gives 83.43% Top-1 and 96.25% Top-5.

### CIFAR-10 evaluation

Download the CIFAR-10 checkpoint into the repository root as
`best_ema_model_SaluNet-C-18_C10_97.27.pth`, then run:

```bash
python scripts/eval.py \
  --config configs/experiments/salunet-c-18/cifar10/salunet-c-18_cifar10.yaml \
  --checkpoint best_ema_model_SaluNet-C-18_C10_97.27.pth \
  --normalization salu \
  --activation galu \
  --weights selected \
  --batch-size 512 \
  --num-workers 2 \
  --device cuda \
  --precision fp32
```

### CIFAR-100 evaluation

Download the CIFAR-100 checkpoint and save it as
`checkpoints/cifar100_salu_swalu/best_ema_model.pth`, then run:

```bash
python scripts/eval.py \
  --config configs/experiments/salunet-c-18/cifar100/salunet-c-18_cifar100.yaml \
  --checkpoint checkpoints/cifar100_salu_swalu/best_ema_model.pth \
  --normalization salu \
  --activation swalu \
  --weights selected \
  --batch-size 512 \
  --num-workers 2 \
  --device cuda \
  --precision fp32
```

## Evaluation and reproducibility

Use `scripts/eval.py` with the options shown by `--help`. Evaluate checkpoints with the same architecture, activation definitions, positivity policy, clipping settings, and preprocessing as the corresponding training run. Distinguish ordinary model weights from exponential moving average (EMA) weights when reporting accuracy.

Record the configuration, random seed, software versions, device, and checkpoint selection rule for each experiment. A fixed seed alone does not guarantee identical results across software versions or devices. Deterministic execution and performance-oriented CUDA settings are separate choices.

## Repository structure

| Location | Contents |
|---|---|
| `configs/experiments/salunet-c18/` | ResNet-18 experiment configurations |
| `configs/experiments/salunet-t/` | Vision Transformer experiment configurations |
| `configs/models/` | Model configuration files |
| `scripts/train.py` | Training entry point |
| `scripts/eval.py` | Evaluation entry point |
| `scripts/inspect_model.py` | Model inspection utility |
| `src/salunet/activations/` | Gated activation implementations |
| `src/salunet/layers/` | SALU and supporting layers |
| `src/salunet/models/` | Model implementations |
| `src/salunet/lightning/` | Lightning modules and callbacks |
| `src/salunet/optim/` | Parameter groups and learning-rate schedules |
| `src/salunet/recipes/` | Training utilities, losses, mixing, and EMA |
| `src/salunet/signal_control/` | Signal-control alternatives |
| `LICENSES/` | Upstream license notices |

## Acknowledgments

This repository includes components adapted from or related to:

- [ViT-CIFAR](https://github.com/omihub777/ViT-CIFAR), by Omiita.
- [Derf](https://github.com/zlab-princeton/Derf), by Chen MingZhi.

Their upstream MIT license notices are retained in [`LICENSES/ViT-CIFAR-MIT.txt`](LICENSES/ViT-CIFAR-MIT.txt) and [`LICENSES/Derf-MIT.txt`](LICENSES/Derf-MIT.txt).

## License

See [`LICENSE`](LICENSE) for the project license and [`LICENSES/`](LICENSES/) for retained upstream notices.

## Contact

Maintained by [Mourad Zaied](https://github.com/MouradZaied). For questions about the implementation or configurations, please open an issue in this repository.
