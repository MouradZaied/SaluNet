# SaluNet

**Learnable, statistics-free signal transformations for convolutional networks and Vision Transformers.**

SaluNet explores replacing feature normalization layers with **SALU**, a learnable elementwise transformation that does not compute batch or token feature statistics. Its gated variants, **SWALU** and **GALU**, provide learnable activation functions.

This repository contains PyTorch and PyTorch Lightning implementations, training and evaluation scripts, and experiment configurations for CIFAR-10 and CIFAR-100.

## Operators

For positive coefficients $a$ and $b$, SALU is defined as

$$
\phi(x;a,b)=\frac{ax}{\sqrt{1+abx^2}}.
$$

Its slope at the origin is $a$, its asymptotic magnitude is $\sqrt{a/b}$, and its characteristic transition scale is $(ab)^{-1/2}$. The coefficients are learned during training; their positivity policy and initialization depend on the experiment configuration.

The corresponding gated activations are

$$
mathrm{SWALU}(x;a,b)=\frac{x}{2}\left(1+\phi(x;a,b)\right),
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

### SaluNet-C / CIFAR-100

```bash
python scripts/train.py --config configs/experiments/salunet-c18/cifar100/cifar100_resnet18_salu_swalu_freeze_seed42-reproducible.yaml
```

### SaluNet-T / CIFAR-100

```bash
python scripts/train.py --config configs/experiments/salunet-t/cifar100/cifar100_vit_cifar_salu_swalu_seed_42.yaml
```

Additional configurations are available under `configs/experiments/`, including CIFAR-10 experiments and alternative normalization/activation combinations. Use the configuration associated with the experiment you intend to reproduce; different configurations may use different recipes.

To inspect command-line options:

```bash
python scripts/train.py --help
python scripts/eval.py --help
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
