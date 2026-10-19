import torch
import torch.nn as nn


class DyT(nn.Module):
    """Dynamic Tanh signal-control module.

    This is a pointwise normalization-free replacement. It is shape-compatible
    with CNNs and Transformers when used as a pointwise module.
    """

    def __init__(self, alpha_init: float = 0.5, affine: bool = False, dim: int | None = None):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.affine = affine
        if affine:
            if dim is None:
                raise ValueError("dim is required when affine=True.")
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.tanh(self.alpha * x)
        if self.affine:
            # Works for (..., dim) Transformer tensors. CNN affine should use affine=False
            # or a dedicated wrapper.
            y = y * self.weight + self.bias
        return y
