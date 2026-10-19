import torch
import torch.nn as nn


class Derf(nn.Module):
    """Dynamic erf signal-control module.

    Experimental normalization-free replacement:
        Derf(x) = erf(alpha*x + shift)

    Use as a configurable comparison module; the exact published variant should
    be verified before final benchmark claims.
    """

    def __init__(self, alpha_init: float = 0.5, shift_init: float = 0.0, affine: bool = False, dim: int | None = None):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(float(shift_init), dtype=torch.float32))
        self.affine = affine
        if affine:
            if dim is None:
                raise ValueError("dim is required when affine=True.")
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.erf(self.alpha * x + self.shift)
        if self.affine:
            y = y * self.weight + self.bias
        return y
