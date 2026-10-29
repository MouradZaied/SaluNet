"""SALU and SWALU used by the reference ImageNet ViT-B/16 experiment."""

import torch
import torch.nn as nn


class SALU(nn.Module):
    """Learn a positive coefficient pair through exponential parameters."""

    def __init__(self, a_init=1.0, b_init=0.1):
        super().__init__()
        self.loga = nn.Parameter(torch.log(torch.tensor(a_init, dtype=torch.float32)))
        self.logb = nn.Parameter(torch.log(torch.tensor(b_init, dtype=torch.float32)))

    def forward(self, x):
        x = x.float()
        a = torch.exp(self.loga)
        b = torch.exp(self.logb)
        denom = torch.sqrt(1 + a * b * x**2)
        return (a * x) / denom


class SWALU(nn.Module):
    """Apply an unclamped SALU-based gate with abs + eps coefficients."""

    def __init__(self, a=0.5, b=0.5, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        a = torch.abs(self.a) + self.eps
        b = torch.abs(self.b) + self.eps
        x = x.float()
        denom = torch.sqrt(1 + a * b * x**2)
        salu_x = (a * x) / denom
        return 0.5 * x * (1 + salu_x)


def convert_ln_to_salu(module):
    """Replace child LayerNorm modules recursively with independent SALU modules."""
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            new_salu = SALU(a_init=1.0, b_init=0.1)
            setattr(module, name, new_salu)
        else:
            convert_ln_to_salu(child)
    return module


def convert_gelu_to_swalu(module, a=1.0, b=1.0, clamp_value=10.0, eps=1e-6):
    """Replace child GELU modules recursively with independent SWALU modules.

    The conversion initializes gates at a=b=1 by default. The unused
    clamp_value argument is retained for compatibility; no clamp is applied.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, SWALU(a=a, b=b, eps=eps))
        else:
            convert_gelu_to_swalu(child, a=a, b=b, eps=eps)
    return module
