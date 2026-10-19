import torch.nn as nn

from salunet.layers.salu import SALU
from salunet.signal_control.derf import Derf
from salunet.signal_control.dyt import DyT


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


def make_signal_control(
    name: str,
    *,
    family: str = "cnn",
    num_channels: int | None = None,
    dim: int | None = None,
    num_groups: int = 32,
    **kwargs,
) -> nn.Module:
    name = (name or "none").lower()
    family = (family or "cnn").lower()

    if name in {"none", "identity"}:
        return nn.Identity()

    if name == "bn":
        if num_channels is None:
            raise ValueError("num_channels is required for BatchNorm2d.")
        return nn.BatchNorm2d(num_channels)

    if name == "groupnorm":
        if num_channels is None:
            raise ValueError("num_channels is required for GroupNorm.")
        groups = min(num_groups, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)

    if name == "ln":
        if family in {"transformer", "vit", "mlp"}:
            if dim is None:
                raise ValueError("dim is required for LayerNorm.")
            return nn.LayerNorm(dim)
        if num_channels is None:
            raise ValueError("num_channels is required for LayerNorm2d.")
        return LayerNorm2d(num_channels)

    if name == "rmsnorm":
        if dim is None:
            raise ValueError("dim is required for RMSNorm.")
        return nn.RMSNorm(dim)

    if name == "salu":
        return SALU(**kwargs)

    if name == "dyt":
        return DyT(dim=dim, **kwargs)

    if name == "derf":
        return Derf(dim=dim, **kwargs)

    raise ValueError(f"Unknown signal_control module: {name}")
