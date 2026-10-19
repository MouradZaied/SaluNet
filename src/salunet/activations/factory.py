import torch.nn as nn

from salunet.activations.galu import GALU
from salunet.activations.swalu import SWALU
from salunet.layers.salu import SALU


def make_activation(name: str, **kwargs) -> nn.Module:
    name = (name or "relu").lower()
    inplace = kwargs.pop("inplace", True)

    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=inplace)
    if name == "gelu":
        return nn.GELU()
    if name == "salu":
        return SALU(**kwargs)
    if name == "swalu":
        return SWALU(**kwargs)
    if name == "galu":
        return GALU(**kwargs)
    if name in {"identity", "none"}:
        return nn.Identity()

    raise ValueError(f"Unknown activation: {name}")
