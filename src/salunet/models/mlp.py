import torch.nn as nn

from salunet.activations.factory import make_activation
from salunet.models.registry import register_model
from salunet.signal_control.factory import make_signal_control


@register_model("mlp")
def build_mlp(
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    depth: int = 3,
    signal_control: str = "none",
    activation: str = "salu",
    salu_kwargs: dict | None = None,
    activation_kwargs: dict | None = None,
    **kwargs,
):
    salu_kwargs = salu_kwargs or {}
    activation_kwargs = activation_kwargs or {}
    layers = [nn.Flatten()]
    in_dim = input_dim
    for _ in range(depth - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(make_signal_control(signal_control, family="mlp", dim=hidden_dim, **salu_kwargs))
        layers.append(make_activation(activation, **activation_kwargs))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, num_classes))
    return nn.Sequential(*layers)
