import torch.nn as nn
from torchvision.models import vision_transformer

from salunet.activations.factory import make_activation
from salunet.models.registry import register_model
from salunet.signal_control.factory import make_signal_control


def _replace_vit_modules(module: nn.Module, *, signal_control: str, activation: str, dim: int, salu_kwargs: dict | None = None, activation_kwargs: dict | None = None):
    salu_kwargs = salu_kwargs or {}
    activation_kwargs = activation_kwargs or {}
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, make_signal_control(signal_control, family="transformer", dim=child.normalized_shape[0], **salu_kwargs))
        elif isinstance(child, nn.GELU):
            setattr(module, name, make_activation(activation, **activation_kwargs))
        else:
            _replace_vit_modules(child, signal_control=signal_control, activation=activation, dim=dim, salu_kwargs=salu_kwargs, activation_kwargs=activation_kwargs)


@register_model("vit_b16")
def build_vit_b16(
    num_classes: int,
    signal_control: str = "ln",
    activation: str = "gelu",
    pretrained: bool = False,
    salu_kwargs: dict | None = None,
    activation_kwargs: dict | None = None,
    **kwargs,
):
    weights = None
    if pretrained:
        weights = vision_transformer.ViT_B_16_Weights.DEFAULT
    model = vision_transformer.vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    _replace_vit_modules(model, signal_control=signal_control, activation=activation, dim=model.hidden_dim, salu_kwargs=salu_kwargs, activation_kwargs=activation_kwargs)
    return model
