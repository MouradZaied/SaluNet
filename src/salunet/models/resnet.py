import types

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck

from salunet.activations.factory import make_activation
from salunet.layers.drop_path import DropPath
from salunet.models.registry import register_model
from salunet.signal_control.factory import make_signal_control


def _normalize_bn_index_overrides(overrides: dict | None) -> dict[int, dict]:
    """Normalize config keys for per-BatchNorm SALU initialization overrides.

    The BN index is the global traversal order inside the torchvision ResNet,
    starting at 1. This matches the original ImageNet script convention.
    """

    if not overrides:
        return {}

    if "by_bn_index" in overrides:
        overrides = overrides.get("by_bn_index") or {}

    normalized: dict[int, dict] = {}
    for key, value in overrides.items():
        normalized[int(key)] = dict(value or {})
    return normalized


def _make_salu_kwargs_for_bn(
    *,
    bn_index: int,
    base_kwargs: dict,
    bn_index_overrides: dict[int, dict],
) -> tuple[dict, bool]:
    """Return per-BN SALU kwargs and whether replacement should happen."""

    local_kwargs = dict(base_kwargs)
    override = bn_index_overrides.get(bn_index)
    if override is None:
        local_kwargs.setdefault("name", f"salu_bn_{bn_index}")
        return local_kwargs, True

    if override.get("replace", True) is False:
        return local_kwargs, False

    if "a_init" in override or "a" in override:
        local_kwargs["a_init"] = override.get("a_init", override.get("a"))
    if "b_init" in override or "b" in override:
        local_kwargs["b_init"] = override.get("b_init", override.get("b"))
    if "positivity" in override:
        local_kwargs["positivity"] = override["positivity"]

    local_kwargs["name"] = override.get("name", f"salu_bn_{bn_index}")
    return local_kwargs, True


def _replace_modules(
    module: nn.Module,
    *,
    signal_control: str,
    activation: str,
    family: str = "cnn",
    salu_kwargs: dict | None = None,
    activation_kwargs: dict | None = None,
    bn_state: dict | None = None,
    bn_index_overrides: dict[int, dict] | None = None,
):
    salu_kwargs = salu_kwargs or {}
    activation_kwargs = activation_kwargs or {}
    bn_state = bn_state if bn_state is not None else {"idx": 0}
    bn_index_overrides = bn_index_overrides or {}

    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            bn_state["idx"] += 1
            bn_index = int(bn_state["idx"])
            local_salu_kwargs, replace = _make_salu_kwargs_for_bn(
                bn_index=bn_index,
                base_kwargs=salu_kwargs,
                bn_index_overrides=bn_index_overrides,
            )
            if not replace:
                continue

            module_name = local_salu_kwargs.pop("name", f"salu_bn_{bn_index}")
            replacement = make_signal_control(
                signal_control,
                family=family,
                num_channels=child.num_features,
                **local_salu_kwargs,
            )
            if hasattr(replacement, "name"):
                replacement.name = module_name
            setattr(module, name, replacement)
        elif isinstance(child, nn.ReLU):
            setattr(module, name, make_activation(activation, **activation_kwargs))
        else:
            _replace_modules(
                child,
                signal_control=signal_control,
                activation=activation,
                family=family,
                salu_kwargs=salu_kwargs,
                activation_kwargs=activation_kwargs,
                bn_state=bn_state,
                bn_index_overrides=bn_index_overrides,
            )


def _iter_resnet_blocks(model: nn.Module):
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, layer_name, None)
        if layer is None:
            continue
        for block in layer:
            if isinstance(block, (BasicBlock, Bottleneck)):
                yield block


def _basicblock_forward_with_droppath(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out = self.drop_path(out) if hasattr(self, "drop_path") else out
    out += identity
    out = self.relu(out)
    return out


def _bottleneck_forward_with_droppath(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out = self.drop_path(out) if hasattr(self, "drop_path") else out
    out += identity
    out = self.relu(out)
    return out


def _add_drop_path_to_resnet(
    model: nn.Module,
    drop_path_rate: float = 0.0,
    *,
    scale_by_keep: bool = True,
    reverse: bool = False,
) -> None:
    """Add DropPath to ResNet residual branches.

    DropPath is applied to the residual branch immediately before adding the
    shortcut, matching the intended stochastic-depth behavior. Rates are
    linearly assigned across blocks. By default, later/deeper blocks receive
    larger rates. ``reverse=True`` is provided only to reproduce legacy notebook
    experiments that reversed the schedule.
    """

    drop_path_rate = float(drop_path_rate)
    if drop_path_rate <= 0.0:
        return

    blocks = list(_iter_resnet_blocks(model))
    if not blocks:
        return

    if len(blocks) == 1:
        rates = [drop_path_rate]
    else:
        rates = [drop_path_rate * (i / (len(blocks) - 1)) for i in range(len(blocks))]
    if reverse:
        rates = list(reversed(rates))

    for block, rate in zip(blocks, rates):
        block.drop_path = DropPath(rate, scale_by_keep=scale_by_keep)
        if isinstance(block, BasicBlock):
            block.forward = types.MethodType(_basicblock_forward_with_droppath, block)
        elif isinstance(block, Bottleneck):
            block.forward = types.MethodType(_bottleneck_forward_with_droppath, block)


def build_resnet(
    depth: int,
    num_classes: int,
    dataset_style: str = "cifar",
    family: str = "cnn",
    signal_control: str = "bn",
    activation: str = "relu",
    pretrained: bool = False,
    dropout_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    drop_path_scale_by_keep: bool = True,
    drop_path_reverse: bool = False,
    compile_model: bool = False,
    compile_mode: str | None = None,
    compile_dynamic: bool | None = None,
    compile_fullgraph: bool | None = None,
    channels_last: bool = False,
    salu_kwargs: dict | None = None,
    activation_kwargs: dict | None = None,
    bn_index_overrides: dict | None = None,
):
    if depth == 18:
        model = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
    elif depth == 34:
        model = models.resnet34(weights=None if not pretrained else models.ResNet34_Weights.DEFAULT)
    elif depth == 50:
        model = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported ResNet depth: {depth}")

    if dataset_style.lower() in {"cifar", "cifar10", "cifar100"}:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    if dropout_rate and float(dropout_rate) > 0:
        model.fc = nn.Sequential(
            nn.Dropout(float(dropout_rate)),
            nn.Linear(model.fc.in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    _replace_modules(
        model,
        signal_control=signal_control,
        activation=activation,
        salu_kwargs=salu_kwargs,
        activation_kwargs=activation_kwargs,
        family=family,
        bn_state={"idx": 0},
        bn_index_overrides=_normalize_bn_index_overrides(bn_index_overrides),
    )

    _add_drop_path_to_resnet(
        model,
        drop_path_rate=drop_path_rate,
        scale_by_keep=drop_path_scale_by_keep,
        reverse=drop_path_reverse,
    )

    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    if compile_model:
        kwargs = {}
        if compile_mode:
            kwargs["mode"] = compile_mode
        if compile_dynamic is not None:
            kwargs["dynamic"] = bool(compile_dynamic)
        if compile_fullgraph is not None:
            kwargs["fullgraph"] = bool(compile_fullgraph)
        print(
            "torch.compile enabled: "
            f"mode={kwargs.get('mode', 'default')}, "
            f"dynamic={kwargs.get('dynamic', 'default')}, "
            f"fullgraph={kwargs.get('fullgraph', 'default')}"
        )
        try:
            model = torch.compile(model, **kwargs)
        except Exception as exc:
            print(f"torch.compile setup failed; continuing without compile: {exc}")

    return model


@register_model("resnet18")
def build_resnet18(**kwargs):
    return build_resnet(depth=18, **kwargs)


@register_model("resnet34")
def build_resnet34(**kwargs):
    return build_resnet(depth=34, **kwargs)


@register_model("resnet50")
def build_resnet50(**kwargs):
    return build_resnet(depth=50, **kwargs)
