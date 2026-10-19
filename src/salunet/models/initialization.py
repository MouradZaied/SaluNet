import math
from typing import Any

import torch.nn as nn


def salu_fanin_init(module: nn.Module, a: float = 1.0) -> None:
    """Initialize Conv2d/Linear weights with Var(w)=1/(a^2 * fan_in).

    This is an experimental SALU-aware initialization derived from the local
    forward condition sigma_w^2 * a^2 = 1. It is not the default behavior and
    should be enabled only through config for ablation runs.
    """

    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        return

    fan_in = nn.init._calculate_correct_fan(module.weight, mode="fan_in")
    std = 1.0 / (float(a) * math.sqrt(float(fan_in)))
    nn.init.normal_(module.weight, mean=0.0, std=std)

    if module.bias is not None:
        nn.init.zeros_(module.bias)


def apply_initialization(model: nn.Module, cfg: dict[str, Any]) -> None:
    """Apply optional model initialization configured under ``initialization``."""

    init_cfg = cfg.get("initialization", {}) or {}
    init_type = init_cfg.get("type", "default")

    if init_type in ("default", None):
        return

    if init_type != "salu_fanin":
        raise ValueError(f"Unknown initialization type: {init_type}")

    a = float(init_cfg.get("a", 1.0))
    apply_to = init_cfg.get("apply_to", {}) or {}
    apply_conv = bool(apply_to.get("conv", True))
    apply_linear = bool(apply_to.get("linear", True))

    for module in model.modules():
        if isinstance(module, nn.Conv2d) and apply_conv:
            salu_fanin_init(module, a=a)
        elif isinstance(module, nn.Linear) and apply_linear:
            salu_fanin_init(module, a=a)
