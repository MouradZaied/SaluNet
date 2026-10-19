import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


class SALU(nn.Module):
    """Saturated Adaptive Linear Unit.

    Implementation coordinates are always `a,b`; analysis coordinates are
    derived as `S,W,gain`.

    positivity:
        - "exp_log": train unconstrained loga/logb, then exp.
        - "abs": train raw_a/raw_b, then abs().clamp_min(eps).
        - "abs_eps_legacy": train raw_a/raw_b, then abs() + eps. This exactly preserves older tested notebook recipes.
    """

    def __init__(
        self,
        a_init: float = 1.0,
        b_init: float = 0.1,
        eps: float = 1e-6,
        positivity: str = "exp_log",
        name: str = "",
    ) -> None:
        super().__init__()
        if a_init <= 0 or b_init <= 0:
            raise ValueError("a_init and b_init must be positive.")

        # Canonical names are "exp_log" and "abs". Accept common aliases
        # because experiment configs often use human-readable "exp/log".
        positivity = positivity.lower().replace("-", "_")
        if positivity in {"exp/log", "explog", "log_exp"}:
            positivity = "exp_log"
        if positivity in {"abs_eps", "abs+eps", "legacy_abs", "abs_legacy"}:
            positivity = "abs_eps_legacy"
        if positivity not in {"exp_log", "abs", "abs_eps_legacy"}:
            raise ValueError("positivity must be 'exp_log'/'exp/log', 'abs', or 'abs_eps_legacy'.")

        self.eps = eps
        self.positivity = positivity
        self.name = name

        if positivity == "exp_log":
            self.loga = nn.Parameter(torch.tensor(math.log(a_init), dtype=torch.float32))
            self.logb = nn.Parameter(torch.tensor(math.log(b_init), dtype=torch.float32))
        else:
            self.raw_a = nn.Parameter(torch.tensor(float(a_init), dtype=torch.float32))
            self.raw_b = nn.Parameter(torch.tensor(float(b_init), dtype=torch.float32))

    def get_ab(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.positivity == "exp_log":
            return torch.exp(self.loga), torch.exp(self.logb)
        if self.positivity == "abs_eps_legacy":
            return self.raw_a.abs() + self.eps, self.raw_b.abs() + self.eps
        return self.raw_a.abs().clamp_min(self.eps), self.raw_b.abs().clamp_min(self.eps)

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.get_ab()
        # Compute in fp32 for numerical stability, then return to original dtype.
        dtype = x.dtype
        z = x.float()
        a = a.float()
        b = b.float()
        y = (a * z) / torch.sqrt(1.0 + a * b * z.square() )   #!+ self.eps
        return y.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.phi(x)

    @torch.no_grad()
    def get_geometry(self) -> Dict[str, torch.Tensor]:
        a, b = self.get_ab()
        S = torch.sqrt(a / b)
        W = 1.0 / torch.sqrt(a * b)
        gain = S / W
        return {"a": a.detach(), "b": b.detach(), "S": S.detach(), "W": W.detach(), "gain": gain.detach()}

    @torch.no_grad()
    def get_params_for_logging(self, prefix: str = "") -> Dict[str, float]:
        geom = self.get_geometry()
        base = prefix or self.name or "salu"
        return {f"{base}/{k}": float(v.detach().cpu()) for k, v in geom.items()}
