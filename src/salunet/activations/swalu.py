import torch
import torch.nn as nn

from salunet.layers.salu import SALU


class SWALU(nn.Module):
    """SALU-gated Swish-like activation.

    Paper definition:
        SWALU(x; a,b) = 0.5 * x * (1 + SALU_phi(x; a,b))
    """

    def __init__(self, a_init: float = 1.0, b_init: float = 1.0, eps: float = 1e-6, positivity: str = "exp_log", name: str = "") -> None:
        super().__init__()
        self.salu = SALU(a_init=a_init, b_init=b_init, eps=eps, positivity=positivity, name=name or "swalu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + self.salu.phi(x))

    def get_ab(self):
        return self.salu.get_ab()

    def get_geometry(self):
        return self.salu.get_geometry()
