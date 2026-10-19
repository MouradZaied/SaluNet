import math

import torch
import torch.nn as nn

from salunet.layers.salu import SALU


class GALU(nn.Module):
    """SALU-gated GELU-like activation.

    Paper definition:
        h(x) = sqrt(2/pi) * (x + 0.044715*x^3)
        GALU(x; a,b) = 0.5 * x * (1 + SALU_phi(h(x); a,b))

    Important: 0.5*x*(1+SALU_phi(x)) is SWALU, not GALU.
    """

    def __init__(self, a_init: float = 1.0, b_init: float = 1.0, eps: float = 1e-6, positivity: str = "exp_log", name: str = "") -> None:
        super().__init__()
        self.salu = SALU(a_init=a_init, b_init=b_init, eps=eps, positivity=positivity, name=name or "galu")

    @staticmethod
    def gelu_tanh_argument(x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.gelu_tanh_argument(x)
        return 0.5 * x * (1.0 + self.salu.phi(h))

    def get_ab(self):
        return self.salu.get_ab()

    def get_geometry(self):
        return self.salu.get_geometry()
