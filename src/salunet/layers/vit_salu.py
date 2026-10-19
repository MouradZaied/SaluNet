"""SALU numerical path from the supplied CIFAR Transformer.

The CNN SALU class is unchanged. This subclass reuses its parameters and
geometry helpers, while preserving the ViT rsqrt/input-dtype implementation.
"""
import torch
from salunet.layers.salu import SALU

class ViTSALU(SALU):
    def phi(self, x):
        a, b = self.get_ab()
        
        return a * x * torch.rsqrt(1.0 + a * b * x.square())
