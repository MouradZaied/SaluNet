import torch
import torch.nn as nn


class DropPath(nn.Module):
    """DropPath / stochastic depth.

    This is the per-sample residual-branch drop used in modern CNN/ViT
    recipes. It is applied to the residual branch before adding the shortcut.

    Args:
        drop_prob: probability of dropping the residual branch.
        scale_by_keep: when True, divide kept paths by ``1 - drop_prob``.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)
        return x * mask

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}, scale_by_keep={self.scale_by_keep}"
