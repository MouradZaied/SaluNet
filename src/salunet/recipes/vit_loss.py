import torch
import torch.nn as nn

class ViTLabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing used by the original ViT-CIFAR implementation."""

    def __init__(self, classes: int, smoothing: float) -> None:
        super().__init__()
        self.classes = classes
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = logits.log_softmax(dim=-1)
        with torch.no_grad():
            distribution = torch.full_like(
                log_probabilities, self.smoothing / (self.classes - 1)
            )
            distribution.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-distribution * log_probabilities, dim=-1))

