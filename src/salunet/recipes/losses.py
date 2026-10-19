import torch
import torch.nn as nn


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing
    Reduces overconfidence and improves generalization
    """
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        targets_one_hot = torch.zeros_like(logits)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        smoothed_targets = targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = - (smoothed_targets * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss