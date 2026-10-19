import math

import torch


class MixupCutmix:
    """MixUp/CutMix recipe matching the original SaluNet CIFAR behavior.

    Original behavior:
    - effective mix probability ramps during the first 10 epochs;
    - early training prefers MixUp;
    - CutMix becomes more likely later through progress-dependent sampling.
    """

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 0.0,
        use_cutmix: bool = True,
        ramp_epochs: int = 10,
        progressive: bool = True,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.use_cutmix = use_cutmix
        self.ramp_epochs = ramp_epochs
        self.progressive = progressive

    def __call__(self, images: torch.Tensor, targets: torch.Tensor, *, current_epoch: int = 0, max_epochs: int = 1):
        if self.prob <= 0:
            return images, targets, None, None, "none"

        if self.progressive:
            current_prob = self.prob * min(1.0, float(current_epoch + 1) / float(max(1, self.ramp_epochs)))
        else:
            current_prob = self.prob

        if torch.rand(1, device=images.device).item() >= current_prob:
            return images, targets, None, None, "none"

        progress = float(current_epoch) / float(max(1, max_epochs))
        do_mixup = True
        if self.use_cutmix:
            # Matches the original repo: MixUp dominates early, CutMix increases later.
            do_mixup = torch.rand(1, device=images.device).item() < (1.0 - progress)

        alpha = self.mixup_alpha if do_mixup else self.cutmix_alpha
        lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
        index = torch.randperm(images.size(0), device=images.device)

        if do_mixup:
            mixed = lam * images + (1.0 - lam) * images[index]
            return mixed, targets, targets[index], lam, "mixup"

        _, _, H, W = images.shape
        cut_rat = math.sqrt(1.0 - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx = torch.randint(W, (1,), device=images.device).item()
        cy = torch.randint(H, (1,), device=images.device).item()
        x1, y1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
        x2, y2 = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)

        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
        return mixed, targets, targets[index], lam, "cutmix"
