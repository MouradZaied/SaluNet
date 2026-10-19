import math


def cosine_warmup_lambda(current_step: int, *, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0):
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def signal_cosine_lambda(
    current_step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
    decay_factor: float = 1.0,
    after_min: str = "hold_min",
):
    """LR lambda for SALU/SWALU/GALU parameter groups.

    after_min modes:
      - "legacy_bounce": do not clamp the scaled cosine phase. This reproduces
        older notebooks where the signal LR reaches its minimum early and then
        rises slightly near the end.
      - "hold_min": clamp at the minimum once the accelerated cosine reaches it.
      - "freeze_zero": set the signal LR to zero after the accelerated cosine
        reaches its minimum. This preserves the user's 83.27 CIFAR100 variant.
    """

    if total_steps <= 0:
        return 1.0

    if warmup_steps > 0 and current_step < warmup_steps:
        return min(1.0, 2.0 * float(current_step) / float(max(1, warmup_steps)))

    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    scaled_progress = progress * float(decay_factor)

    after_min = str(after_min or "hold_min").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "legacy": "legacy_bounce",
        "bounce": "legacy_bounce",
        "legacy_bounce": "legacy_bounce",
        "hold": "hold_min",
        "hold_min": "hold_min",
        "min": "hold_min",
        "freeze": "freeze_zero",
        "zero": "freeze_zero",
        "freeze_zero": "freeze_zero",
        "freeze_signal_after_min": "freeze_zero",
    }
    after_min = aliases.get(after_min, after_min)

    if after_min == "freeze_zero":
        if scaled_progress >= 1.0:
            return 0.0
        # Before the first minimum, use the accelerated cosine normally.
        pass
    elif after_min == "hold_min":
        scaled_progress = min(1.0, scaled_progress)
    elif after_min == "legacy_bounce":
        pass
    else:
        raise ValueError(
            "scheduler.signal.after_min must be one of: "
            "'legacy_bounce', 'hold_min', 'freeze_zero'. "
            f"Got {after_min!r}."
        )

    cosine = 0.5 * (1.0 + math.cos(math.pi * scaled_progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
