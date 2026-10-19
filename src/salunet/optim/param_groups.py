def build_parameter_groups(model, base_lr: float, weight_decay: float, signal_weight_decay: float = 0.0, signal_lr_factor: float = 2.0):
    """Create parameter groups with special treatment for SALU/SWALU/GALU parameters."""
    normal_params = []
    signal_params = []
    norm_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if any(key in lname for key in ("loga", "logb", "raw_a", "raw_b", "salu", "swalu", "galu", "alpha", "shift")):
            signal_params.append(param)
        elif any(key in lname for key in ("bn", "norm", "ln", "rmsnorm")):
            norm_params.append(param)
        else:
            normal_params.append(param)

    groups = []
    if normal_params:
        groups.append({"params": normal_params, "lr": base_lr, "weight_decay": weight_decay, "name": "normal"})
    if norm_params:
        groups.append({"params": norm_params, "lr": base_lr, "weight_decay": signal_weight_decay, "name": "norm"})
    if signal_params:
        groups.append({"params": signal_params, "lr": base_lr * signal_lr_factor, "weight_decay": signal_weight_decay, "name": "signal_control_activation"})
    return groups
