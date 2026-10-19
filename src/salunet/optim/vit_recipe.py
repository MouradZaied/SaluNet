"""Adam and epoch-wise schedule from the supplied SaluNet-T recipe."""
import math
import torch
from salunet.layers.vit_salu import ViTSALU
from salunet.activations.galu import GALU
from salunet.signal_control.dyt import DyT


def build_vit_optimizers(model, cfg):
    opt = cfg.get("optimizer", {})
    sched = cfg.get("scheduler", {})
    lr = float(opt.get("lr", 1e-3))
    factor = float(opt.get("signal_lr_factor", 0.1))
    gate_factor = opt.get("galu_lr_factor")
    grouped, seen = {"network": [], "dyt_no_decay": [], "activations": [], "galu": []}, set()
    galu_ids = {id(p) for m in model.modules() if isinstance(m, GALU)
                for p in m.parameters()}
    for m in model.modules():
        if isinstance(m, (ViTSALU, DyT)):
            for p in m.parameters(recurse=False):
                if id(p) in seen or not p.requires_grad:
                    continue
                seen.add(id(p))
                name = ("dyt_no_decay" if isinstance(m, DyT) else
                        "galu" if gate_factor is not None and id(p) in galu_ids else "activations")
                grouped[name].append(p)
    grouped["network"] = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    groups = []
    for name, params in grouped.items():
        if not params:
            continue
        multiplier = factor if name == "activations" else float(gate_factor) if name == "galu" else 1.0
        decay = float(opt.get("weight_decay", 5e-5)) if name == "network" else (
            0.0 if name == "dyt_no_decay" else float(opt.get("signal_weight_decay", 0.0)))
        groups.append(dict(params=params, name=name, lr=lr*multiplier, weight_decay=decay))
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999))
    epochs = int(cfg.get("epochs", 200))
    warmup = int(sched.get("warmup_epochs", 5))
    if not 0 <= warmup < epochs:
        raise ValueError("warmup_epochs must be in [0, epochs).")
    def schedule(epoch, floor, speed):
        if epoch <= warmup:
            return epoch / max(1, warmup)
        progress = min(max((epoch-warmup) / max(1, epochs-warmup) * speed, 0.0), 1.0)
        return floor + (1-floor) * 0.5 * (1+math.cos(math.pi*progress))
    schedules = []
    for group in groups:
        special = group["name"] in {"activations", "galu"}
        floor_lr = float(sched.get("signal_min_lr", 1e-6) if special else sched.get("min_lr", 1e-5))
        if not 0 <= floor_lr <= group["lr"]:
            raise ValueError("Scheduler floor must be between zero and initial group LR.")
        floor = floor_lr / group["lr"]
        speed = float(sched.get("signal_decay_factor", 1.0)) if special else 1.0
        schedules.append(lambda epoch, f=floor, s=speed: schedule(epoch, f, s))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedules)
    return {"optimizer": optimizer, "lr_scheduler": {
        "scheduler": scheduler, "interval": "epoch", "frequency": 1}}
