import math

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics

from salunet.models import build_model
from salunet.models.initialization import apply_initialization
from salunet.optim.param_groups import build_parameter_groups
from salunet.optim.schedulers import signal_cosine_lambda
from salunet.recipes.losses import LabelSmoothingCrossEntropy
from salunet.recipes.mix import MixupCutmix


class ClassificationModule(pl.LightningModule):
    """Generic LightningModule for SaluNet classification experiments.

    The CIFAR path intentionally preserves the behavior of the original working
    Lightning repo: progressive MixUp/CutMix, separate LR lambdas for standard
    and SALU/SWALU/GALU parameters, percentage metrics, and EMA handled by a
    callback.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        model_cfg = cfg["model"].copy()
        model_name = model_cfg.pop("name")
        signal_cfg = cfg.get("signal_control", {"type": "none"})
        act_cfg = cfg.get("activation", {"type": "relu"})
        salu_cfg = cfg.get("salu", {})

        signal_kwargs = {
            "a_init": signal_cfg.get("a_init", salu_cfg.get("a_init", 1.0)),
            "b_init": signal_cfg.get("b_init", salu_cfg.get("b_init", 0.1)),
            "positivity": signal_cfg.get("positivity", salu_cfg.get("positivity", "exp_log")),
        }
        activation_kwargs = {
            "a_init": act_cfg.get("a_init", 1.0),
            "b_init": act_cfg.get("b_init", 1.0),
            "positivity": act_cfg.get("positivity", salu_cfg.get("positivity", "exp_log")),
        }

        self.model = build_model(
            model_name,
            num_classes=cfg["num_classes"],
            signal_control=signal_cfg.get("type", "none"),
            activation=act_cfg.get("type", "relu"),
            salu_kwargs=signal_kwargs,
            activation_kwargs=activation_kwargs,
            bn_index_overrides=signal_cfg.get("bn_index_overrides", {}),
            **model_cfg,
        )

        apply_initialization(self.model, cfg)

        self.criterion = LabelSmoothingCrossEntropy(smoothing=cfg.get("loss", {}).get("label_smoothing", 0.0))
        if model_name == "vit_cifar":
            from salunet.recipes.vit_loss import ViTLabelSmoothingCrossEntropy
            self.criterion = ViTLabelSmoothingCrossEntropy(
                cfg["num_classes"], cfg.get("loss", {}).get("label_smoothing", 0.1))
        mix_cfg = cfg.get("mix", {})
        self.mix = MixupCutmix(
            mixup_alpha=mix_cfg.get("mixup_alpha", 0.2),
            cutmix_alpha=mix_cfg.get("cutmix_alpha", 1.0),
            prob=mix_cfg.get("mix_prob", 0.0),
            use_cutmix=mix_cfg.get("use_cutmix", True),
            ramp_epochs=mix_cfg.get("ramp_epochs", 10),
            progressive=mix_cfg.get("progressive", True),
        )

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg["num_classes"])
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg["num_classes"])
        self.val_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=cfg["num_classes"], top_k=min(5, cfg["num_classes"]))
        self.train_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=cfg["num_classes"], top_k=min(5, cfg["num_classes"]))
        self.channels_last = bool(cfg.get("model", {}).get("channels_last", cfg.get("trainer", {}).get("channels_last", False)))

    def setup(self, stage=None):
        # The CNN setup/execution path is unchanged.
        model_cfg = self.cfg.get("model", {})
        if (stage == "fit" and model_cfg.get("name") == "vit_cifar"
                and model_cfg.get("compile_model", False)
                and not getattr(self, "_vit_compiled", False)):
            if not hasattr(self.model, "compile"):
                raise RuntimeError("ViT compilation requires a PyTorch version with Module.compile.")
            self.model.compile(mode=model_cfg.get("compile_mode", "default"), fullgraph=False)
            self._vit_compiled = True

    def forward(self, x):
        return self.model(x)

    def _loss(self, logits, y, y2=None, lam=None):
        if y2 is None or lam is None:
            return self.criterion(logits, y)
        return lam * self.criterion(logits, y) + (1.0 - lam) * self.criterion(logits, y2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.channels_last and x.ndim == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        x, y1, y2, lam, mix_type = self.mix(
            x,
            y,
            current_epoch=int(self.current_epoch),
            max_epochs=int(self.cfg.get("epochs", 1)),
        )
        logits = self(x)
        loss = self._loss(logits, y1, y2, lam)

        # Match the original repo: true train accuracy is updated only on
        # batches without MixUp/CutMix; mixed batches log an approximate step acc.
        if y2 is None:
            self.train_acc(logits.detach(), y1)
            self.train_top5(logits.detach(), y1)
            approx_acc = self.train_acc.compute() * 100.0
        else:
            approx_acc = (logits.detach().argmax(dim=1) == y1).float().mean() * 100.0

        self.log("train/loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/acc_step", approx_acc, prog_bar=True, on_step=True, on_epoch=False)
        if lam is not None:
            self.log("train/mix_lam", float(lam), on_step=True, on_epoch=False)

        opt = self.optimizers(use_pl_optimizer=False)
        if opt is not None and hasattr(opt, "param_groups"):
            for idx, group in enumerate(opt.param_groups):
                name = group.get("name", f"group_{idx}")
                self.log(f"lr/{name}", group["lr"], on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def on_train_epoch_start(self):
        self.train_acc.reset()
        self.train_top5.reset()

    def on_train_epoch_end(self):
        train_acc = self.train_acc.compute() * 100.0
        train_top5 = self.train_top5.compute() * 100.0
        self.log("train/acc", train_acc, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/top5", train_top5, prog_bar=False, on_step=False, on_epoch=True)
        self.train_acc.reset()
        self.train_top5.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.channels_last and x.ndim == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits, y)
        self.val_top5(logits, y)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_acc.reset()
        self.val_top5.reset()

    def on_validation_epoch_end(self):
        self.log("val/acc", self.val_acc.compute() * 100.0, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/top5", self.val_top5.compute() * 100.0, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.val_acc.reset()
        self.val_top5.reset()

    def configure_optimizers(self):
        if self.cfg.get("model", {}).get("name") == "vit_cifar":
            from salunet.optim.vit_recipe import build_vit_optimizers
            return build_vit_optimizers(self.model, self.cfg)
        opt_cfg = self.cfg.get("optimizer", {})
        sched_cfg = self.cfg.get("scheduler", {})
        lr = float(opt_cfg.get("lr", 0.1))
        signal_lr_factor = float(opt_cfg.get("signal_lr_factor", self.cfg.get("activation", {}).get("lr_factor", 2.0)))

        groups = build_parameter_groups(
            self.model,
            base_lr=lr,
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
            signal_weight_decay=float(opt_cfg.get("signal_weight_decay", self.cfg.get("activation", {}).get("weight_decay", 0.0))),
            signal_lr_factor=signal_lr_factor,
        )
        optimizer = optim.SGD(
            groups,
            lr=lr,
            momentum=float(opt_cfg.get("momentum", 0.9)),
            nesterov=bool(opt_cfg.get("nesterov", True)),
        )

        total_steps = int(self.trainer.estimated_stepping_batches) if self.trainer is not None else 0
        steps_per_epoch = max(1, total_steps // max(1, int(self.cfg.get("epochs", 1))))
        warmup_steps = int(sched_cfg.get("warmup_epochs", 0) * steps_per_epoch)
        min_lr = float(sched_cfg.get("min_lr", 0.0))
        signal_sched_cfg = sched_cfg.get("signal", {}) or {}
        cosine_decay_factor = float(signal_sched_cfg.get("decay_factor", sched_cfg.get("cosine_decay_factor", 1.0)))
        signal_after_min = signal_sched_cfg.get("after_min", sched_cfg.get("signal_after_min", "hold_min"))

        min_lr_ratio_normal = min_lr / lr if lr > 0 else 0.0
        signal_lr = lr * signal_lr_factor
        min_lr_ratio_signal = min_lr / signal_lr if signal_lr > 0 else 0.0

        def lr_lambda_normal(step):
            if warmup_steps > 0 and step < warmup_steps:
                return (float(step) / float(max(1, warmup_steps))) ** 0.5
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return min_lr_ratio_normal + (1.0 - min_lr_ratio_normal) * cosine

        def lr_lambda_signal(step):
            return signal_cosine_lambda(
                int(step),
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=min_lr_ratio_signal,
                decay_factor=cosine_decay_factor,
                after_min=signal_after_min,
            )

        lambdas = []
        for group in groups:
            if group.get("name") == "signal_control_activation":
                lambdas.append(lr_lambda_signal)
            else:
                lambdas.append(lr_lambda_normal)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
