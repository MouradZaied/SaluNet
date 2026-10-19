import os
import time
from datetime import datetime

import torch
import pytorch_lightning as pl

from salunet.recipes.ema import ModelEma


def _move_batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_batch_to_device(v, device) for v in batch)
    return batch


class EMACallback(pl.Callback):
    """Maintain EMA weights and optionally evaluate them after each validation epoch.

    Logged metrics when ``evaluate_on_validation=True``:

    - ``val_ema/loss``
    - ``val_ema/acc``
    - ``val_ema/top5``

    This runs a second validation pass with the EMA model. It is intentionally
    explicit because EMA validation has a real compute cost.
    """

    def __init__(
        self,
        decay: float = 0.9999,
        evaluate_on_validation: bool = True,
        log_prefix: str = "val_ema",
        max_batches: int | None = None,
    ):
        super().__init__()
        self.decay = decay
        self.evaluate_on_validation = evaluate_on_validation
        self.log_prefix = log_prefix.rstrip("/")
        self.max_batches = max_batches
        self.ema = None
        self._pending_ema_state_dict = None

    def on_fit_start(self, trainer, pl_module):
        self.ema = ModelEma(pl_module.model, decay=self.decay)
        if self._pending_ema_state_dict is not None:
            self.ema.model.load_state_dict(self._pending_ema_state_dict)
            self._pending_ema_state_dict = None

    def state_dict(self):
        if self.ema is None:
            return {}
        return {
            "ema_model_state_dict": self.ema.model.state_dict(),
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict):
        # The EMA object is created in on_fit_start. If Lightning restores this
        # callback before then, the state can be safely ignored; best/checkpoint
        # files also store explicit EMA weights for evaluation.
        if "ema_model_state_dict" in state_dict:
            if self.ema is not None:
                self.ema.model.load_state_dict(state_dict["ema_model_state_dict"])
            else:
                self._pending_ema_state_dict = state_dict["ema_model_state_dict"]
        if "decay" in state_dict:
            self.decay = float(state_dict["decay"])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.update(pl_module.model)

    def _get_val_loader(self, trainer):
        """Return Lightning's already-created validation dataloader.

        Do not call ``trainer.datamodule.val_dataloader()`` here. Recreating a
        DataLoader at every validation epoch can spawn a fresh worker pool and
        eventually exhaust file descriptors when ``num_workers > 0`` and/or
        ``persistent_workers=True``. This callback performs a second EMA
        evaluation pass over the dataloader that Lightning already owns.
        """

        loaders = getattr(trainer, "val_dataloaders", None)
        if loaders is None:
            return None

        # Lightning may expose a single DataLoader, a list/tuple, or a dict.
        if isinstance(loaders, (list, tuple)):
            return loaders[0] if loaders else None
        if isinstance(loaders, dict):
            return next(iter(loaders.values()), None)
        return loaders

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ema is None or not self.evaluate_on_validation:
            return
        if getattr(trainer, "sanity_checking", False):
            return

        loader = self._get_val_loader(trainer)
        if loader is None:
            return

        device = pl_module.device
        ema_model = self.ema.model.to(device)
        ema_model.eval()

        total_loss = torch.tensor(0.0, device=device)
        total = torch.tensor(0.0, device=device)
        correct1 = torch.tensor(0.0, device=device)
        correct5 = torch.tensor(0.0, device=device)

        for batch_idx, batch in enumerate(loader):
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break

            batch = _move_batch_to_device(batch, device)
            x, y = batch[:2] if isinstance(batch, (list, tuple)) else (batch["x"], batch["y"])
            if getattr(pl_module, "channels_last", False) and x.ndim == 4:
                x = x.contiguous(memory_format=torch.channels_last)

            logits = ema_model(x)
            loss = pl_module.criterion(logits, y)

            batch_size = y.numel()
            total_loss += loss.detach() * batch_size
            total += batch_size

            pred1 = logits.argmax(dim=1)
            correct1 += pred1.eq(y).sum()

            k = min(5, logits.shape[1])
            pred5 = logits.topk(k, dim=1).indices
            correct5 += pred5.eq(y.view(-1, 1)).any(dim=1).sum()

        if total.item() == 0:
            return

        metrics = {
            f"{self.log_prefix}/loss": total_loss / total,
            f"{self.log_prefix}/acc": 100.0 * correct1 / total,
            f"{self.log_prefix}/top5": 100.0 * correct5 / total,
        }

        pl_module.log_dict(
            metrics,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        # Make metrics immediately visible to callbacks that run after this one,
        # including BestModelCheckpoint monitoring val_ema/acc.
        for key, value in metrics.items():
            trainer.callback_metrics[key] = value.detach()


class BestModelCheckpoint(pl.Callback):
    """Save separate best checkpoints for the raw model and the EMA model.

    Files saved by default:

    - ``best_mp_model.pth``: best main/raw model according to ``raw_monitor``.
    - ``best_ema_model.pth``: best EMA model according to ``ema_monitor``.
    - ``last.ckpt``: full Lightning checkpoint for resume, including optimizer
      and scheduler states. This is saved with ``trainer.save_checkpoint``.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        monitor: str = "val/acc",
        mode: str = "max",
        raw_monitor: str | None = "val/acc",
        raw_mode: str = "max",
        ema_monitor: str | None = "val_ema/acc",
        ema_mode: str = "max",
        save_raw_best: bool = True,
        save_ema_best: bool = True,
        save_last: bool = True,
        raw_filename: str = "best_mp_model.pth",
        ema_filename: str = "best_ema_model.pth",
        last_filename: str = "last.ckpt",
    ):
        super().__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.raw_monitor = raw_monitor or monitor
        self.raw_mode = raw_mode or mode
        self.ema_monitor = ema_monitor
        self.ema_mode = ema_mode
        self.save_raw_best = save_raw_best
        self.save_ema_best = save_ema_best
        self.save_last = save_last
        self.raw_filename = raw_filename
        self.ema_filename = ema_filename
        self.last_filename = last_filename
        self.best_raw = -float("inf") if self.raw_mode == "max" else float("inf")
        self.best_ema = -float("inf") if self.ema_mode == "max" else float("inf")
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _find_ema_callback(trainer):
        for callback in trainer.callbacks:
            if isinstance(callback, EMACallback) and callback.ema is not None:
                return callback
        return None

    @staticmethod
    def _metric_value(trainer, name):
        value = trainer.callback_metrics.get(name)
        if value is None:
            return None
        return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)

    @staticmethod
    def _is_better(value: float, best: float, mode: str) -> bool:
        return value > best if mode == "max" else value < best

    def _base_payload(self, trainer, pl_module, ema_callback):
        payload = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "raw_model_state_dict": pl_module.model.state_dict(),
            "hyper_parameters": pl_module.hparams,
        }
        if ema_callback is not None:
            payload["ema_model_state_dict"] = ema_callback.ema.model.state_dict()
        return payload

    def _save_raw_best(self, trainer, pl_module, value: float, ema_callback):
        path = os.path.join(self.save_dir, self.raw_filename)
        checkpoint = self._base_payload(trainer, pl_module, ema_callback)
        checkpoint.update(
            {
                "model_state_dict": pl_module.model.state_dict(),
                "monitor": self.raw_monitor,
                "monitor_value": value,
                "is_ema_checkpoint": False,
                "checkpoint_type": "main_model",
            }
        )
        torch.save(checkpoint, path)

    def _save_ema_best(self, trainer, pl_module, value: float, ema_callback):
        if ema_callback is None:
            return
        path = os.path.join(self.save_dir, self.ema_filename)
        checkpoint = self._base_payload(trainer, pl_module, ema_callback)
        checkpoint.update(
            {
                "model_state_dict": ema_callback.ema.model.state_dict(),
                "monitor": self.ema_monitor,
                "monitor_value": value,
                "is_ema_checkpoint": True,
                "checkpoint_type": "ema_model",
            }
        )
        torch.save(checkpoint, path)

    def on_validation_epoch_end(self, trainer, pl_module):
        ema_callback = self._find_ema_callback(trainer)

        if self.save_raw_best and self.raw_monitor:
            raw_value = self._metric_value(trainer, self.raw_monitor)
            if raw_value is not None and self._is_better(raw_value, self.best_raw, self.raw_mode):
                self.best_raw = raw_value
                self._save_raw_best(trainer, pl_module, raw_value, ema_callback)

        if self.save_ema_best and self.ema_monitor:
            ema_value = self._metric_value(trainer, self.ema_monitor)
            if ema_value is not None and self._is_better(ema_value, self.best_ema, self.ema_mode):
                self.best_ema = ema_value
                self._save_ema_best(trainer, pl_module, ema_value, ema_callback)

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.save_last:
            return
        path = os.path.join(self.save_dir, self.last_filename)
        trainer.save_checkpoint(path)



class EpochTimerCallback(pl.Callback):
    """Log epoch duration to the active Lightning loggers.

    This mirrors the timing callback from the original CIFAR notebook while
    keeping terminal output compact.
    """

    def __init__(self, metric_name: str = "time/epoch_seconds"):
        super().__init__()
        self.metric_name = metric_name
        self._start = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._start is None:
            return
        elapsed = time.time() - self._start
        pl_module.log(self.metric_name, elapsed, on_step=False, on_epoch=True, logger=True, prog_bar=False)
        self._start = None


class CleanConsoleDisplayCallback(pl.Callback):
    """Render compact, stable epoch summaries without changing training state."""

    SEPARATOR = "=" * 80
    SUB_SEPARATOR = "-" * 60

    def __init__(self, batch_interval: int = 50):
        super().__init__()
        self.batch_interval = max(1, int(batch_interval))
        self._epoch_start_time = None

    @staticmethod
    def _metric(trainer, name):
        value = trainer.callback_metrics.get(name)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    @staticmethod
    def _learning_rates(trainer):
        if not trainer.optimizers:
            return None, None
        normal_lr = None
        activation_lr = None
        for index, group in enumerate(trainer.optimizers[0].param_groups):
            name = group.get("name", f"group_{index}")
            if name == "signal_control_activation":
                activation_lr = float(group["lr"])
            elif normal_lr is None:
                normal_lr = float(group["lr"])
        return normal_lr, activation_lr

    @staticmethod
    def _parameter_statistics(model):
        grouped = {"SALU": {"a": [], "b": []}, "SWALU": {"a": [], "b": []}}
        nested_gate_salu_ids = {
            id(module.salu)
            for module in model.modules()
            if module.__class__.__name__.upper() in {"SWALU", "GALU"} and hasattr(module, "salu")
        }
        for module in model.modules():
            class_name = module.__class__.__name__.upper()
            label = "SWALU" if class_name == "SWALU" else "SALU" if class_name == "SALU" else None
            if label is None or not hasattr(module, "get_ab") or id(module) in nested_gate_salu_ids:
                continue
            a, b = module.get_ab()
            grouped[label]["a"].append(a.detach().float().reshape(-1).cpu())
            grouped[label]["b"].append(b.detach().float().reshape(-1).cpu())

        summaries = []
        for label in ("SALU", "SWALU"):
            if not grouped[label]["a"]:
                continue
            a = torch.cat(grouped[label]["a"])
            b = torch.cat(grouped[label]["b"])
            summaries.append(
                f"  {label}: a = {a.mean().item():.4f}±{a.std(unbiased=False).item():.4f}, "
                f"b = {b.mean().item():.4f}±{b.std(unbiased=False).item():.4f}"
            )
        return summaries

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = time.time()
        normal_lr, activation_lr = self._learning_rates(trainer)
        print(f"\n{self.SEPARATOR}")
        print(f"EPOCH {trainer.current_epoch + 1}/{trainer.max_epochs}")
        print(self.SEPARATOR)
        if normal_lr is not None:
            print(f"Learning rate (standard parameters): {normal_lr:.2e}")
        if activation_lr is not None:
            print(f"Learning rate (activation parameters a,b): {activation_lr:.2e}")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch_number = batch_idx + 1
        total_batches = trainer.num_training_batches
        if batch_number != 1 and batch_number % self.batch_interval != 0:
            return
        loss = self._metric(trainer, "train/loss_step")
        accuracy = self._metric(trainer, "train/acc_step")
        normal_lr, activation_lr = self._learning_rates(trainer)
        fields = [f"  Batch {batch_number:4d}/{int(total_batches):4d}"]
        if loss is not None:
            fields.append(f"Loss: {loss:.4f}")
        if accuracy is not None:
            fields.append(f"Acc: {accuracy:.2f}%")
        if normal_lr is not None:
            fields.append(f"LR_normal: {normal_lr:.2e}")
        if activation_lr is not None:
            fields.append(f"LR_activation: {activation_lr:.2e}")
        print(" | ".join(fields))

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        elapsed = time.time() - self._epoch_start_time if self._epoch_start_time is not None else 0.0
        num_batches = max(1, int(trainer.num_training_batches))
        val_loss = self._metric(trainer, "val/loss")
        val_acc = self._metric(trainer, "val/acc")
        val_top5 = self._metric(trainer, "val/top5")
        ema_acc = self._metric(trainer, "val_ema/acc")
        ema_top5 = self._metric(trainer, "val_ema/top5")

        print(f"\n{self.SUB_SEPARATOR}")
        print("VALIDATION RESULTS:")
        if val_loss is not None:
            print(f"  Loss: {val_loss:.4f}")
        if val_acc is not None or val_top5 is not None:
            print(f"  Top-1 accuracy: {val_acc:.2f}%       Top-5 accuracy: {val_top5:.2f}%")
        if ema_acc is not None or ema_top5 is not None:
            print(f"  EMA Top-1 accuracy: {ema_acc:.2f}%   EMA Top-5 accuracy: {ema_top5:.2f}%")

        print("\nEXECUTION TIME:")
        print(f"  Epoch duration: {elapsed:.2f}s")
        print(f"  Average time per batch: {elapsed / num_batches:.3f}s")
        print(f"  Finished at: {datetime.now().strftime('%H:%M:%S')}")
        print(self.SEPARATOR)
        summaries = self._parameter_statistics(pl_module.model)
        if summaries:
            print("PARAMETER STATISTICS:")
            for summary in summaries:
                print(summary)

class SaluDiagnosticsCallback(pl.Callback):
    """Log SALU/SWALU/GALU geometry, parameter gradient norms, and activation variances."""

    def __init__(self, every_n_steps: int = 100, max_modules: int = 16, log_variance: bool = True, log_grad_norm_ab: bool = True):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.max_modules = max_modules
        self.log_variance = log_variance
        self.log_grad_norm_ab = log_grad_norm_ab
        self.handles = []
        self.last_variances = {}

    def setup(self, trainer, pl_module, stage=None):
        if not self.log_variance:
            return
        count = 0
        for name, module in pl_module.model.named_modules():
            if hasattr(module, "get_geometry") and count < self.max_modules:
                self.handles.append(module.register_forward_hook(self._make_forward_hook(name)))
                count += 1

    def teardown(self, trainer, pl_module, stage=None):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_forward_hook(self, name):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                with torch.no_grad():
                    self.last_variances[f"diagnostics/{name}/output_var"] = output.detach().float().var().item()
            if inputs and isinstance(inputs[0], torch.Tensor):
                x = inputs[0]
                if x.requires_grad:
                    x.register_hook(lambda grad, n=name: self.last_variances.__setitem__(f"diagnostics/{n}/grad_x_norm", grad.detach().float().norm().item()))
        return hook

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n_steps != 0:
            return
        logs = dict(self.last_variances)
        self.last_variances.clear()

        count = 0
        for name, module in pl_module.model.named_modules():
            if hasattr(module, "get_geometry") and count < self.max_modules:
                geom = module.get_geometry()
                for k, v in geom.items():
                    logs[f"diagnostics/{name}/{k}"] = float(v.detach().cpu())
                if self.log_grad_norm_ab:
                    for pname, p in module.named_parameters(recurse=True):
                        if p.grad is not None and any(key in pname.lower() for key in ("loga", "logb", "raw_a", "raw_b")):
                            logs[f"diagnostics/{name}/grad_{pname}_norm"] = float(p.grad.detach().float().norm().cpu())
                count += 1

        if logs:
            pl_module.log_dict(logs, on_step=True, on_epoch=False, logger=True, prog_bar=False)
