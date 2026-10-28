import argparse
import logging
import os
import sys
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor

# Allow running without installing the package.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from salunet.data import build_datamodule
from salunet.lightning.callbacks import (
    BestModelCheckpoint,
    CleanConsoleDisplayCallback,
    EMACallback,
    EpochTimerCallback,
    SaluDiagnosticsCallback,
)
from salunet.lightning.classification_module import ClassificationModule
from salunet.logging.loggers import build_loggers
from salunet.utils.config import load_config
from salunet.utils.seed import set_all_seeds


class _WeightsOnlyMessageFilter(logging.Filter):
    """Hide only Lightning's repetitive torch.load compatibility notice."""

    def filter(self, record):
        message = record.getMessage()
        return not ("weights_only" in message and "was not set" in message and "defaulting to `False`" in message)


def _silence_weights_only_notice():
    warnings.filterwarnings(
        "ignore",
        message=r".*weights_only.*was not set.*defaulting to [`']?False[`']?.*",
    )
    message_filter = _WeightsOnlyMessageFilter()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(message_filter)
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.addFilter(message_filter)
            for handler in logger.handlers:
                handler.addFilter(message_filter)


def train(config_path: str, resume_path: str | None = None):
    _silence_weights_only_notice()
    cfg = load_config(config_path)

    gpu_cfg = cfg.get("gpu_optimizations", {})
    if gpu_cfg.get("enabled", True):
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", str(gpu_cfg.get("cuda_launch_blocking", "0")))
        os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", str(gpu_cfg.get("torch_cudnn_v8_api_enabled", "1")))
        if gpu_cfg.get("pytorch_cuda_alloc_conf"):
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", str(gpu_cfg.get("pytorch_cuda_alloc_conf")))
        if gpu_cfg.get("cublas_workspace_config"):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", str(gpu_cfg.get("cublas_workspace_config")))

    torch.set_float32_matmul_precision(cfg.get("trainer", {}).get("matmul_precision", "high"))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = cfg.get("trainer", {}).get("allow_tf32", True)
        torch.backends.cudnn.allow_tf32 = cfg.get("trainer", {}).get("allow_tf32", True)
        torch.backends.cudnn.benchmark = cfg.get("trainer", {}).get("cudnn_benchmark", True)

    set_all_seeds(cfg.get("seed", 42), deterministic=cfg.get("deterministic", False))

    dm = build_datamodule(cfg["data"])
    model = ClassificationModule(cfg)

    experiment_name = cfg.get("experiment_name", os.path.splitext(os.path.basename(config_path))[0])
    loggers = build_loggers(cfg, experiment_name)

    callbacks = []
    ema_cfg = cfg.get("ema", {})
    if ema_cfg.get("use_ema", False):
        callbacks.append(
            EMACallback(
                decay=float(ema_cfg.get("decay", 0.9999)),
                evaluate_on_validation=ema_cfg.get("evaluate_on_validation", True),
                log_prefix=ema_cfg.get("log_prefix", "val_ema"),
                max_batches=ema_cfg.get("max_val_batches"),
            )
        )

    ckpt_cfg = cfg.get("checkpoint", {})
    if ckpt_cfg.get("save_checkpoints", True):
        callbacks.append(
            BestModelCheckpoint(
                save_dir=ckpt_cfg.get("save_dir", "checkpoints"),
                monitor=ckpt_cfg.get("monitor", "val/acc"),
                mode=ckpt_cfg.get("mode", "max"),
                raw_monitor=ckpt_cfg.get("raw_monitor", "val/acc"),
                raw_mode=ckpt_cfg.get("raw_mode", "max"),
                ema_monitor=ckpt_cfg.get("ema_monitor", "val_ema/acc"),
                ema_mode=ckpt_cfg.get("ema_mode", "max"),
                save_raw_best=ckpt_cfg.get("save_raw_best", True),
                save_ema_best=ckpt_cfg.get("save_ema_best", True),
                save_last=ckpt_cfg.get("save_last", True),
                raw_filename=ckpt_cfg.get("raw_filename", "best_mp_model.pth"),
                ema_filename=ckpt_cfg.get("ema_filename", "best_ema_model.pth"),
                last_filename=ckpt_cfg.get("last_filename", "last.ckpt"),
            )
        )

    diag_cfg = cfg.get("logging", {}).get("diagnostics", {})
    if diag_cfg.get("enabled", False):
        callbacks.append(
            SaluDiagnosticsCallback(
                every_n_steps=diag_cfg.get("every_n_steps", 100),
                max_modules=diag_cfg.get("max_modules", 16),
                log_variance=diag_cfg.get("log_variance", True),
                log_grad_norm_ab=diag_cfg.get("log_grad_norm_ab", True),
            )
        )



    lr_monitor_cfg = cfg.get("logging", {}).get("learning_rate", {})
    if lr_monitor_cfg.get("enabled", True):
        callbacks.append(
            LearningRateMonitor(
                logging_interval=lr_monitor_cfg.get("logging_interval", "step"),
                log_momentum=lr_monitor_cfg.get("log_momentum", False),
            )
        )

    timer_cfg = cfg.get("logging", {}).get("epoch_timer", {})
    if timer_cfg.get("enabled", True):
        callbacks.append(EpochTimerCallback(metric_name=timer_cfg.get("metric_name", "time/epoch_seconds")))

    console_cfg = cfg.get("logging", {}).get("console", {})
    callbacks.append(CleanConsoleDisplayCallback(batch_interval=console_cfg.get("batch_interval", 50)))

    trainer_cfg = cfg.get("trainer", {})
    trainer = pl.Trainer(
        max_epochs=cfg.get("epochs", 1),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", "auto"),
        precision=trainer_cfg.get("precision", "32-true"),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 0.0),
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=cfg.get("logging", {}).get("log_every_n_steps", 50),
        fast_dev_run=trainer_cfg.get("fast_dev_run", False),
        deterministic=cfg.get("deterministic", False),
        benchmark=trainer_cfg.get("cudnn_benchmark", True),
        check_val_every_n_epoch=trainer_cfg.get("check_val_every_n_epoch", 1),
        enable_model_summary=trainer_cfg.get("enable_model_summary", True),
        enable_progress_bar=False,
    )

    # Trainer/logger construction may install additional handlers.
    _silence_weights_only_notice()
    trainer.fit(model, datamodule=dm, ckpt_path=resume_path)
    return trainer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to an experiment YAML configuration.")
    parser.add_argument("--resume", default=None, help="Full Lightning checkpoint to resume.")
    args = parser.parse_args()
    train(args.config, resume_path=args.resume)
