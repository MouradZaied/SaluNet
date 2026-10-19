from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger


def build_loggers(cfg: dict, experiment_name: str):
    logging_cfg = cfg.get("logging", {})
    loggers = []

    wandb_cfg = logging_cfg.get("wandb", {})
    if wandb_cfg.get("enabled", logging_cfg.get("use_wandb", False)):
        loggers.append(
            WandbLogger(
                project=wandb_cfg.get("project", logging_cfg.get("wandb_project", "salunet")),
                entity=wandb_cfg.get("entity", logging_cfg.get("wandb_entity")),
                name=wandb_cfg.get("name", experiment_name),
                tags=wandb_cfg.get("tags", []),
                log_model=wandb_cfg.get("log_model", False),
                offline=wandb_cfg.get("offline", False),
                save_dir=wandb_cfg.get("save_dir", "./wandb_logs"),
                config=cfg,
            )
        )

    tb_cfg = logging_cfg.get("tensorboard", {})
    if tb_cfg.get("enabled", False):
        loggers.append(TensorBoardLogger(save_dir=tb_cfg.get("save_dir", "logs/tensorboard"), name=experiment_name))

    if not loggers and logging_cfg.get("csv", {}).get("enabled", True):
        loggers.append(CSVLogger(save_dir=logging_cfg.get("csv", {}).get("save_dir", "logs/csv"), name=experiment_name))

    return loggers
