from trainer.logging.console_logger import ConsoleLogger
from trainer.logging.tensorboard_logger import TensorboardLogger
from trainer.logging.wandb_logger import WandbLogger


def init_dashboard_logger(config):
    project_name = config.get("model", "coqui-model")

    if config.dashboard_logger == "tensorboard":
        dashboard_logger = TensorboardLogger(config.output_log_path, model_name=project_name)

    elif config.dashboard_logger == "wandb":
        if config.project_name:
            project_name = config.project_name

        dashboard_logger = WandbLogger(
            project=project_name,
            name=config.run_name,
            config=config,
            entity=config.wandb_entity,
        )

    dashboard_logger.add_text("model-config", f"<pre>{config.to_json()}</pre>", 0)
    return dashboard_logger
