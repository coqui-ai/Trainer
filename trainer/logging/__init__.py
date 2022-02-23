import os

from trainer.logging.console_logger import ConsoleLogger


def get_mlflow_tracking_url():
    if "MLFLOW_TRACKING_URI" in os.environ:
        return os.environ["MLFLOW_TRACKING_URI"]
    return None


def get_ai_repo_url():
    if "AIM_TRACKING_URI" in os.environ:
        return os.environ["AIM_TRACKING_URI"]
    return None


def logger_factory(config, output_path):
    run_name = config.run_name
    project_name = config.project_name
    log_uri = config.logger_uri if config.logger_uri else output_path

    if config.dashboard_logger == "tensorboard":
        from trainer.logging.tensorboard_logger import TensorboardLogger

        model_name = f"{project_name}@{run_name}" if project_name else run_name
        dashboard_logger = TensorboardLogger(log_uri, model_name=model_name)

    elif config.dashboard_logger == "wandb":
        from trainer.logging.wandb_logger import WandbLogger

        dashboard_logger = WandbLogger(
            project=project_name,
            name=run_name,
            config=config,
            entity=config.wandb_entity,
        )

    elif config.dashboard_logger == "mlflow":
        from trainer.logging.mlflow_logger import MLFlowLogger

        dashboard_logger = MLFlowLogger(log_uri=log_uri, model_name=project_name)

    elif config.dashboard_logger == "aim":
        from trainer.logging.aim_logger import AimLogger

        dashboard_logger = AimLogger(repo=log_uri, model_name=project_name)

    elif config.dashboard_logger == "clearml":
        from trainer.logging.clearml_logger import ClearMLLogger

        dashboard_logger = ClearMLLogger(output_uri=log_uri, project_name=project_name, task_name=run_name)

    else:
        raise ValueError(f"Unknown dashboard logger: {config.dashboard_logger}")

    dashboard_logger.add_config(config)
    return dashboard_logger
