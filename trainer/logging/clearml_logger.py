import os
from typing import Any

import torch

from trainer.logging.tensorboard_logger import TensorboardLogger
from trainer.trainer_utils import is_clearml_available
from trainer.utils.distributed import rank_zero_only

if is_clearml_available():
    from clearml import Task  # pylint: disable=import-error
else:
    raise ImportError("ClearML is not installed. Please install it with `pip install clearml`")


class ClearMLLogger(TensorboardLogger):
    """ClearML Logger using TensorBoard in the background.

    TODO:
        - Add hyperparameter handling
        - Use ClearML logger for plots
        - Handle continuing training

    Args:
        output_uri (str): URI of the ClearML repository.
        local_path (str): Path to the local directory where the model is saved.
        project_name (str): Name of the ClearML project.
        task_name (str): Name of the ClearML task.
        tags (str): Comma separated list of tags to add to the ClearML task.
    """

    def __init__(
        self,
        output_uri: str,
        local_path: str,
        project_name: str,
        task_name: str,
        tags: str = None,
    ):
        self._context = None
        self.local_path = local_path
        self.task_name = task_name
        self.tags = tags.split(",") if tags else []
        self.run = Task.init(project_name=project_name, task_name=task_name, tags=self.tags, output_uri=output_uri)

        if tags:
            for tag in tags.split(","):
                self.run.add_tag(tag)

        super().__init__("run", None)

    @rank_zero_only
    def add_config(self, config):
        """Upload config file(s) to ClearML."""
        self.add_text("run_config", f"{config.to_json()}", 0)
        self.run.connect_configuration(name="model_config", configuration=config.to_dict())
        self.run.set_comment(config.run_description)
        self.run.upload_artifact("model_config", config.to_dict())
        self.run.upload_artifact("configs", artifact_object=os.path.join(self.local_path, "*.json"))

    @rank_zero_only
    def add_artifact(self, file_or_dir, name, **kwargs):  # pylint: disable=unused-argument, arguments-differ
        """Upload artifact to ClearML."""
        self.run.upload_artifact(name, artifact_object=file_or_dir)

    @staticmethod
    @rank_zero_only
    def save_model(state: Any, path: str):
        torch.save(state, path)
