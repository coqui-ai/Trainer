import os
import shutil
import tempfile
import traceback

import soundfile as sf
import torch

from trainer.logging.base_dash_logger import BaseDashboardLogger
from trainer.logging.tensorboard_logger import TensorboardLogger
from trainer.trainer_utils import is_clearml_available
from trainer.utils.distributed import rank_zero_only

if is_clearml_available():
    from clearml import Task
else:
    raise ImportError("ClearML is not installed. Please install it with `pip install clearml`")


class ClearMLLogger(TensorboardLogger):
    """ClearML Logger

    TODO:
        - Add hyperparameter handling
        - Use ClearML logger for plots
        - Handle continuing training
    """

    def __init__(
        self,
        output_uri: str,
        project_name: str,
        task_name: str,
        tags: str = None,
    ):
        self._context = None
        self.task_name = task_name
        self.tags = tags.split(",") if tags else []
        self.run = Task.init(project_name=project_name, task_name=task_name, tags=self.tags, output_uri=output_uri)

        if tags:
            for tag in tags.split(","):
                self.run.add_tag(tag)

        super().__init__("run", None)

    def add_config(self, config):
        self.add_text("run_config", f"{config.to_json()}", 0)
        self.run.connect_configuration(name="dictionary", configuration=config.to_dict())
        self.run.set_comment(config.run_description)
