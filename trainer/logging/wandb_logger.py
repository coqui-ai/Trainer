# pylint: disable=W0613

import traceback
from collections import defaultdict
from pathlib import Path
from typing import Union

from trainer.logging.base_dash_logger import BaseDashboardLogger
from trainer.trainer_utils import is_wandb_available
from trainer.utils.distributed import rank_zero_only

if is_wandb_available():
    import wandb  # pylint: disable=import-error


class WandbLogger(BaseDashboardLogger):
    def __init__(self, **kwargs):
        if not wandb:
            raise RuntimeError("install wandb using `pip install wandb` to use WandbLogger")

        self.run = None
        self.run = wandb.init(**kwargs) if not wandb.run else wandb.run
        self.model_name = self.run.config.model
        # dictionary of dictionaries - record stats per step
        self.log_dict = defaultdict(dict)

    def model_weights(self, model, step):
        layer_num = 1
        for name, param in model.named_parameters():
            if param.numel() == 1:
                self.add_scalars("weights", {"layer{}-{}/value".format(layer_num, name): param.max()}, step)
            else:
                self.add_scalars("weights", {"layer{}-{}/max".format(layer_num, name): param.max()}, step)
                self.add_scalars("weights", {"layer{}-{}/min".format(layer_num, name): param.min()}, step)
                self.add_scalars("weights", {"layer{}-{}/mean".format(layer_num, name): param.mean()}, step)
                self.add_scalars("weights", {"layer{}-{}/std".format(layer_num, name): param.std()}, step)
                self.log_dict[step]["weights/layer{}-{}/param".format(layer_num, name)] = wandb.Histogram(param)
                self.log_dict[step]["weights/layer{}-{}/grad".format(layer_num, name)] = wandb.Histogram(param.grad)
            layer_num += 1

    def add_scalars(self, scope_name, scalars, step):
        for key, value in scalars.items():
            self.log_dict[step]["{}/{}".format(scope_name, key)] = value

    def add_figures(self, scope_name, figures, step):
        for key, value in figures.items():
            self.log_dict[step]["{}/{}".format(scope_name, key)] = wandb.Image(value)

    def add_audios(self, scope_name, audios, step, sample_rate):
        for key, value in audios.items():
            if value.dtype == "float16":
                value = value.astype("float32")
            try:
                self.log_dict[step]["{}/{}".format(scope_name, key)] = wandb.Audio(value, sample_rate=sample_rate)
            except RuntimeError:
                traceback.print_exc()

    def add_text(self, title, text, step):
        pass

    def add_scalar(self, title: str, value: float, step: int) -> None:
        pass

    def add_figure(
        self,
        title: str,
        figure: Union["matplotlib.figure.Figure", "plotly.graph_objects.Figure"],
        step: int,
    ) -> None:
        pass

    def add_audio(self, title: str, audio: "np.ndarray", step: int, sample_rate: int) -> None:
        pass

    @rank_zero_only
    def add_config(self, config):
        pass

    def flush(self):
        if self.run:
            for step in sorted(self.log_dict.keys()):
                wandb.log(self.log_dict[step], step)
        self.log_dict.clear()

    def finish(self):
        if self.run:
            self.run.finish()

    def add_artifact(self, file_or_dir, name, artifact_type, aliases=None):
        if not self.run:
            return
        name = "_".join([self.run.id, name])
        artifact = wandb.Artifact(name, type=artifact_type)
        data_path = Path(file_or_dir)
        if data_path.is_dir():
            artifact.add_dir(str(data_path))
        elif data_path.is_file():
            artifact.add_file(str(data_path))

        self.run.log_artifact(artifact, aliases=aliases)
