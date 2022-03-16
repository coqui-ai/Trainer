# pylint: disable=W0613

import traceback
from pathlib import Path

from trainer.logging.base_dash_logger import BaseDashboardLogger
from trainer.trainer_utils import is_wandb_available
from trainer.utils.distributed import rank_zero_only

if is_wandb_available():
    import wandb


class WandbLogger(BaseDashboardLogger):
    def __init__(self, **kwargs):

        if not wandb:
            raise Exception("install wandb using `pip install wandb` to use WandbLogger")

        self.run = None
        self.run = wandb.init(**kwargs) if not wandb.run else wandb.run
        self.model_name = self.run.config.model
        self.log_dict = {}

    def model_weights(self, model):
        layer_num = 1
        for name, param in model.named_parameters():
            if param.numel() == 1:
                self.add_scalars("weights", {"layer{}-{}/value".format(layer_num, name): param.max()})
            else:
                self.add_scalars("weights", {"layer{}-{}/max".format(layer_num, name): param.max()})
                self.add_scalars("weights", {"layer{}-{}/min".format(layer_num, name): param.min()})
                self.add_scalars("weights", {"layer{}-{}/mean".format(layer_num, name): param.mean()})
                self.add_scalars("weights", {"layer{}-{}/std".format(layer_num, name): param.std()})
                self.log_dict["weights/layer{}-{}/param".format(layer_num, name)] = wandb.Histogram(param)
                self.log_dict["weights/layer{}-{}/grad".format(layer_num, name)] = wandb.Histogram(param.grad)
            layer_num += 1

    def add_scalars(self, scope_name, stats):
        for key, value in stats.items():
            self.log_dict["{}/{}".format(scope_name, key)] = value

    def add_figures(self, scope_name, figures):
        for key, value in figures.items():
            self.log_dict["{}/{}".format(scope_name, key)] = wandb.Image(value)

    def add_audios(self, scope_name, audios, sample_rate):
        for key, value in audios.items():
            if value.dtype == "float16":
                value = value.astype("float32")
            try:
                self.log_dict["{}/{}".format(scope_name, key)] = wandb.Audio(value, sample_rate=sample_rate)
            except RuntimeError:
                traceback.print_exc()

    def log(self, log_dict, prefix="", flush=False):
        for key, value in log_dict.items():
            self.log_dict[prefix + key] = value
        if flush:  # for cases where you don't want to accumulate data
            self.flush()

    def add_text(self, title, text, step):
        pass

    @rank_zero_only
    def add_config(self, config):
        pass

    def flush(self):
        if self.run:
            wandb.log(self.log_dict)
        self.log_dict = {}

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

        self.run.add_artifact(artifact, aliases=aliases)
