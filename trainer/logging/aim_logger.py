import torch

from trainer.logging.base_dash_logger import BaseDashboardLogger
from trainer.trainer_utils import is_aim_available
from trainer.utils.distributed import rank_zero_only

if is_aim_available():
    from aim import Audio, Image, Repo, Text  # pylint: disable=import-error
    from aim.sdk.run import Run  # pylint: disable=import-error


# pylint: disable=too-many-public-methods
class AimLogger(BaseDashboardLogger):
    def __init__(
        self,
        repo: str,
        model_name: str,
        tags: str = None,
    ):
        self._context = None
        self.model_name = model_name
        self.run = Run(repo=repo, experiment=model_name)
        self.repo = Repo(repo)

        # query = f"runs.name == '{model_name}'"
        # runs = self.repo.query_runs(query=query)

        if tags:
            for tag in tags.split(","):
                self.run.add_tag(tag)

    # @staticmethod
    # def __fig_to_pil(image):
    #     """Convert Matplotlib figure to PIL image."""
    #     return PIL.Image.frombytes("RGB", image.canvas.get_width_height(), image.canvas.tostring_rgb())

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, context):
        self._context = context

    def model_weights(self, model, step):
        layer_num = 1
        for name, param in model.named_parameters():
            if param.numel() == 1:
                self.run.log_metric("layer{}-{}/value".format(layer_num, name), param.max(), step)
            else:
                self.run.log_metric("layer{}-{}/max".format(layer_num, name), param.max(), step)
                self.run.log_metric("layer{}-{}/min".format(layer_num, name), param.min(), step)
                self.run.log_metric("layer{}-{}/mean".format(layer_num, name), param.mean(), step)
                self.run.log_metric("layer{}-{}/std".format(layer_num, name), param.std(), step)
                # MlFlow does not support histograms
                # self.client.addå_histogram("layer{}-{}/param".format(layer_num, name), param, step)
                # self.client.add_histogram("layer{}-{}/grad".format(layer_num, name), param.grad, step)
            layer_num += 1

    def add_config(self, config):
        """TODO: Add config to AIM"""
        # self.run['hparams'] = config.to_dict()
        self.add_text("model-config", f"<pre>{config.to_json()}</pre>", 0)

    def add_scalar(self, title, value, step):
        self.run.track(value, name=title, step=step, context=self.context)

    def add_text(self, title, text, step):
        self.run.track(
            Text(text),  # Pass a string you want to track
            name=title,  # The name of distributions
            step=step,  # Step index (optional)
            context=self.context,
        )

    def add_figure(self, title, figure, step):
        self.run.track(
            Image(figure, title),  # Pass image data and/or caption
            name=title,  # The name of image set
            step=step,  # Step index (optional)
            context=self.context,
        )

    def add_artifact(self, file_or_dir, name, artifact_type, aliases=None):  # pylint: disable=W0613
        # AIM does not support artifacts
        ...

    def add_audio(self, title, audio, step, sample_rate):
        self.run.track(
            Audio(audio),  # Pass audio file or numpy array
            name=title,  # The name of distributions
            step=step,  # Step index (optional)
            context=self.context,
        )

    @rank_zero_only
    def add_scalars(self, scope_name, scalars, step):
        for key, value in scalars.items():
            if torch.is_tensor(value):
                value = value.item()
            self.run.track(value, name="{}-{}".format(scope_name, key), step=step, context=self.context)

    @rank_zero_only
    def add_figures(self, scope_name, figures, step):
        for key, value in figures.items():
            title = "{}/{}/{}.png".format(scope_name, key, step)
            self.run.track(
                Image(value, title),  # Pass image data and/or caption
                name=title,  # The name of image set
                step=step,  # Step index (optional)
                context=self.context,
            )

    @rank_zero_only
    def add_audios(self, scope_name, audios, step, sample_rate):
        for key, value in audios.items():
            title = "{}/{}/{}.wav".format(scope_name, key, step)
            self.run.track(
                Audio(value),  # Pass audio file or numpy array
                name=title,  # The name of distributions
                step=step,  # Step index (optional)
                context=self.context,
            )

    def train_step_stats(self, step, stats):
        self.context = {"subset": "train"}
        super().train_step_stats(step, stats)

    def train_epoch_stats(self, step, stats):
        self.context = {"subset": "train"}
        super().train_epoch_stats(step, stats)

    def train_figures(self, step, figures):
        self.context = {"subset": "train"}
        super().train_figures(step, figures)

    def train_audios(self, step, audios, sample_rate):
        self.context = {"subset": "train"}
        super().train_audios(step, audios, sample_rate)

    def eval_stats(self, step, stats):
        self.context = {"subset": "eval"}
        super().eval_stats(step, stats)

    def eval_figures(self, step, figures):
        self.context = {"subset": "eval"}
        super().eval_figures(step, figures)

    def eval_audios(self, step, audios, sample_rate):
        self.context = {"subset": "eval"}
        super().eval_audios(step, audios, sample_rate)

    def test_audios(self, step, audios, sample_rate):
        self.context = {"subset": "test"}
        super().test_audios(step, audios, sample_rate)

    def test_figures(self, step, figures):
        self.context = {"subset": "test"}
        super().test_figures(step, figures)

    def flush(self):
        pass

    @rank_zero_only
    def finish(self):
        super().close()
