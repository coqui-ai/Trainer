import os
import shutil
import tempfile
import traceback

import soundfile as sf
import torch

from trainer.logging.base_dash_logger import BaseDashboardLogger
from trainer.trainer_utils import is_mlflow_available
from trainer.utils.distributed import rank_zero_only

if is_mlflow_available():
    from mlflow.tracking import MlflowClient
    from mlflow.tracking.context.registry import resolve_tags
    from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

# pylint: skip-file


class MLFlowLogger(BaseDashboardLogger):
    def __init__(
        self,
        log_uri: str,
        model_name: str,
        tags: str = None,
    ):
        self.model_name = model_name
        self.client = MlflowClient(tracking_uri=os.path.join(log_uri))

        experiment = self.client.get_experiment_by_name(model_name)
        if experiment is None:
            self.experiment_id = self.client.create_experiment(name=model_name)
        else:
            self.experiment_id = experiment.experiment_id

        if tags is not None:
            self.client.set_experiment_tag(self.experiment_id, MLFLOW_RUN_NAME, tags)
        run = self.client.create_run(experiment_id=self.experiment_id, tags=resolve_tags(tags))
        self.run_id = run.info.run_id

    def model_weights(self, model, step):
        layer_num = 1
        for name, param in model.named_parameters():
            if param.numel() == 1:
                self.client.log_metric("layer{}-{}/value".format(layer_num, name), param.max(), step)
            else:
                self.client.log_metric("layer{}-{}/max".format(layer_num, name), param.max(), step)
                self.client.log_metric("layer{}-{}/min".format(layer_num, name), param.min(), step)
                self.client.log_metric("layer{}-{}/mean".format(layer_num, name), param.mean(), step)
                self.client.log_metric("layer{}-{}/std".format(layer_num, name), param.std(), step)
                # MlFlow does not support histograms
                # self.client.add_histogram("layer{}-{}/param".format(layer_num, name), param, step)
                # self.client.add_histogram("layer{}-{}/grad".format(layer_num, name), param.grad, step)
            layer_num += 1

    def add_config(self, config):
        self.add_text("model-config", f"<pre>{config.to_json()}</pre>", 0)

    def add_scalar(self, title, value, step):
        self.client.log_metric(self.run_id, title, value, step)

    def add_text(self, title, text, step):
        self.client.log_text(self.run_id, text, "{}/{}.txt".format(title, step))

    def add_figure(self, title, figure, step):
        self.client.log_figure(figure, "{}/{}.png".format(title, step))

    def add_artifact(self, file_or_dir, name, artifact_type, aliases=None):  # pylint: disable=W0613, R0201
        self.client.log_artifacts(self.run_id, file_or_dir)

    def add_audio(self, title, audio, step, sample_rate):
        self.client.log_audio(self.run_id, audio, "{}/{}.wav".format(title, step), sample_rate)

    @rank_zero_only
    def add_scalars(self, scope_name, stats, step):
        for key, value in stats.items():
            if torch.is_tensor(value):
                value = value.item()
            self.client.log_metric(self.run_id, "{}-{}".format(scope_name, key), value, step)

    @rank_zero_only
    def add_figures(self, scope_name, figures, step):
        for key, value in figures.items():
            self.client.log_figure(self.run_id, value, "{}/{}/{}.png".format(scope_name, key, step))

    @rank_zero_only
    def add_audios(self, scope_name, audios, step, sample_rate):
        for key, value in audios.items():
            if value.dtype == "float16":
                value = value.astype("float32")
            try:
                tmp_audio_path = tempfile.NamedTemporaryFile(suffix=".wav")
                sf.write(tmp_audio_path, value, sample_rate)
                self.client.log_artifact(
                    self.run_id,
                    tmp_audio_path,
                    "{}/{}/{}.wav".format(scope_name, key, step),
                )
                shutil.rmtree(tmp_audio_path)
            except RuntimeError:
                traceback.print_exc()

    def train_step_stats(self, step, stats):
        self.client.set_tag(self.run_id, "Mode", "training")
        super().train_step_stats(step, stats)

    def train_epoch_stats(self, step, stats):
        self.client.set_tag(self.run_id, "Mode", "training")
        super().train_epoch_stats(step, stats)

    def train_figures(self, step, figures):
        self.client.set_tag(self.run_id, "Mode", "training")
        super().train_figures(step, figures)

    def train_audios(self, step, audios, sample_rate):
        self.client.set_tag(self.run_id, "Mode", "training")
        super().train_audios(step, audios, sample_rate)

    def eval_stats(self, step, stats):
        self.client.set_tag(self.run_id, "Mode", "evaluation")
        super().eval_stats(step, stats)

    def eval_figures(self, step, figures):
        self.client.set_tag(self.run_id, "Mode", "evaluation")
        super().eval_figures(step, figures)

    def eval_audios(self, step, audios, sample_rate):
        self.client.set_tag(self.run_id, "Mode", "evaluation")
        super().eval_audios(step, audios, sample_rate)

    def test_audios(self, step, audios, sample_rate):
        self.client.set_tag(self.run_id, "Mode", "test")
        super().test_audios(step, audios, sample_rate)

    def test_figures(self, step, figures):
        self.client.set_tag(self.run_id, "Mode", "test")
        super().test_figures(step, figures)

    def flush(self):
        pass

    @rank_zero_only
    def finish(self):
        super().finalize(status)
        status = "FINISHED" if status == "success" else status
        if self.client.get_run(self.run_id):
            self.client.set_terminated(self.run_id, status)
