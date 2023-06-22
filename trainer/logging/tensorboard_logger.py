import traceback

from torch.utils.tensorboard import SummaryWriter

from trainer.logging.base_dash_logger import BaseDashboardLogger


class TensorboardLogger(BaseDashboardLogger):
    def __init__(self, log_dir, model_name):
        self.model_name = model_name
        self.writer = SummaryWriter(log_dir)

    def model_weights(self, model, step):
        layer_num = 1
        for name, param in model.named_parameters():
            if param.numel() == 1:
                self.writer.add_scalar("layer{}-{}/value".format(layer_num, name), param.max(), step)
            else:
                self.writer.add_scalar("layer{}-{}/max".format(layer_num, name), param.max(), step)
                self.writer.add_scalar("layer{}-{}/min".format(layer_num, name), param.min(), step)
                self.writer.add_scalar("layer{}-{}/mean".format(layer_num, name), param.mean(), step)
                self.writer.add_scalar("layer{}-{}/std".format(layer_num, name), param.std(), step)
                self.writer.add_histogram("layer{}-{}/param".format(layer_num, name), param, step)
                self.writer.add_histogram("layer{}-{}/grad".format(layer_num, name), param.grad, step)
            layer_num += 1

    def add_config(self, config):
        self.add_text("model-config", f"<pre>{config.to_json()}</pre>", 0)

    def add_scalar(self, title: str, value: float, step: int) -> None:
        self.writer.add_scalar(title, value, step)

    def add_audio(self, title, audio, step, sample_rate):
        self.writer.add_audio(title, audio, step, sample_rate=sample_rate)

    def add_text(self, title, text, step):
        self.writer.add_text(title, text, step)

    def add_figure(self, title, figure, step):
        self.writer.add_figure(title, figure, step)

    def add_artifact(self, file_or_dir, name, artifact_type, aliases=None):  # pylint: disable=W0613
        yield

    def add_scalars(self, scope_name, scalars, step):
        for key, value in scalars.items():
            self.add_scalar("{}/{}".format(scope_name, key), value, step)

    def add_figures(self, scope_name, figures, step):
        for key, value in figures.items():
            self.writer.add_figure("{}/{}".format(scope_name, key), value, step)

    def add_audios(self, scope_name, audios, step, sample_rate):
        for key, value in audios.items():
            if value.dtype == "float16":
                value = value.astype("float32")
            try:
                self.add_audio(
                    "{}/{}".format(scope_name, key),
                    value,
                    step,
                    sample_rate=sample_rate,
                )
            except RuntimeError:
                traceback.print_exc()

    def flush(self):
        self.writer.flush()

    def finish(self):
        self.writer.close()
