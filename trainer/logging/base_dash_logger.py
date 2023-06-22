from abc import ABC, abstractmethod
from typing import Dict, Union

from trainer.io import save_fsspec
from trainer.utils.distributed import rank_zero_only


# pylint: disable=too-many-public-methods
class BaseDashboardLogger(ABC):
    @abstractmethod
    def add_scalar(self, title: str, value: float, step: int) -> None:
        pass

    @abstractmethod
    def add_figure(
        self,
        title: str,
        figure: Union["matplotlib.figure.Figure", "plotly.graph_objects.Figure"],
        step: int,
    ) -> None:
        pass

    @abstractmethod
    def add_config(self, config):
        pass

    @abstractmethod
    def add_audio(self, title: str, audio: "np.ndarray", step: int, sample_rate: int) -> None:
        pass

    @abstractmethod
    def add_text(self, title: str, text: str, step: int) -> None:
        pass

    @abstractmethod
    def add_artifact(self, file_or_dir: str, name: str, artifact_type: str, aliases=None):
        pass

    @abstractmethod
    def add_scalars(self, scope_name: str, scalars: Dict, step: int):
        pass

    @abstractmethod
    def add_figures(self, scope_name: str, figures: Dict, step: int):
        pass

    @abstractmethod
    def add_audios(self, scope_name: str, audios: Dict, step: int, sample_rate: int):
        pass

    @abstractmethod
    def flush(self):
        pass

    @abstractmethod
    def finish(self):
        pass

    @staticmethod
    @rank_zero_only
    def save_model(state: Dict, path: str):
        save_fsspec(state, path)

    def train_step_stats(self, step, stats):
        self.add_scalars(scope_name="TrainIterStats", scalars=stats, step=step)

    def train_epoch_stats(self, step, stats):
        self.add_scalars(scope_name="TrainEpochStats", scalars=stats, step=step)

    def train_figures(self, step, figures):
        self.add_figures(scope_name="TrainFigures", figures=figures, step=step)

    def train_audios(self, step, audios, sample_rate):
        self.add_audios(scope_name="TrainAudios", audios=audios, step=step, sample_rate=sample_rate)

    def eval_stats(self, step, stats):
        self.add_scalars(scope_name="EvalStats", scalars=stats, step=step)

    def eval_figures(self, step, figures):
        self.add_figures(scope_name="EvalFigures", figures=figures, step=step)

    def eval_audios(self, step, audios, sample_rate):
        self.add_audios(scope_name="EvalAudios", audios=audios, step=step, sample_rate=sample_rate)

    def test_audios(self, step, audios, sample_rate):
        self.add_audios(scope_name="TestAudios", audios=audios, step=step, sample_rate=sample_rate)

    def test_figures(self, step, figures):
        self.add_figures(scope_name="TestFigures", figures=figures, step=step)
