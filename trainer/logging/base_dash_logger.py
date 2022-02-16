from abc import ABC, abstractmethod
from typing import Dict, Union


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
    def add_audios(self, scope_name: str, audios: Dict, step: int):
        pass

    @abstractmethod
    def flush():
        pass

    @abstractmethod
    def finish():
        pass

    def train_step_stats(self, step, stats):
        self.add_scalars("TrainIterStats", stats, step)

    def train_epoch_stats(self, step, stats):
        self.add_scalars("TrainEpochStats", stats, step)

    def train_figures(self, step, figures):
        self.add_figures("TrainFigures", figures, step)

    def train_audios(self, step, audios, sample_rate):
        self.add_audios("TrainAudios", audios, step, sample_rate)

    def eval_stats(self, step, stats):
        self.add_scalars("EvalStats", stats, step)

    def eval_figures(self, step, figures):
        self.add_figures("EvalFigures", figures, step)

    def eval_audios(self, step, audios, sample_rate):
        self.add_audios("EvalAudios", audios, step, sample_rate)

    def test_audios(self, step, audios, sample_rate):
        self.add_audios("TestAudios", audios, step, sample_rate)

    def test_figures(self, step, figures):
        self.add_figures("TestFigures", figures, step)
