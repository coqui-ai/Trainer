from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
from coqpit import Coqpit
from torch import nn

# pylint: skip-file


class TrainerModel(ABC, nn.Module):
    """Abstract 🐸TTS class. Every new 🐸TTS model must inherit this."""

    @abstractmethod
    def forward(self, input: torch.Tensor, *args, aux_input={}, **kwargs) -> Dict:
        """Forward ... for the model mainly used in training.

        You can be flexible here and use different number of arguments and argument names since it is intended to be
        used by `train_step()` without exposing it out of the model.

        Args:
            input (torch.Tensor): Input tensor.
            aux_input (Dict): Auxiliary model inputs like embeddings, durations or any other sorts of inputs.

        Returns:
            Dict: Model outputs. Main model output must be named as "model_outputs".
        """
        outputs_dict = {"model_outputs": None}
        ...
        return outputs_dict

    def format_batch(self, batch: Dict) -> Dict:
        """Format batch returned by the data loader before sending it to the model.

        If not implemented, model uses the batch as is.
        Can be used for data augmentation, feature ectraction, etc.
        """
        return batch

    def format_batch_on_device(self, batch: Dict) -> Dict:
        """Format batch on device before sending it to the model.

        If not implemented, model uses the batch as is.
        Can be used for data augmentation, feature ectraction, etc.
        """
        return batch

    @abstractmethod
    def train_step(self, batch: Dict, criterion: nn.Module) -> Tuple[Dict, Dict]:
        """Perform a single training step. Run the model forward ... and compute losses.

        Args:
            batch (Dict): Input tensors.
            criterion (nn.Module): Loss layer designed for the model.

        Returns:
            Tuple[Dict, Dict]: Model ouputs and computed losses.
        """
        outputs_dict = {}
        loss_dict = {}  # this returns from the criterion
        ...
        return outputs_dict, loss_dict

    def train_log(self, batch: Dict, outputs: Dict, logger: "Logger", assets: Dict, steps: int) -> None:
        """Create visualizations and waveform examples for training.

        For example, here you can plot spectrograms and generate sample sample waveforms from these spectrograms to
        be projected onto Tensorboard.

        Args:
            ap (AudioProcessor): audio processor used at training.
            batch (Dict): Model inputs used at the previous training step.
            outputs (Dict): Model outputs generated at the previoud training step.

        Returns:
            Tuple[Dict, np.ndarray]: training plots and output waveform.
        """
        ...

    @abstractmethod
    def eval_step(self, batch: Dict, criterion: nn.Module) -> Tuple[Dict, Dict]:
        """Perform a single evaluation step. Run the model forward ... and compute losses. In most cases, you can
        call `train_step()` with no changes.

        Args:
            batch (Dict): Input tensors.
            criterion (nn.Module): Loss layer designed for the model.

        Returns:
            Tuple[Dict, Dict]: Model ouputs and computed losses.
        """
        outputs_dict = {}
        loss_dict = {}  # this returns from the criterion
        ...
        return outputs_dict, loss_dict

    def eval_log(self, batch: Dict, outputs: Dict, logger: "Logger", assets: Dict, steps: int) -> None:
        """The same as `train_log()`"""
        ...

    @abstractmethod
    def get_data_loader(
        self, config: Coqpit, assets: Dict, is_eval: True, data_items: List, verbose: bool, num_gpus: int
    ):
        ...

    def init_for_training(self) -> None:
        """Initialize model for training."""
        ...

    # def get_optimizer(self) -> Union["Optimizer", List["Optimizer"]]:
    #     """Setup an return optimizer or optimizers."""
    #     ...

    # def get_lr(self) -> Union[float, List[float]]:
    #     """Return learning rate(s).

    #     Returns:
    #         Union[float, List[float]]: Model's initial learning rates.
    #     """
    #     ...

    # def get_scheduler(self, optimizer: torch.optim.Optimizer):
    #     ...

    # def get_criterion(self):
    #     ...
