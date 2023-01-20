from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import torch
from coqpit import Coqpit
from torch import nn

from trainer.trainer_utils import is_apex_available

if is_apex_available():
    from apex import amp


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
        Can be used for data augmentation, feature ectraction, etc.`
        """
        return batch

    def train_step(self, *args: Any, **kwargs: Any) -> Tuple[Dict, Dict]:
        """Perform a single training step. Run the model forward ... and compute losses.

        Args:
            batch (Dict): Input tensors.
            criterion (nn.Module): Loss layer designed for the model.
            optimizer_idx (int): Index of optimizer to use. 0 for the generator and 1 for the discriminator networks.

        Returns:
            Tuple[Dict, Dict]: Model ouputs and computed losses.
        """
        ...
        raise NotImplementedError(" [!] `train_step()` is not implemented.")

    def train_log(self, *args: Any, **kwargs: Any) -> None:
        """Create visualizations and waveform examples for training.

        For example, here you can plot spectrograms and generate sample sample waveforms from these spectrograms to
        be projected onto Tensorboard.

        Args:
            batch (Dict): Model inputs used at the previous training step.
            outputs (Dict): Model outputs generated at the previoud training step.
            logger (Logger): Logger instance to log training plots.
            assets (Dict): Assets to be used for logging from the trainer's closure.
            steps (int): Number of training steps taken so far.

        Returns:
            Tuple[Dict, np.ndarray]: training plots and output waveform.
        """
        ...
        raise NotImplementedError(" [!] `train_log()` is not implemented.")

    @torch.no_grad()
    def eval_step(self, *args: Any, **kwargs: Any):
        """Perform a single evaluation step. Run the model forward ... and compute losses. In most cases, you can
        call `train_step()` with no changes.

        Args:
            batch (Dict): Input tensors.
            criterion (nn.Module): Loss layer designed for the model.
            optimizer_idx (int): Index of optimizer to use. 0 for the generator and 1 for the discriminator networks.

        Returns:
            Tuple[Dict, Dict]: Model ouputs and computed losses.
        """
        raise NotImplementedError(" [!] `eval_step()` is not implemented.")

    def eval_log(self, *args: Any, **kwargs: Any) -> None:
        """The same as `train_log()`"""
        ...
        raise NotImplementedError(" [!] `eval_log()` is not implemented.")

    @abstractmethod
    def get_data_loader(*args: Any, **kwargs: Any) -> torch.utils.data.DataLoader:
        """Get data loader for the model.

        Args:
            config (Coqpit): Configuration object.
            assets (Dict): Additional assets to be used for data loading.
            is_eval (bool): If True, returns evaluation data loader.
            samples (Union[List[Dict], List[List]]): List of samples to be used for data loading.
            verbose (bool): If True, prints data loading information.
            num_gpus (int): Number of GPUs used for training.
            rank (int): Rank of the current GPU.

        Returns:
            torch.utils.data.DataLoader: Data loader for the model.
        """

        ...
        raise NotImplementedError(" [!] `get_data_loader()` is not implemented.")

    def init_for_training(self) -> None:
        """Initialize model for training."""
        ...

    def optimize(self, *args: Any, **kwargs: Any) -> Tuple[Dict, Dict, float]:
        """Model specific optimization step that must perform the following steps:
            1. Forward pass
            2. Compute loss
            3. Backward pass
            4. Update weights

        Use `self.scaled_backward()` instead of `loss.backward()` to be able to use Mixed Precision Training.

        Args:
            batch (Dict): Input tensors.
            trainer (Trainer): Trainer instance to be able to access the training closure.

        Returns:
            Tuple[Dict, Dict, float]: Model outputs, loss dictionary and grad_norm value.
        """
        ...
        raise NotImplementedError(" [!] `optimize()` is not implemented.")

    def scaled_backward(
        self, loss: torch.Tensor, trainer: "Trainer", optimizer: "Optimizer", *args: Any, **kwargs: Any
    ) -> Tuple[float, bool]:
        """Backward pass with gradient scaling for custom `optimize` calls.

        Args:
            loss (torch.Tensor): Loss to be backpropagated.
            trainer (Trainer): Trainer instance to be able to access the training closure.
            optimizer (Optimizer): Optimizer for APEX AMP based scaled `backward` calls.
        """
        if trainer.use_amp_scaler:
            if trainer.use_apex:
                # https://nvidia.github.io/apex/advanced.html?highlight=accumulate#backward-passes-with-multiple-optimizers
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # model optimizer step in mixed precision mode
                trainer.scaler.scale(loss).backward()
        else:
            # main model optimizer step
            loss.backward()

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
