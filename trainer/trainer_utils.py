import importlib
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch

from trainer.logger import logger
from trainer.torch import NoamLR, StepwiseGradualLR
from trainer.utils.distributed import rank_zero_logger_info


def is_apex_available():
    return importlib.util.find_spec("apex") is not None


def is_mlflow_available():
    return importlib.util.find_spec("mlflow") is not None


def is_aim_available():
    return importlib.util.find_spec("aim") is not None


def is_wandb_available():
    return importlib.util.find_spec("wandb") is not None


def is_clearml_available():
    return importlib.util.find_spec("clearml") is not None


def print_training_env(args, config):
    """Print training environment."""
    rank_zero_logger_info(" > Training Environment:", logger)

    if args.use_accelerate:
        rank_zero_logger_info(" | > Backend: Accelerate", logger)
    else:
        rank_zero_logger_info(" | > Backend: Torch", logger)

    if config.mixed_precision:
        rank_zero_logger_info(" | > Mixed precision: True", logger)
        rank_zero_logger_info(f" | > Precision: {config.precision}", logger)
    else:
        rank_zero_logger_info(" | > Mixed precision: False", logger)
        rank_zero_logger_info(" | > Precision: float32", logger)

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        rank_zero_logger_info(f" | > Current device: {torch.cuda.current_device()}", logger)
        rank_zero_logger_info(f" | > Num. of GPUs: {torch.cuda.device_count()}", logger)

    rank_zero_logger_info(f" | > Num. of CPUs: {os.cpu_count()}", logger)
    rank_zero_logger_info(f" | > Num. of Torch Threads: {torch.get_num_threads()}", logger)
    rank_zero_logger_info(f" | > Torch seed: {torch.initial_seed()}", logger)
    rank_zero_logger_info(f" | > Torch CUDNN: {torch.backends.cudnn.enabled}", logger)
    rank_zero_logger_info(f" | > Torch CUDNN deterministic: {torch.backends.cudnn.deterministic}", logger)
    rank_zero_logger_info(f" | > Torch CUDNN benchmark: {torch.backends.cudnn.benchmark}", logger)
    rank_zero_logger_info(f" | > Torch TF32 MatMul: {torch.backends.cuda.matmul.allow_tf32}", logger)


def setup_torch_training_env(
    args: "TrainerArgs",
    cudnn_enable: bool,
    cudnn_benchmark: bool,
    cudnn_deterministic: bool,
    use_ddp: bool = False,
    training_seed=54321,
    allow_tf32: bool = False,
    gpu=None,
) -> Tuple[bool, int]:
    """Setup PyTorch environment for training.

    Args:
        cudnn_enable (bool): Enable/disable CUDNN.
        cudnn_benchmark (bool): Enable/disable CUDNN benchmarking. Better to set to False if input sequence length is
            variable between batches.
        cudnn_deterministic (bool): Enable/disable CUDNN deterministic mode.
        use_ddp (bool): DDP flag. True if DDP is enabled, False otherwise.
        allow_tf32 (bool): Enable/disable TF32. TF32 is only available on Ampere GPUs.
        torch_seed (int): Seed for torch random number generator.

    Returns:
        Tuple[bool, int]: is cuda on or off and number of GPUs in the environment.
    """
    # clear cache before training
    torch.cuda.empty_cache()

    # set_nvidia_flags
    # set the correct cuda visible devices (using pci order)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if "CUDA_VISIBLE_DEVICES" not in os.environ and gpu is not None:
        torch.cuda.set_device(int(gpu))
        num_gpus = 1
    else:
        num_gpus = torch.cuda.device_count()

    if num_gpus > 1 and (not use_ddp and not args.use_accelerate):
        raise RuntimeError(
            f" [!] {num_gpus} active GPUs. Define the target GPU by `CUDA_VISIBLE_DEVICES`. For multi-gpu training use `TTS/bin/distribute.py`."
        )

    random.seed(training_seed)
    os.environ["PYTHONHASHSEED"] = str(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed(training_seed)

    # set torch backend flags.
    # set them true if they are already set true
    torch.backends.cudnn.deterministic = cudnn_deterministic or torch.backends.cudnn.deterministic
    torch.backends.cudnn.enabled = cudnn_enable or torch.backends.cudnn.enabled
    torch.backends.cudnn.benchmark = cudnn_benchmark or torch.backends.cudnn.benchmark
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32 or torch.backends.cuda.matmul.allow_tf32

    use_cuda = torch.cuda.is_available()
    return use_cuda, num_gpus


def get_scheduler(
    lr_scheduler: str, lr_scheduler_params: Dict, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:  # pylint: disable=protected-access
    """Find, initialize and return a Torch scheduler.

    Args:
        lr_scheduler (str): Scheduler name.
        lr_scheduler_params (Dict): Scheduler parameters.
        optimizer (torch.optim.Optimizer): Optimizer to pass to the scheduler.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Functional scheduler.
    """
    if lr_scheduler is None:
        return None
    if lr_scheduler.lower() == "noamlr":
        scheduler = NoamLR
    elif lr_scheduler.lower() == "stepwisegraduallr":
        scheduler = StepwiseGradualLR
    else:
        scheduler = getattr(torch.optim.lr_scheduler, lr_scheduler)
    return scheduler(optimizer, **lr_scheduler_params)


def get_optimizer(
    optimizer_name: str,
    optimizer_params: dict,
    lr: float,
    model: torch.nn.Module = None,
    parameters: List = None,
) -> torch.optim.Optimizer:
    """Find, initialize and return a Torch optimizer.

    Args:
        optimizer_name (str): Optimizer name.
        optimizer_params (dict): Optimizer parameters.
        lr (float): Initial learning rate.
        model (torch.nn.Module): Model to pass to the optimizer.

    Returns:
        torch.optim.Optimizer: Functional optimizer.
    """
    if optimizer_name.lower() == "radam":
        module = importlib.import_module("TTS.utils.radam")
        optimizer = getattr(module, "RAdam")
    else:
        optimizer = getattr(torch.optim, optimizer_name)
    if model is not None:
        parameters = model.parameters()
    return optimizer(parameters, lr=lr, **optimizer_params)
