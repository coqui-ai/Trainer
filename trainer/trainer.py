# -*- coding: utf-8 -*-
import gc
import importlib
import logging
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from inspect import signature
from typing import Callable, Dict, List, Tuple, Union

import torch
import torch.distributed as dist
from coqpit import Coqpit
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP_th
from torch.utils.data import DataLoader

from trainer.callbacks import TrainerCallback
from trainer.generic_utils import (
    KeepAverage,
    count_parameters,
    get_experiment_folder_path,
    get_git_branch,
    remove_experiment_folder,
    set_partial_state_dict,
    to_cuda,
)
from trainer.io import (
    copy_model_files,
    get_last_checkpoint,
    load_fsspec,
    save_best_model,
    save_checkpoint,
)
from trainer.logging import ConsoleLogger, DummyLogger, logger_factory
from trainer.trainer_utils import (
    get_optimizer,
    get_scheduler,
    is_apex_available,
    setup_torch_training_env,
)
from trainer.utils.cuda_memory import cuda_meminfo, should_reduce_batch_size
from trainer.utils.distributed import init_distributed

logger = logging.getLogger("trainer")

if is_apex_available():
    from apex import amp


@dataclass
class TrainerConfig(Coqpit):
    """Config fields tweaking the Trainer for a model.
    A ````ModelConfig```, by inheriting ```TrainerConfig``` must be defined for using 👟.
    Inherit this by a new model config and override the fields as needed.
    All the fields can be overridden from comman-line as ```--coqpit.arg_name=value```.

    Example::

        Run the training code by overriding the ```lr``` and ```plot_step``` fields.

        >>> python train.py --coqpit.plot_step=22 --coqpit.lr=0.001

        Defining a model using ```TrainerConfig```.

        >>> from trainer import TrainerConfig
        >>> class MyModelConfig(TrainerConfig):
        ...     optimizer: str = "Adam"
        ...     lr: float = 0.001
        ...     epochs: int = 1
        ...     ...
        >>> class MyModel(nn.module):
        ...    def __init__(self, config):
        ...        ...
        >>> model = MyModel(MyModelConfig())

    """

    # Fields for the run
    output_path: str = field(default="output")
    logger_uri: str = field(
        default=None,
        metadata={
            "help": "URI to save training artifacts by the logger. If not set, logs will be saved in the output_path. Defaults to None"
        },
    )
    run_name: str = field(default="run", metadata={"help": "Name of the run. Defaults to 'run'"})
    project_name: str = field(default=None, metadata={"help": "Name of the project. Defaults to None"})
    run_description: str = field(
        default="🐸Coqui trainer run.",
        metadata={"help": "Notes and description about the run. Defaults to '🐸Coqui trainer run.'"},
    )
    # Fields for logging
    print_step: int = field(
        default=25, metadata={"help": "Print training stats on the terminal every print_step steps. Defaults to 25"}
    )
    plot_step: int = field(
        default=100, metadata={"help": "Plot training stats on the logger every plot_step steps. Defaults to 100"}
    )
    model_param_stats: bool = field(
        default=False, metadata={"help": "Log model parameters stats on the logger dashboard. Defaults to False"}
    )
    wandb_entity: str = field(default=None, metadata={"help": "Wandb entity to log the run. Defaults to None"})
    dashboard_logger: str = field(
        default="tensorboard", metadata={"help": "Logger to use for the tracking dashboard. Defaults to 'tensorboard'"}
    )
    # Fields for checkpointing
    log_model_step: int = field(
        default=None,
        metadata={
            "help": "Save checkpoint to the logger every log_model_step steps. If not defined `save_step == log_model_step`."
        },
    )
    save_step: int = field(
        default=10000, metadata={"help": "Save local checkpoint every save_step steps. Defaults to 10000"}
    )
    save_n_checkpoints: int = field(default=5, metadata={"help": "Keep n local checkpoints. Defaults to 5"})
    save_checkpoints: bool = field(default=True, metadata={"help": "Save checkpoints locally. Defaults to True"})
    save_all_best: bool = field(
        default=False, metadata={"help": "Save all best checkpoints and keep the older ones. Defaults to False"}
    )
    save_best_after: int = field(
        default=10000, metadata={"help": "Wait N steps to save best checkpoints. Defaults to 10000"}
    )
    target_loss: str = field(
        default=None, metadata={"help": "Target loss name to select the best model. Defaults to None"}
    )
    # Fields for eval and test run
    print_eval: bool = field(default=False, metadata={"help": "Print eval steps on the terminal. Defaults to False"})
    test_delay_epochs: int = field(default=0, metadata={"help": "Wait N epochs before running the test. Defaults to 0"})
    run_eval: bool = field(
        default=True, metadata={"help": "Run evalulation epoch after training epoch. Defaults to True"}
    )
    run_eval_steps: int = field(
        default=None,
        metadata={
            "help": "Run evalulation epoch after N steps. If None, waits until training epoch is completed. Defaults to None"
        },
    )
    # Fields for distributed training
    distributed_backend: str = field(
        default="nccl", metadata={"help": "Distributed backend to use. Defaults to 'nccl'"}
    )
    distributed_url: str = field(
        default="tcp://localhost:54321",
        metadata={"help": "Distributed url to use. Defaults to 'tcp://localhost:54321'"},
    )
    # Fields for training specs
    mixed_precision: bool = field(default=False, metadata={"help": "Use mixed precision training. Defaults to False"})
    epochs: int = field(default=1000, metadata={"help": "Number of epochs to train. Defaults to 1000"})
    batch_size: int = field(default=32, metadata={"help": "Batch size to use. Defaults to 32"})
    eval_batch_size: int = field(default=16, metadata={"help": "Batch size to use for eval. Defaults to 16"})
    grad_clip: float = field(
        default=0.0, metadata={"help": "Gradient clipping value. Disabled if <= 0. Defaults to 0.0"}
    )
    scheduler_after_epoch: bool = field(
        default=True,
        metadata={"help": "Step the scheduler after each epoch else step after each iteration. Defaults to True"},
    )
    # Fields for optimzation
    lr: Union[float, List[float]] = field(
        default=0.001, metadata={"help": "Learning rate for each optimizer. Defaults to 0.001"}
    )
    optimizer: Union[str, List[str]] = field(default=None, metadata={"help": "Optimizer(s) to use. Defaults to None"})
    optimizer_params: Union[Dict, List[Dict]] = field(
        default_factory=dict, metadata={"help": "Optimizer(s) arguments. Defaults to {}"}
    )
    lr_scheduler: Union[str, List[str]] = field(
        default=None, metadata={"help": "Learning rate scheduler(s) to use. Defaults to None"}
    )
    lr_scheduler_params: Dict = field(
        default_factory=dict, metadata={"help": "Learning rate scheduler(s) arguments. Defaults to {}"}
    )
    use_grad_scaler: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable gradient scaler explicitly. It is enabled by default with AMP training. Defaults to False"
        },
    )
    cudnn_enable: bool = field(default=True, metadata={"help": "Enable/disable cudnn explicitly. Defaults to True"})
    cudnn_deterministic: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable deterministic cudnn operations. Set this True for reproducibility but it slows down training significantly.  Defaults to False."
        },
    )
    cudnn_benchmark: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable cudnn benchmark explicitly. Set this False if your input size change constantly. Defaults to False"
        },
    )
    training_seed: int = field(
        default=54321,
        metadata={"help": "Global seed for torch, random and numpy random number generator. Defaults to 54321"},
    )


@dataclass
class TrainerArgs(Coqpit):
    """Trainer arguments that can be accessed from the command line.

    Examples::
        >>> python train.py --restore_path /path/to/checkpoint.pth
    """

    continue_path: str = field(
        default="",
        metadata={
            "help": "Path to a training folder to continue training. Restore the model from the last checkpoint and continue training under the same folder."
        },
    )
    restore_path: str = field(
        default="",
        metadata={
            "help": "Path to a model checkpoit. Restore the model with the given checkpoint and start a new training."
        },
    )
    best_path: str = field(
        default="",
        metadata={
            "help": "Best model file to be used for extracting the best loss. If not specified, the latest best model in continue path is used"
        },
    )
    use_ddp: bool = field(
        default=False,
        metadata={"help": "Use DDP in distributed training. It is to set in `distribute.py`. Do not set manually."},
    )
    grad_accum_steps: int = field(
        default=1,
        metadata={
            "help": "Number of gradient accumulation steps. It is used to accumulate gradients over multiple batches."
        },
    )
    overfit_batch: bool = field(default=False, metadata={"help": "Overfit a single batch for debugging."})
    skip_train_epoch: bool = field(
        default=False,
        metadata={"help": "Skip training and only run evaluation and test."},
    )
    small_run: int = field(
        default=None,
        metadata={
            "help": "Only use a subset of the samples for debugging. Set the number of samples to use. Defaults to None. "
        },
    )
    gpu: int = field(
        default=None, metadata={"help": "GPU ID to use if ```CUDA_VISIBLE_DEVICES``` is not set. Defaults to None."}
    )
    # only for DDP
    rank: int = field(default=0, metadata={"help": "Process rank in a distributed training. Don't set manually."})
    group_id: str = field(
        default="", metadata={"help": "Process group id in a distributed training. Don't set manually."}
    )


class Trainer:
    def __init__(  # pylint: disable=dangerous-default-value
        self,
        args: TrainerArgs,
        config: Coqpit,
        output_path: str,
        c_logger: ConsoleLogger = None,
        dashboard_logger: "Logger" = None,
        model: nn.Module = None,
        get_model: Callable = None,
        get_data_samples: Callable = None,
        train_samples: List = None,
        eval_samples: List = None,
        test_samples: List = None,
        training_assets: Dict = {},
        parse_command_line_args: bool = True,
        callbacks: Dict[str, Callable] = {},
        gpu: int = None,
    ) -> None:
        """Simple yet powerful 🐸💬 TTS trainer for PyTorch. It can train all the available `tts` and `vocoder` models
        or easily be customized.

        Notes:

            Supports Automatic Mixed Precision training. If `Apex` is availabe, it automatically picks that, else
            it uses PyTorch's native `amp` module. `Apex` may provide more stable training in some cases.

        Args:

            args (Union[Coqpit, Namespace]): Training arguments parsed either from console by `argparse` or `TrainerArgs`
                config object.

            config (Coqpit): Model config object. It includes all the values necessary for initializing, training, evaluating
                and testing the model.

            output_path (str): Path to the output training folder. All the files are saved under thi path.

            c_logger (ConsoleLogger, optional): Console logger for printing training status. If not provided, the default
                console logger is used. Defaults to None.

            dashboard_logger Union[TensorboardLogger, WandbLogger]: Dashboard logger. If not provided, the tensorboard logger is used.
                Defaults to None.

            model (nn.Module, optional): Initialized and ready-to-train model. If it is not defined, `Trainer`
                initializes a model from the provided config. Defaults to None.

            get_model (Callable):
                A function that returns a model. It is used to initialize the model when `model` is not provided.
                It either takes the config as the only argument or does not take any argument.
                Defaults to None

            get_data_samples (Callable):
                A function that returns a list of training and evaluation samples. Used if `train_samples` and
                `eval_samples` are None. Defaults to None.

            train_samples (List):
                A list of training samples used by the model's `get_train_data_loader` to init the `dataset` and the
                `data_loader`. Defaults to None.

            eval_samples (List):
                A list of evaluation samples used by the model's `get_eval_data_loader` to init the `dataset` and the
                `data_loader`. Defaults to None.

            test_samples (List):
                A list of test samples used by the model's `get_test_data_loader` to init the `dataset` and the
                `data_loader`. If None, the ```model.test_run()``` is expected to load the data. Defaults to None.

            training_assets (Dict):
                A dictionary of assets to be used at training and passed to the model's ```train_log(), eval_log(), get_data_loader()```
                during training. It can include  `AudioProcessor` or/and `Tokenizer`. Defaults to {}.

            parse_command_line_args (bool):
                If true, parse command-line arguments and update `TrainerArgs` and model `config` values. Set it
                to false if you parse the arguments yourself. Defaults to True.

            callbacks (Dict[str, Callable]):
                A dictionary of callbacks to be used during training. The keys are the callback names and the values

            gpu (int):
                GPU ID to use for training If "CUDA_VISIBLE_DEVICES" is not set. Defaults to None.

        Example::

            Running trainer with a model.

            >>> args = TrainerArgs(...)
            >>> config = ModelConfig(...)
            >>> model = Model(config)
            >>> trainer = Trainer(args, config, output_path, model=model)
            >>> trainer.fit()

            TODO:
                - Wrap model for not calling .module in DDP.
                - Deepspeed integration
                - Profiler integration.
                - Overfitting to a batch.
                - TPU training
        """
        if parse_command_line_args:
            # parse command-line arguments to override TrainerArgs()
            args, coqpit_overrides = self.parse_argv(args)

            # get ready for training and parse command-line arguments to override the model config
            config, new_fields = self.init_training(args, coqpit_overrides, config)
        elif args.continue_path or args.restore_path:
            config, new_fields = self.init_training(args, {}, config)
        else:
            new_fields = {}

        # set the output path
        if args.continue_path:
            # use the same path as the continuing run
            output_path = args.continue_path
        else:
            # override the output path if it is provided
            output_path = config.output_path if output_path is None else output_path
            # create a new output folder name
            output_path = get_experiment_folder_path(config.output_path, config.run_name)
            os.makedirs(output_path, exist_ok=True)

        # copy training assets to the output folder
        copy_model_files(config, output_path, new_fields)

        # init class members
        self.args = args
        self.config = config
        self.output_path = output_path
        self.training_assets = training_assets
        self.grad_accum_steps = args.grad_accum_steps
        self.overfit_batch = args.overfit_batch
        self.skip_train_epoch = args.skip_train_epoch

        assert self.grad_accum_steps > 0, " [!] grad_accum_steps must be greater than 0."

        # setup logging
        log_file = os.path.join(self.output_path, f"trainer_{args.rank}_log.txt")
        self._setup_logger_config(log_file)

        # setup training environment
        self.use_cuda, self.num_gpus = self.setup_training_environment(args=args, config=config, gpu=gpu)

        # init loggers
        self.dashboard_logger, self.c_logger = self.init_loggers(
            self.args, self.config, output_path, dashboard_logger, c_logger
        )
        # self.c_logger.logger = logger

        if not self.config.log_model_step:
            self.config.log_model_step = self.config.save_step

        self.total_steps_done = 0
        self.epochs_done = 0
        self.restore_step = 0
        self.restore_epoch = 0
        self.best_loss = float("inf")
        self.train_loader = None
        self.test_loader = None
        self.eval_loader = None

        self.keep_avg_train = None
        self.keep_avg_eval = None

        self.use_apex = self._is_apex_available()
        self.use_amp_scaler = self.use_cuda if self.config.mixed_precision else self.config.use_grad_scaler

        if train_samples is not None:
            # use the provided samples
            self.train_samples = train_samples
            self.eval_samples = eval_samples
            self.test_samples = test_samples
        elif get_data_samples is not None:
            # run `get_data_samples` to init the data samples
            (  # pylint: disable=unbalanced-tuple-unpacking
                self.train_samples,
                self.eval_samples,
                self.test_samples,
            ) = self.run_get_data_samples(config, get_data_samples)
        else:
            # expecting to load the samples in `model.get_data_loader()`
            self.train_samples = None
            self.eval_samples = None
            self.test_samples = None

        # only use a subset of the samples if small_run is set
        self.setup_small_run(args.small_run)

        # init the model
        if model is None and get_model is None:
            raise ValueError("[!] `model` and `get_model` cannot both be None.")
        if model is not None:
            self.model = model
        else:
            self.run_get_model(self.config, get_model)

        # init model's training assets
        if hasattr(self.model, "init_for_training"):
            self.model.init_for_training()

        # setup criterion
        self.criterion = self.get_criterion(self.model)

        # DISTRUBUTED
        if self.num_gpus > 1:
            init_distributed(
                args.rank,
                self.num_gpus,
                args.group_id,
                self.config.distributed_backend,
                self.config.distributed_url,
            )

        if self.use_cuda:
            self.model.cuda()
            if isinstance(self.criterion, list):
                for criterion in self.criterion:
                    if isinstance(criterion, torch.nn.Module):
                        criterion.cuda()
            else:
                if isinstance(self.criterion, torch.nn.Module):
                    self.criterion.cuda()

        # setup optimizer
        self.optimizer = self.get_optimizer(self.model, self.config)

        # CALLBACK
        self.callbacks = TrainerCallback()
        self.callbacks.parse_callbacks_dict(callbacks)
        self.callbacks.on_init_start(self)

        # init AMP
        if self.use_amp_scaler:
            if self.use_apex:
                self.scaler = None
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # restore model
        if self.args.restore_path:
            (self.model, self.optimizer, self.scaler, self.restore_step, self.restore_epoch) = self.restore_model(
                self.config, args.restore_path, self.model, self.optimizer, self.scaler
            )
            self.scaler = torch.cuda.amp.GradScaler()

        # setup scheduler
        self.scheduler = self.get_scheduler(self.model, self.config, self.optimizer)
        self.scheduler = self.restore_scheduler(
            self.scheduler, self.args, self.config, self.restore_epoch, self.restore_step
        )

        # DISTRIBUTED
        if self.num_gpus > 1:
            self.model = DDP_th(self.model, device_ids=[args.rank], output_device=args.rank)

        # count model size
        num_params = count_parameters(self.model)
        logger.info("\n > Model has %i parameters", num_params)

        self.callbacks.on_init_end(self)
        self.dashboard_logger.add_config(config)
        self.save_training_script()

    def save_training_script(self):
        """Save the training script to tracking dashboard and output path."""
        file_path = sys.argv[0]
        if os.path.isfile(file_path):
            file_name = os.path.basename(file_path)
            self.dashboard_logger.add_artifact(file_or_dir=file_path, name=file_name, artifact_type="file")
            with open(file_path, "r", encoding="utf8") as f:
                self.dashboard_logger.add_text("training-script", f"{f.read()}", 0)
            shutil.copyfile(file_path, os.path.join(self.output_path, file_name))

    @staticmethod
    def parse_argv(args: Union[Coqpit, List]):
        """Parse command line arguments to init or override `TrainerArgs()`."""
        if isinstance(args, Coqpit):
            parser = args.init_argparse(arg_prefix="")
        else:
            train_config = TrainerArgs()
            parser = train_config.init_argparse(arg_prefix="")
        training_args, coqpit_overrides = parser.parse_known_args()
        args.parse_args(training_args)
        return args, coqpit_overrides

    @staticmethod
    def init_loggers(args: "Coqpit", config: "Coqpit", output_path: str, dashboard_logger=None, c_logger=None):
        """Init console and dashboard loggers.
        Use the given logger if passed externally else use config values to pick the right logger.
        Return a dashboard logger only for the rank 0 process in DDP
        Define a console logger for each process in DDP

        Args:
            args (argparse.Namespace or Coqpit): Parsed trainer arguments.
            config (Coqpit): Model config.
            output_path (str): Output path to save the training artifacts.
            dashboard_logger (DashboardLogger): Object passed to the trainer from outside.
            c_logger (ConsoleLogger): Object passed to the trained from outside.

        Returns:
            Initialized dashboard_logger and console_logger objects.
        """
        c_logger = ConsoleLogger() if c_logger is None else c_logger

        # only allow dashboard logging for the main process in DDP mode
        if args.rank:
            return DummyLogger(), c_logger
        if dashboard_logger is None:
            dashboard_logger = logger_factory(config, output_path)
        return dashboard_logger, c_logger

    def setup_small_run(self, small_run: int = None):
        """Use a subset of samples for training, evaluation and testing."""
        if small_run is not None:
            logger.info("[!] Small Run, only using %i samples.", small_run)
            self.train_samples = None if self.train_samples is None else self.train_samples[:small_run]
            self.eval_samples = None if self.eval_samples is None else self.eval_samples[:small_run]
            self.test_samples = None if self.test_samples is None else self.test_samples[:small_run]

    def init_training(
        self, args: TrainerArgs, coqpit_overrides: Dict, config: Coqpit = None
    ):  # pylint: disable=no-self-use
        """Initialize training and update model configs from command line arguments.

        Args:
            args (argparse.Namespace or dict like): Parsed trainer arguments.
            config_overrides (argparse.Namespace or dict like): Parsed config overriding arguments.
            config (Coqpit): Model config. If none, it is generated from `args`. Defaults to None.

        Returns:
            config (Coqpit): Config paramaters.
        """
        # set arguments for continuing training
        if args.continue_path:
            args.config_path = os.path.join(args.continue_path, "config.json")
            args.restore_path, best_model = get_last_checkpoint(args.continue_path)
            if not args.best_path:
                args.best_path = best_model
            # use the same config
            if config:
                config.load_json(args.config_path)
            else:
                coqpit = Coqpit()
                coqpit.load_json(args.config_path)

        # override config values from command-line args
        # TODO: Maybe it is better to do it outside
        if len(coqpit_overrides) > 0:
            config.parse_known_args(coqpit_overrides, relaxed_parser=True)

        # update the config.json fields and copy it to the output folder
        new_fields = {}
        if args.rank == 0:
            if args.restore_path:
                new_fields["restore_path"] = args.restore_path
            new_fields["github_branch"] = get_git_branch()
        return config, new_fields

    @staticmethod
    def setup_training_environment(args, config, gpu):
        if platform.system() != "Windows":
            # https://github.com/pytorch/pytorch/issues/973
            import resource  # pylint: disable=import-outside-toplevel

            rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

        # set and initialize Pytorch runtime
        use_cuda, num_gpus = setup_torch_training_env(
            cudnn_enable=config.cudnn_enable,
            cudnn_deterministic=config.cudnn_deterministic,
            cudnn_benchmark=config.cudnn_benchmark,
            use_ddp=args.use_ddp,
            training_seed=config.training_seed,
            gpu=gpu if args.gpu is None else args.gpu,
        )
        return use_cuda, num_gpus

    @staticmethod
    def run_get_model(config: Coqpit, get_model: Callable) -> nn.Module:
        """Run the `get_model` function and return the model.

        Args:
            config (Coqpit): Model config.

        Returns:
            nn.Module: initialized model.
        """
        if len(signature(get_model).sig.parameters) == 1:
            model = get_model(config)
        else:
            model = get_model()
        return model

    @staticmethod
    def run_get_data_samples(config: Coqpit, get_data_samples: Callable) -> nn.Module:
        if callable(get_data_samples):
            if len(signature(get_data_samples).sig.parameters) == 1:
                train_samples, eval_samples = get_data_samples(config)
            else:
                train_samples, eval_samples = get_data_samples()
            return train_samples, eval_samples
        return None, None

    def restore_model(
        self,
        config: Coqpit,
        restore_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler = None,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, torch.cuda.amp.GradScaler, int]:
        """Restore training from an old run. It restores model, optimizer, AMP scaler and training stats.

        Args:
            config (Coqpit): Model config.
            restore_path (str): Path to the restored training run.
            model (nn.Module): Model to restored.
            optimizer (torch.optim.Optimizer): Optimizer to restore.
            scaler (torch.cuda.amp.GradScaler, optional): AMP scaler to restore. Defaults to None.

        Returns:
            Tuple[nn.Module, torch.optim.Optimizer, torch.cuda.amp.GradScaler, int]: [description]
        """

        def _restore_list_objs(states, obj):
            if isinstance(obj, list):
                for idx, state in enumerate(states):
                    obj[idx].load_state_dict(state)
            else:
                obj.load_state_dict(states)
            return obj

        logger.info(" > Restoring from %s ...", os.path.basename(restore_path))
        checkpoint = load_fsspec(restore_path, map_location="cpu")
        try:
            logger.info(" > Restoring Model...")
            model.load_state_dict(checkpoint["model"])
            logger.info(" > Restoring Optimizer...")
            optimizer = _restore_list_objs(checkpoint["optimizer"], optimizer)
            if "scaler" in checkpoint and self.use_amp_scaler and checkpoint["scaler"]:
                logger.info(" > Restoring Scaler...")
                scaler = _restore_list_objs(checkpoint["scaler"], scaler)
        except (KeyError, RuntimeError, ValueError):
            logger.info(" > Partial model initialization...")
            model_dict = model.state_dict()
            model_dict = set_partial_state_dict(model_dict, checkpoint["model"], config)
            model.load_state_dict(model_dict)
            del model_dict

        optimizer = self.restore_lr(config, self.args, model, optimizer)

        logger.info(" > Model restored from step %i", checkpoint["step"])
        restore_step = checkpoint["step"] + 1  # +1 not to immediately checkpoint if the model is restored
        restore_epoch = checkpoint["epoch"]
        torch.cuda.empty_cache()
        return model, optimizer, scaler, restore_step, restore_epoch

    def restore_lr(self, config, args, model, optimizer):
        # use the same lr if continue training
        if not args.continue_path:
            if isinstance(optimizer, list):
                for idx, optim in enumerate(optimizer):
                    for group in optim.param_groups:
                        group["lr"] = self.get_lr(model, config)[idx]
            else:
                for group in optimizer.param_groups:
                    group["lr"] = self.get_lr(model, config)
        return optimizer

    #########################
    # DATA LOADING FUNCTIONS
    #########################

    def _get_loader(
        self,
        model: nn.Module,
        config: Coqpit,
        assets: Dict,
        is_eval: str,
        samples: List,
        verbose: bool,
        num_gpus: int,
    ) -> DataLoader:
        if num_gpus > 1:
            if hasattr(model.module, "get_data_loader"):
                loader = model.module.get_data_loader(
                    config,
                    assets,
                    is_eval,
                    samples,
                    verbose,
                    num_gpus,
                    self.args.rank,
                )
        else:
            if hasattr(model, "get_data_loader"):
                loader = model.get_data_loader(
                    config=config, assets=assets, is_eval=is_eval, samples=samples, verbose=verbose, num_gpus=num_gpus
                )
        return loader

    def get_train_dataloader(self, training_assets: Dict, samples: List, verbose: bool) -> DataLoader:
        """Initialize and return a training data loader.
        Call ```model.get_train_data_loader``` if it is implemented, else call ```model.get_data_loader```
        and set ```is_eval=False```.

        Args:
            ap (AudioProcessor): Audio processor.
            samples (List): Data samples used for training.
            verbose (bool): enable/disable printing loader stats at initialization.

        Returns:
            DataLoader: Initialized training data loader.
        """
        if self.num_gpus > 1:
            if hasattr(self.model.module, "get_train_data_loader"):
                loader = self.model.module.get_train_data_loader(
                    self.config,
                    self.training_assets,
                    samples,
                    verbose,
                    self.num_gpus,
                    self.args.rank,
                )
                return loader
        else:
            if hasattr(self.model, "get_train_data_loader"):
                loader = self.model.get_train_data_loader(
                    self.config, self.training_assets, samples, verbose, self.num_gpus
                )
                return loader

        return self._get_loader(
            self.model,
            self.config,
            training_assets,
            False,
            samples,
            verbose,
            self.num_gpus,
        )

    def get_eval_dataloader(self, training_assets: Dict, samples: List, verbose: bool) -> DataLoader:
        """Initialize and return a evaluation data loader.
        Call ```model.get_eval_data_loader``` if it is implemented, else call ```model.get_data_loader```
        and set ```is_eval=True```.

        Args:
            ap (AudioProcessor): Audio processor.
            samples (List): Data samples used for training.
            verbose (bool): enable/disable printing loader stats at initialization.

        Returns:
            DataLoader: Initialized training data loader.
        """
        if self.num_gpus > 1:
            if hasattr(self.model.module, "get_eval_data_loader"):
                loader = self.model.module.get_eval_data_loader(
                    self.config,
                    self.training_assets,
                    samples,
                    verbose,
                    self.num_gpus,
                    self.args.rank,
                )
                return loader
        else:
            if hasattr(self.model, "get_eval_data_loader"):
                loader = self.model.get_eval_data_loader(
                    self.config, self.training_assets, samples, verbose, self.num_gpus
                )
                return loader

        return self._get_loader(
            self.model,
            self.config,
            training_assets,
            True,
            samples,
            verbose,
            self.num_gpus,
        )

    def get_test_dataloader(self, training_assets: Dict, samples: List, verbose: bool) -> DataLoader:
        """Initialize and return a evaluation data loader.
        Call ```model.get_test_data_loader``` if it is implemented, else call ```model.get_data_loader```
        and set ```is_eval=True```.

        Args:
            ap (AudioProcessor): Audio processor.
            samples (List): Data samples used for training.
            verbose (bool): enable/disable printing loader stats at initialization.

        Returns:
            DataLoader: Initialized training data loader.
        """
        if self.num_gpus > 1:
            if hasattr(self.model.module, "get_test_data_loader"):
                loader = self.model.module.get_test_data_loader(
                    self.config,
                    self.training_assets,
                    samples,
                    verbose,
                    self.num_gpus,
                    self.args.rank,
                )
                return loader
        else:
            if hasattr(self.model, "get_test_data_loader"):
                loader = self.model.get_test_data_loader(
                    self.config, self.training_assets, samples, verbose, self.num_gpus
                )
                return loader

        return self._get_loader(
            self.model,
            self.config,
            training_assets,
            True,
            samples,
            verbose,
            self.num_gpus,
        )

    def format_batch(self, batch: List) -> Dict:
        """Format the dataloader output and return a batch.

        1. Call ```model.format_batch```.
        2. Pass the batch to the Device.
        3. Call ```model.format_batch_on_device```.

        Args:
            batch (List): Batch returned by the dataloader.

        Returns:
            Dict: Formatted batch.
        """
        try:
            if self.num_gpus > 1:
                batch = self.model.module.format_batch(batch)
            else:
                batch = self.model.format_batch(batch)
        except NotImplementedError:
            pass

        if isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = to_cuda(v)
        elif isinstance(batch, list):
            batch = [to_cuda(v) for v in batch]

        try:
            if self.num_gpus > 1:
                batch = self.model.module.format_batch_on_device(batch)
            else:
                batch = self.model.format_batch_on_device(batch)
        except NotImplementedError:
            pass
        return batch

    ######################
    # TRAIN FUNCTIONS
    ######################

    @staticmethod
    def master_params(optimizer: torch.optim.Optimizer):
        """Generator over parameters owned by the optimizer.

        Used to select parameters used by the optimizer for gradient clipping.

        Args:
            optimizer: Target optimizer.
        """
        for group in optimizer.param_groups:
            for p in group["params"]:
                yield p

    @staticmethod
    def _model_train_step(
        batch: Dict, model: nn.Module, criterion: nn.Module, optimizer_idx: int = None
    ) -> Tuple[Dict, Dict]:
        """
        Perform a trainig forward step. Compute model outputs and losses.

        Args:
            batch (Dict): [description]
            model (nn.Module): [description]
            criterion (nn.Module): [description]
            optimizer_idx (int, optional): [description]. Defaults to None.

        Returns:
            Tuple[Dict, Dict]: [description]
        """
        input_args = [batch, criterion]
        if optimizer_idx is not None:
            input_args.append(optimizer_idx)
        # unwrap model in DDP training
        if hasattr(model, "module"):
            return model.module.train_step(*input_args)
        return model.train_step(*input_args)

    def _optimize(
        self,
        batch: Dict,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: "AMPScaler",
        criterion: nn.Module,
        scheduler: Union[torch.optim.lr_scheduler._LRScheduler, List],  # pylint: disable=protected-access
        config: Coqpit,
        optimizer_idx: int = None,
        step_optimizer: bool = True,
        num_optimizers: int = 1,
    ) -> Tuple[Dict, Dict, int]:
        """Perform a forward - backward pass and run the optimizer.

        Args:
            batch (Dict): Input batch. If
            model (nn.Module): Model for training. Defaults to None.
            optimizer (Union[nn.optim.Optimizer, List]): Model's optimizer. If it is a list then, `optimizer_idx` must be defined to indicate the optimizer in use.
            scaler (AMPScaler): AMP scaler.
            criterion (nn.Module): Model's criterion.
            scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler used by the optimizer.
            config (Coqpit): Model config.
            optimizer_idx (int, optional): Target optimizer being used. Defaults to None.
            step_optimizer (bool, optional): Whether step the optimizer. If False, gradients are accumulated but
                but model parameters are not updated. Defaults to True.
            num_optimizers (int, optional): Number of optimizers. Defaults to 1.

        Raises:
            RuntimeError: When the loss is NaN.

        Returns:
            Tuple[Dict, Dict, int, torch.Tensor]: model outputs, losses, step time and gradient norm.
        """

        step_start_time = time.time()

        # forward pass and loss computation
        with torch.cuda.amp.autocast(enabled=config.mixed_precision):
            if optimizer_idx is not None:
                outputs, loss_dict = self._model_train_step(batch, model, criterion, optimizer_idx=optimizer_idx)
            else:
                outputs, loss_dict = self._model_train_step(batch, model, criterion)

        # skip the rest
        if not outputs:
            if loss_dict:
                raise RuntimeError(" [!] Model must return outputs when losses are computed.")
            step_time = time.time() - step_start_time
            return None, {}, step_time

        # accumulated gradients adjustment
        loss_dict["loss"] = loss_dict["loss"] / float(self.grad_accum_steps)

        # set gradient clipping threshold
        if "grad_clip" in config and config.grad_clip is not None:
            if optimizer_idx is not None:
                grad_clip = config.grad_clip[optimizer_idx]
            else:
                grad_clip = config.grad_clip
        else:
            grad_clip = 0.0  # meaning no gradient clipping

        # optimizer step
        grad_norm = 0
        update_lr_scheduler = True

        if self.use_amp_scaler:
            if self.use_apex:
                # TODO: verify AMP use for GAN training in TTS
                # https://nvidia.github.io/apex/advanced.html?highlight=accumulate#backward-passes-with-multiple-optimizers
                with amp.scale_loss(loss_dict["loss"], optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.callbacks.before_gradient_clipping(self)
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), grad_clip)
            else:
                # model optimizer step in mixed precision mode
                scaler.scale(loss_dict["loss"]).backward()
                # gradient accumulation
                if step_optimizer:
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        self.callbacks.before_gradient_clipping(self)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params(optimizer), grad_clip)
                    scale_prev = scaler.get_scale()
                    scaler.step(optimizer)
                    # update the scaler at the end of all the optimizer steps
                    if optimizer_idx is None or (optimizer_idx + 1 == num_optimizers):
                        scaler.update()
                        loss_dict["amp_scaler"] = scaler.get_scale()  # for logging
                    update_lr_scheduler = scale_prev <= scaler.get_scale()
        else:
            self.callbacks.before_backward_pass(self, loss_dict)
            # main model optimizer step
            loss_dict["loss"].backward()
            # gradient accumulation
            if step_optimizer:
                self.callbacks.before_gradient_clipping(self)
                if grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params(optimizer), grad_clip)
                optimizer.step()

        # pytorch skips the step when the norm is 0. So ignore the norm value when it is NaN
        if isinstance(grad_norm, torch.Tensor) and (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
            grad_norm = 0

        step_time = time.time() - step_start_time

        # setup lr
        if scheduler is not None and update_lr_scheduler and not self.config.scheduler_after_epoch and step_optimizer:
            scheduler.step()

        # detach losses for logging
        loss_dict_detached = self._detach_loss_dict(loss_dict)
        loss_dict_detached["loss"] = loss_dict_detached["loss"] * float(self.grad_accum_steps)

        if optimizer_idx is not None:
            loss_dict_detached[f"loss_{optimizer_idx}"] = loss_dict_detached.pop("loss")
            if step_optimizer:
                loss_dict_detached[f"grad_norm_{optimizer_idx}"] = grad_norm
        else:
            if step_optimizer:
                loss_dict_detached["grad_norm"] = grad_norm

        # zero-out optimizer
        if step_optimizer:
            optimizer.zero_grad(set_to_none=True)
        return outputs, loss_dict_detached, step_time

    def train_step(self, batch: Dict, batch_n_steps: int, step: int, loader_start_time: float) -> Tuple[Dict, Dict]:
        """Perform a training step on a batch of inputs and log the process.

        Args:
            batch (Dict): Input batch.
            batch_n_steps (int): Number of steps needed to complete an epoch. Needed for logging.
            step (int): Current step number in this epoch.
            loader_start_time (float): The time when the data loading is started. Needed for logging.

        Returns:
            Tuple[Dict, Dict]: Model outputs and losses.
        """
        self.callbacks.on_train_step_start(self)
        # format data
        batch = self.format_batch(batch)
        loader_time = time.time() - loader_start_time

        # conteainers to hold model outputs and losses for each optimizer.
        outputs_per_optimizer = None
        loss_dict = {}

        # gradient accumulation
        # TODO: grad accumulation for each optimizer
        step_optimizer = True
        if ((step + 1) % self.grad_accum_steps != 0) and (step + 1 != batch_n_steps):
            step_optimizer = False

        if not isinstance(self.optimizer, list):
            # training with a single optimizer
            outputs, loss_dict_new, step_time = self._optimize(
                batch,
                self.model,
                self.optimizer,
                self.scaler,
                self.criterion,
                self.scheduler,
                self.config,
                step_optimizer=step_optimizer,
                num_optimizers=1,
            )
            loss_dict.update(loss_dict_new)
        else:
            # training with multiple optimizers (e.g. GAN)
            outputs_per_optimizer = [None] * len(self.optimizer)
            total_step_time = 0
            for idx, optimizer in enumerate(self.optimizer):
                criterion = self.criterion
                # scaler = self.scaler[idx] if self.use_amp_scaler else None
                scaler = self.scaler
                scheduler = self.scheduler[idx]
                outputs, loss_dict_new, step_time = self._optimize(
                    batch,
                    self.model,
                    optimizer,
                    scaler,
                    criterion,
                    scheduler,
                    self.config,
                    idx,
                    step_optimizer=step_optimizer,
                    num_optimizers=len(self.optimizer),
                )
                # skip the rest if the model returns None
                total_step_time += step_time
                outputs_per_optimizer[idx] = outputs
                # merge loss_dicts from each optimizer
                # rename duplicates with the optimizer idx
                # if None, model skipped this optimizer
                if loss_dict_new is not None:
                    for k, v in loss_dict_new.items():
                        if k in loss_dict:
                            loss_dict[f"{k}-{idx}"] = v
                        else:
                            loss_dict[k] = v
                step_time = total_step_time
            outputs = outputs_per_optimizer

        # clear any pesky gradients after gradient accumulation
        if step_optimizer:
            self.model.zero_grad(set_to_none=True)

        # update avg runtime stats
        keep_avg_update = {}
        keep_avg_update["avg_loader_time"] = loader_time
        keep_avg_update["avg_step_time"] = step_time
        self.keep_avg_train.update_values(keep_avg_update)

        # update avg loss stats
        update_eval_values = {}
        for key, value in loss_dict.items():
            update_eval_values["avg_" + key] = value
        self.keep_avg_train.update_values(update_eval_values)

        # print training progress
        if self.total_steps_done % self.config.print_step == 0:
            # log learning rates
            lrs = {}
            if isinstance(self.optimizer, list):
                for idx, optimizer in enumerate(self.optimizer):
                    current_lr = self.optimizer[idx].param_groups[0]["lr"]
                    lrs.update({f"current_lr_{idx}": current_lr})
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]
                lrs = {"current_lr": current_lr}

            # log run-time stats
            loss_dict.update(lrs)
            loss_dict.update(
                {
                    "step_time": round(step_time, 4),
                    "loader_time": round(loader_time, 4),
                }
            )
            self.c_logger.print_train_step(
                batch_n_steps,
                step,
                self.total_steps_done,
                loss_dict,
                self.keep_avg_train.avg_values,
            )

        if self.args.rank == 0:
            # Plot Training Iter Stats
            # reduce TB load and don't log every step
            if self.total_steps_done % self.config.plot_step == 0:
                self.dashboard_logger.train_step_stats(self.total_steps_done, loss_dict)
            if self.total_steps_done % self.config.save_step == 0 and self.total_steps_done != 0:
                if self.config.save_checkpoints:
                    # checkpoint the model
                    target_avg_loss = self._pick_target_avg_loss(self.keep_avg_train)
                    save_checkpoint(
                        self.config,
                        self.model,
                        self.optimizer,
                        self.scaler if self.use_amp_scaler else None,
                        self.total_steps_done,
                        self.epochs_done,
                        self.output_path,
                        model_loss=target_avg_loss,
                        save_n_checkpoints=self.config.save_n_checkpoints,
                        save_func=self.dashboard_logger.save_model,
                    )

                    if self.total_steps_done % self.config.log_model_step == 0:
                        # log checkpoint as artifact
                        aliases = [
                            f"epoch-{self.epochs_done}",
                            f"step-{self.total_steps_done}",
                        ]
                        self.dashboard_logger.add_artifact(
                            file_or_dir=self.output_path, name="checkpoint", artifact_type="model", aliases=aliases
                        )

                # training visualizations
                if hasattr(self.model, "module") and hasattr(self.model.module, "train_log"):
                    self.model.module.train_log(
                        batch,
                        outputs,
                        self.dashboard_logger,
                        self.training_assets,
                        self.total_steps_done,
                    )
                elif hasattr(self.model, "train_log"):
                    self.model.train_log(
                        batch,
                        outputs,
                        self.dashboard_logger,
                        self.training_assets,
                        self.total_steps_done,
                    )

            self.dashboard_logger.flush()

        self.total_steps_done += 1
        self.callbacks.on_train_step_end(self)
        return outputs, loss_dict

    def train_epoch(self) -> None:
        """Main entry point for the training loop. Run training on the all training samples."""
        # initialize the data loader
        self.train_loader = self.get_train_dataloader(
            self.training_assets,
            self.train_samples,
            verbose=True,
        )
        # set model to training mode
        if self.num_gpus > 1:
            self.model.module.train()
        else:
            self.model.train()
        epoch_start_time = time.time()

        self.c_logger.print_train_start()
        loader_start_time = time.time()
        # TRAINING EPOCH -> iterate over the training samples
        batch_num_steps = len(self.train_loader)
        for cur_step, batch in enumerate(self.train_loader):
            _, _ = self.train_step(batch, batch_num_steps, cur_step, loader_start_time)
            loader_start_time = time.time()
            # RUN EVAL -> run evaluation epoch in the middle of training. Useful for big datasets.
            if self.config.run_eval_steps is not None and (self.total_steps_done % self.config.run_eval_steps == 0):
                self.eval_epoch()
                if self.num_gpus > 1:
                    self.model.module.train()
                else:
                    self.model.train()
        epoch_time = time.time() - epoch_start_time
        # scheduler step
        if self.scheduler is not None and self.config.scheduler_after_epoch:
            if isinstance(self.scheduler, list):
                for scheduler in self.scheduler:
                    if scheduler is not None:
                        scheduler.step()
            else:
                self.scheduler.step()
        # plot self.epochs_done Stats
        if self.args.rank == 0:
            epoch_stats = {"epoch_time": epoch_time}
            epoch_stats.update(self.keep_avg_train.avg_values)
            self.dashboard_logger.train_epoch_stats(self.total_steps_done, epoch_stats)
            if self.config.model_param_stats:
                self.dashboard_logger.model_weights(self.model, self.total_steps_done)
        torch.cuda.empty_cache()

    #######################
    # EVAL FUNCTIONS
    #######################

    @staticmethod
    def _model_eval_step(
        batch: Dict, model: nn.Module, criterion: nn.Module, optimizer_idx: int = None
    ) -> Tuple[Dict, Dict]:
        """
        Perform a evaluation forward pass. Compute model outputs and losses with no gradients.

        Args:
            batch (Dict): IBatch of inputs.
            model (nn.Module): Model to call evaluation.
            criterion (nn.Module): Model criterion.
            optimizer_idx (int, optional): Optimizer ID to define the closure in multi-optimizer training. Defaults to None.

        Returns:
            Tuple[Dict, Dict]: model outputs and losses.
        """
        input_args = [batch, criterion]
        if optimizer_idx is not None:
            input_args.append(optimizer_idx)
        if hasattr(model, "module"):
            return model.module.eval_step(*input_args)
        return model.eval_step(*input_args)

    def eval_step(self, batch: Dict, step: int) -> Tuple[Dict, Dict]:
        """Perform a evaluation step on a batch of inputs and log the process.

        Args:
            batch (Dict): Input batch.
            step (int): Current step number in this epoch.

        Returns:
            Tuple[Dict, Dict]: Model outputs and losses.
        """
        with torch.no_grad():
            outputs = []
            loss_dict = {}
            if not isinstance(self.optimizer, list):
                outputs, loss_dict = self._model_eval_step(batch, self.model, self.criterion)
            else:
                outputs = [None] * len(self.optimizer)
                for idx, _ in enumerate(self.optimizer):
                    criterion = self.criterion
                    outputs_, loss_dict_new = self._model_eval_step(batch, self.model, criterion, idx)
                    outputs[idx] = outputs_

                    if loss_dict_new:
                        loss_dict_new[f"loss_{idx}"] = loss_dict_new.pop("loss")
                        loss_dict.update(loss_dict_new)

            loss_dict = self._detach_loss_dict(loss_dict)

            # update avg stats
            update_eval_values = {}
            for key, value in loss_dict.items():
                update_eval_values["avg_" + key] = value
            self.keep_avg_eval.update_values(update_eval_values)

            if self.config.print_eval:
                self.c_logger.print_eval_step(step, loss_dict, self.keep_avg_eval.avg_values)
        return outputs, loss_dict

    def eval_epoch(self) -> None:
        """Main entry point for the evaluation loop. Run evaluation on the all validation samples."""
        self.eval_loader = (
            self.get_eval_dataloader(
                self.training_assets,
                self.eval_samples,
                verbose=True,
            )
            if self.config.run_eval
            else None
        )

        self.model.eval()
        self.c_logger.print_eval_start()
        loader_start_time = time.time()
        batch = None
        for cur_step, batch in enumerate(self.eval_loader):
            # format data
            batch = self.format_batch(batch)
            loader_time = time.time() - loader_start_time
            self.keep_avg_eval.update_values({"avg_loader_time": loader_time})
            outputs, _ = self.eval_step(batch, cur_step)
            loader_start_time = time.time()
        # plot epoch stats, artifacts and figures
        if self.args.rank == 0:
            if hasattr(self.model, "module") and hasattr(self.model.module, "eval_log"):
                self.model.module.eval_log(
                    batch,
                    outputs,
                    self.dashboard_logger,
                    self.training_assets,
                    self.total_steps_done,
                )
            elif hasattr(self.model, "eval_log"):
                self.model.eval_log(
                    batch,
                    outputs,
                    self.dashboard_logger,
                    self.training_assets,
                    self.total_steps_done,
                )
            self.dashboard_logger.eval_stats(self.total_steps_done, self.keep_avg_eval.avg_values)
        torch.cuda.empty_cache()

    ##################################
    # TESTING
    ##################################
    def test_run(self) -> None:
        """Run model test.

        Test run is expected to pass over test samples and produce logging artifacts.

        If ```model.test_run()``` is defined, it will be called and it is expected to set and execute everything
        in the model.

        Else if  ```mode.test()``` is defined, it will be called and it takes an test data loader as an argument
        and iterate over it.
        """
        self.model.eval()
        test_outputs = None
        if hasattr(self.model, "test_run") or (self.num_gpus > 1 and hasattr(self.model.module, "test_run")):
            # handle everything in ```model.test_run()`
            if self.num_gpus > 1:
                test_outputs = self.model.module.test_run(self.training_assets)
            else:
                test_outputs = self.model.test_run(self.training_assets)
        elif hasattr(self.model, "test") or (self.num_gpus > 1 and hasattr(self.model.module, "test")):
            self.test_loader = self.get_test_dataloader(
                self.training_assets,
                self.test_samples if self.test_samples else self.eval_samples,
                verbose=True,
            )
            # use test_loader to load test samples
            if self.num_gpus > 1:
                test_outputs = self.model.module.test(self.training_assets, self.test_loader, None)
            else:
                test_outputs = self.model.test(self.training_assets, self.test_loader, None)
        if hasattr(self.model, "test_log") or (self.num_gpus > 1 and hasattr(self.model.module, "test_log")):
            if self.num_gpus > 1:
                self.model.module.test_log(
                    test_outputs, self.dashboard_logger, self.training_assets, self.total_steps_done
                )
            else:
                self.model.test_log(test_outputs, self.dashboard_logger, self.training_assets, self.total_steps_done)

    def _restore_best_loss(self):
        """Restore the best loss from the args.best_path if provided else
        from the model (`args.continue_path`) used for resuming the training"""
        if self.args.continue_path and (self.restore_step != 0 or self.args.best_path):
            logger.info(" > Restoring best loss from %s ...", os.path.basename(self.args.best_path))
            ch = load_fsspec(self.args.restore_path, map_location="cpu")
            if "model_loss" in ch:
                self.best_loss = ch["model_loss"]
            logger.info(" > Starting with loaded last best loss %f", self.best_loss)

    def test(self, model=None, test_samples=None) -> None:
        """Run evaluation steps on the test data split. You can either provide the model and the test samples
        explicitly or the trainer use values from the initialization.

        Args:
            model (nn.Module, optional): Model to use for testing. If None, use the model given in the initialization.
                Defaults to None.

            test_samples (List[str], optional): List of test samples to use for testing. If None, use the test samples
                given in the initialization. Defaults to None.
        """

        logger.info(" > USING TEST SET...")
        self.keep_avg_eval = KeepAverage()

        if model is not None:
            self.model = model

        eval_samples_cache = self.eval_samples
        if test_samples is not None:
            self.eval_samples = test_samples
        else:
            self.eval_samples = self.test_samples

        self.eval_epoch()
        self.c_logger.print_epoch_end(self.epochs_done, self.keep_avg_eval.avg_values)
        self.eval_samples = eval_samples_cache

    ###################################
    # FIT FUNCTIONS
    ###################################

    def _fit(self) -> None:
        """🏃 train -> evaluate -> test for the number of epochs."""
        self._restore_best_loss()

        self.total_steps_done = self.restore_step

        for epoch in range(0, self.config.epochs):
            if self.num_gpus > 1:
                # let all processes sync up before starting with a new epoch of training
                dist.barrier()
            self.callbacks.on_epoch_start(self)
            self.keep_avg_train = KeepAverage()
            self.keep_avg_eval = KeepAverage() if self.config.run_eval else None
            self.epochs_done = epoch
            self.c_logger.print_epoch_start(epoch, self.config.epochs, self.output_path)
            if not self.skip_train_epoch:
                self.train_epoch()
            if self.config.run_eval:
                self.eval_epoch()
            if epoch >= self.config.test_delay_epochs and self.args.rank <= 0:
                self.test_run()
            self.c_logger.print_epoch_end(
                epoch,
                self.keep_avg_eval.avg_values if self.config.run_eval else self.keep_avg_train.avg_values,
            )
            if self.args.rank in [None, 0]:
                self.save_best_model()
            self.callbacks.on_epoch_end(self)

    def fit_with_largest_batch_size(self, starting_batch_size=2048) -> None:
        cuda_meminfo()
        bs = starting_batch_size
        while True:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                gc.collect()
                torch.cuda.empty_cache()
                self.config.batch_size = bs
                logger.info(" > current batch size: %i", self.config.batch_size)
                self._fit()
            except RuntimeError as exception:
                if bs > 1 and should_reduce_batch_size(exception):
                    bs //= 2
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    raise
            except Exception as exception: #pylint: disable=broad-except
                # catches the torch.cuda.OutOfMemoryError
                if bs > 1 and should_reduce_batch_size(exception):
                    bs //= 2
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    raise
            else:
                break

    def fit(self) -> None:
        """Where the ✨️magic✨️ happens..."""
        try:
            self._fit()
            if self.args.rank == 0:
                self.dashboard_logger.finish()
        except KeyboardInterrupt:
            self.callbacks.on_keyboard_interrupt(self)
            # if the output folder is empty remove the run.
            remove_experiment_folder(self.output_path)
            # clear the DDP processes
            if self.num_gpus > 1:
                dist.destroy_process_group()
            # finish the wandb run and sync data
            if self.args.rank == 0:
                self.dashboard_logger.finish()
            # stop without error signal
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)  # pylint: disable=protected-access
        except BaseException:  # pylint: disable=broad-except
            remove_experiment_folder(self.output_path)
            traceback.print_exc()
            sys.exit(1)

    def profile_fit(self, torch_profiler, epochs=None, small_run=None):
        """Run training under the torch profiler.

        Example::
            Run torch profiler to profile CPU, GPU and memory usage with Tensorboard logging.

            >>> import torch
            >>> profiler = torch.profiler.profile(
            >>>        activities=[
            >>>         torch.profiler.ProfilerActivity.CPU,
            >>>         torch.profiler.ProfilerActivity.CUDA,
            >>>     ],
            >>>     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            >>>     on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler/"),
            >>>     record_shapes=True,
            >>>     profile_memory=True,
            >>>     with_stack=True,
            >>> )
            >>> prof = trainer.profile_fit(profiler, epochs=1, small_run=64)
        """
        self.dashboard_logger = DummyLogger()
        # train the model for a custom number of epochs
        if epochs:
            self.config.epocshs = epochs
        # use a smaller set of training samples for profiling
        if small_run:
            self.setup_small_run(small_run)
        # run profiler
        self.config.run_eval = False
        self.config.test_delay_epochs = 9999999
        self.config.epochs = epochs
        # set a callback to progress the profiler
        self.callbacks_on_train_step_end = [  # pylint: disable=attribute-defined-outside-init
            lambda trainer: trainer.torch_profiler.step()
        ]
        # set the profiler to access in the Trainer
        self.torch_profiler = torch_profiler  # pylint: disable=attribute-defined-outside-init
        # set logger output for Tensorboard
        # self.torch_profiler.on_trace_ready = torch.profiler.tensorboard_trace_handler(self.output_path)
        self.torch_profiler.start()
        self.fit()
        self.torch_profiler.stop()
        return self.torch_profiler

    def save_best_model(self) -> None:
        """Save the best model. It only saves if the current target loss is smaller then the previous."""

        # set the target loss to choose the best model
        target_loss_dict = self._pick_target_avg_loss(self.keep_avg_eval if self.keep_avg_eval else self.keep_avg_train)

        # save the model and update the best_loss
        self.best_loss = save_best_model(
            target_loss_dict,
            self.best_loss,
            self.config,
            self.model,
            self.optimizer,
            self.scaler if self.use_amp_scaler else None,
            self.total_steps_done,
            self.epochs_done,
            self.output_path,
            keep_all_best=self.config.save_all_best,
            keep_after=self.config.save_best_after,
            save_func=self.dashboard_logger.save_model,
        )

    #####################
    # GET FUNCTIONS
    #####################

    @staticmethod
    def get_optimizer(model: nn.Module, config: Coqpit) -> Union[torch.optim.Optimizer, List]:
        """Receive the optimizer from the model if model implements `get_optimizer()` else
        check the optimizer parameters in the config and try initiating the optimizer.

        Args:
            model (nn.Module): Training model.
            config (Coqpit): Training configuration.

        Returns:
            Union[torch.optim.Optimizer, List]: A optimizer or a list of optimizers. GAN models define a list.
        """
        optimizer = None
        if hasattr(model, "get_optimizer"):
            try:
                optimizer = model.get_optimizer()
            except NotImplementedError:
                optimizer = None
        if optimizer is None:
            optimizer_name = config.optimizer
            optimizer_params = {} if config.optimizer_params is None else config.optimizer_params
            return get_optimizer(optimizer_name, optimizer_params, config.lr, model)
        return optimizer

    @staticmethod
    def get_lr(model: nn.Module, config: Coqpit) -> Union[float, List[float]]:
        """Set the initial learning rate by the model if model implements `get_lr()` else try setting the learning rate
        fromthe config.

        Args:
            model (nn.Module): Training model.
            config (Coqpit): Training configuration.

        Returns:
            Union[float, List[float]]: A single learning rate or a list of learning rates, one for each optimzier.
        """
        lr = None
        if hasattr(model, "get_lr"):
            try:
                lr = model.get_lr()
            except NotImplementedError:
                lr = None
        if lr is None:
            lr = config.lr
        return lr

    @staticmethod
    def get_scheduler(
        model: nn.Module, config: Coqpit, optimizer: Union[torch.optim.Optimizer, List]
    ) -> Union[torch.optim.lr_scheduler._LRScheduler, List]:  # pylint: disable=protected-access
        """Receive the scheduler from the model if model implements `get_scheduler()` else
        check the config and try initiating the scheduler.

        Args:
            model (nn.Module): Training model.
            config (Coqpit): Training configuration.

        Returns:
            Union[torch.optim.Optimizer, List]: A scheduler or a list of schedulers, one for each optimizer.
        """
        scheduler = None
        if hasattr(model, "get_scheduler"):
            try:
                scheduler = model.get_scheduler(optimizer)
            except NotImplementedError:
                scheduler = None
        if scheduler is None:
            lr_scheduler = config.lr_scheduler
            lr_scheduler_params = config.lr_scheduler_params
            return get_scheduler(lr_scheduler, lr_scheduler_params, optimizer)
        return scheduler

    @staticmethod
    def restore_scheduler(
        scheduler: Union["Scheduler", List], args: Coqpit, config: Coqpit, restore_epoch: int, restore_step: int
    ) -> Union["Scheduler", List]:
        """Restore scheduler wrt restored model."""
        if scheduler is not None:  # pylint: disable=too-many-nested-blocks
            if args.continue_path:
                if isinstance(scheduler, list):
                    for s in scheduler:
                        if s is not None:
                            if config.scheduler_after_epoch:
                                s.last_epoch = restore_epoch
                            else:
                                s.last_epoch = restore_step
                else:
                    if config.scheduler_after_epoch:
                        scheduler.last_epoch = restore_epoch
                    else:
                        scheduler.last_epoch = restore_step
        return scheduler

    @staticmethod
    def get_criterion(model: nn.Module) -> nn.Module:
        """Receive the criterion from the model. Model must implement `get_criterion()`.

        Args:
            model (nn.Module): Training model.

        Returns:
            nn.Module: Criterion layer.
        """
        criterion = None
        criterion = model.get_criterion()
        return criterion

    ####################
    # HELPER FUNCTIONS
    ####################

    @staticmethod
    def _detach_loss_dict(loss_dict: Dict) -> Dict:
        """Detach loss values from autograp.

        Args:
            loss_dict (Dict): losses.

        Returns:
            Dict: losses detached from autograph.
        """
        loss_dict_detached = {}
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                loss_dict_detached[key] = value
            else:
                loss_dict_detached[key] = value.detach().clone()
        return loss_dict_detached

    def _pick_target_avg_loss(self, keep_avg_target: KeepAverage) -> Dict:
        """Pick the target loss to compare models"""
        target_avg_loss = None

        # return if target loss defined in the model config
        # if not available in Dict use loss_1 as by default loss
        if "target_loss" in self.config and self.config.target_loss:
            if f"avg_{self.config.target_loss}" in keep_avg_target.avg_values.keys():
                return keep_avg_target[f"avg_{self.config.target_loss}"]
            return keep_avg_target["avg_loss_1"]

        # take the average of loss_{optimizer_idx} as the target loss when there are multiple optimizers
        if isinstance(self.optimizer, list):
            target_avg_loss = 0
            for idx in range(len(self.optimizer)):
                if f"avg_loss_{idx}" in keep_avg_target.avg_values:
                    target_avg_loss += keep_avg_target[f"avg_loss_{idx}"]
            target_avg_loss /= len(self.optimizer)
        else:
            target_avg_loss = keep_avg_target["avg_loss"]
        return target_avg_loss

    def _setup_logger_config(self, log_file: str) -> None:
        """Set up the logger based on the process rank in DDP."""

        logger_new = logging.getLogger("trainer")
        handler = logging.FileHandler(log_file, mode="a")
        fmt = logging.Formatter("")
        handler.setFormatter(fmt)
        logger_new.addHandler(handler)

        # only log to a file if rank > 0 in DDP
        if self.args.rank > 0:
            logger_new.handlers = [h for h in logger_new.handlers if not isinstance(h, logging.StreamHandler)]

    @staticmethod
    def _is_apex_available() -> bool:
        """Check if Nvidia's APEX is available."""
        return importlib.util.find_spec("apex") is not None
