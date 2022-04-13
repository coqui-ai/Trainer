import os
import time

import torch

from tests.utils.mnist import MnistModel, MnistModelConfig
from trainer import Trainer, TrainerArgs
from trainer.generic_utils import KeepAverage

is_cuda = torch.cuda.is_available()


def test_train_mnist():
    model = MnistModel()
    # Test StepwiseGradualLR
    config = MnistModelConfig(
        lr_scheduler="StepwiseGradualLR",
        lr_scheduler_params={
            "gradual_learning_rates": [
                [0, 1e-3],
                [2, 1e-4],
            ]
        },
        scheduler_after_epoch=False,
    )
    trainer = Trainer(TrainerArgs(), config, model=model, output_path=os.getcwd(), gpu=0 if is_cuda else None)
    trainer.train_loader = trainer.get_train_dataloader(
        trainer.training_assets,
        trainer.train_samples,
        verbose=True,
    )
    trainer.keep_avg_train = KeepAverage()

    lr_0 = trainer.scheduler.get_lr()
    trainer.train_step(next(iter(trainer.train_loader)), len(trainer.train_loader), 0, time.time())
    lr_1 = trainer.scheduler.get_lr()
    trainer.train_step(next(iter(trainer.train_loader)), len(trainer.train_loader), 1, time.time())
    lr_2 = trainer.scheduler.get_lr()
    assert lr_0 == 1e-3
    assert lr_1 == 1e-3
    assert lr_2 == 1e-4
