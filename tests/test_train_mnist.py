import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from trainer import Trainer, TrainerArgs, TrainerConfig, TrainerModel


@dataclass
class MnistModelConfig(TrainerConfig):
    optimizer: str = "Adam"
    lr: float = 0.001
    epochs: int = 1
    print_step: int = 500
    dashboard_logger: str = "tensorboard"


class MnistModel(TrainerModel):
    def __init__(self):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, height, width)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size, _, _, _ = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    def train_step(self, batch, criterion):
        x, y = batch
        logits = self(x)
        loss = criterion(logits, y)
        return {"model_outputs": logits}, {"loss": loss}

    def eval_step(self, batch, criterion):
        x, y = batch
        logits = self(x)
        loss = criterion(logits, y)
        return {"model_outputs": logits}, {"loss": loss}

    def get_criterion(self):
        return torch.nn.NLLLoss()

    def get_data_loader(self, config, assets, is_eval, samples, verbose, num_gpus, rank=0):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = MNIST(os.getcwd(), train=not is_eval, download=True, transform=transform)
        mnist_train = DataLoader(dataset, batch_size=config.batch_size)
        return mnist_train


def test_train_mnist():
    model = MnistModel()
    trainer = Trainer(TrainerArgs(), MnistModelConfig(), model=model, output_path=os.getcwd())

    trainer.fit()
    loss1 = trainer.keep_avg_train["avg_loss"]

    trainer.fit()
    loss2 = trainer.keep_avg_train["avg_loss"]

    assert loss1 > loss2
