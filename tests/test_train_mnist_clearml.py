import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from trainer import Trainer, TrainerArgs, TrainerConfig, TrainerModel

is_cuda = torch.cuda.is_available()


@dataclass
class MnistModelConfig(TrainerConfig):
    optimizer: str = "Adam"
    lr: float = 0.001
    epochs: int = 1
    print_step: int = 1
    plot_step: int = 5
    save_step: int = 30
    dashboard_logger: str = "clearml"
    project_name: str = "pytorch-mnist-clearml"
    run_name: str = "test-run"


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

    @staticmethod
    def get_criterion():
        return torch.nn.NLLLoss()

    def get_data_loader(
        self, config, assets, is_eval, samples, verbose, num_gpus, rank=0
    ):  # pylint: disable=unused-argument
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = MNIST(os.getcwd(), train=not is_eval, download=True, transform=transform)
        dataset.data = dataset.data[:256]
        dataset.targets = dataset.targets[:256]
        mnist_train = DataLoader(dataset, batch_size=8)
        return mnist_train


def test_train_mnist():
    model = MnistModel()
    trainer = Trainer(
        TrainerArgs(), MnistModelConfig(), model=model, output_path=os.getcwd(), gpu=0 if is_cuda else None
    )
    trainer.fit()


if __name__ == "__main__":
    test_train_mnist()
