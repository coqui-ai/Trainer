"""
This example shows training of a simple GAN model with MNIST dataset using Gradient Accumulation and Advanced
Optimization where you call optimizer steps manually.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from trainer import Trainer, TrainerConfig, TrainerModel
from trainer.trainer import TrainerArgs

is_cuda = torch.cuda.is_available()


# pylint: skip-file


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


@dataclass
class GANModelConfig(TrainerConfig):
    epochs: int = 1
    print_step: int = 2
    training_seed: int = 666


class GANModel(TrainerModel):
    def __init__(self):
        super().__init__()
        data_shape = (1, 28, 28)
        self.generator = Generator(latent_dim=100, img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)

    def forward(self, x):
        ...

    def optimize(self, batch, trainer):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], 100)
        z = z.type_as(imgs)

        # train discriminator
        imgs_gen = self.generator(z)
        logits = self.discriminator(imgs_gen.detach())
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)
        loss_fake = trainer.criterion(logits, fake)

        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        logits = self.discriminator(imgs)
        loss_real = trainer.criterion(logits, valid)
        loss_disc = (loss_real + loss_fake) / 2

        # step dicriminator
        _, _ = self.scaled_backward(loss_disc, None, trainer, trainer.optimizer[0])

        if trainer.total_steps_done % trainer.grad_accum_steps == 0:
            trainer.optimizer[0].step()
            trainer.optimizer[0].zero_grad()

        # train generator
        imgs_gen = self.generator(z)

        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        logits = self.discriminator(imgs_gen)
        loss_gen = trainer.criterion(logits, valid)

        # step generator
        _, _ = self.scaled_backward(loss_gen, None, trainer, trainer.optimizer[1])
        if trainer.total_steps_done % trainer.grad_accum_steps == 0:
            trainer.optimizer[1].step()
            trainer.optimizer[1].zero_grad()
        return {"model_outputs": logits}, {"loss_gen": loss_gen, "loss_disc": loss_disc}

    @torch.no_grad()
    def eval_step(self, batch, criterion):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], 100)
        z = z.type_as(imgs)

        imgs_gen = self.generator(z)
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        logits = self.discriminator(imgs_gen)
        loss_gen = trainer.criterion(logits, valid)
        return {"model_outputs": logits}, {"loss_gen": loss_gen}

    def get_optimizer(self):
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001, betas=(0.5, 0.999))
        return [discriminator_optimizer, generator_optimizer]

    def get_criterion(self):
        return nn.BCELoss()

    def get_data_loader(
        self, config, assets, is_eval, samples, verbose, num_gpus, rank=0
    ):  # pylint: disable=unused-argument
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset = MNIST(os.getcwd(), train=not is_eval, download=True, transform=transform)
        dataset.data = dataset.data[:64]
        dataset.targets = dataset.targets[:64]
        dataloader = DataLoader(dataset, batch_size=config.batch_size, drop_last=True, shuffle=True)
        return dataloader


if __name__ == "__main__":

    config = GANModelConfig()
    config.batch_size = 64
    config.grad_clip = None

    model = GANModel()
    trainer = Trainer(TrainerArgs(), config, model=model, output_path=os.getcwd(), gpu=0 if is_cuda else None)
    trainer.config.epochs = 10
    trainer.fit()
