<p align="center"><img src="https://user-images.githubusercontent.com/1402048/151947958-0bcadf38-3a82-4b4e-96b4-a38d3721d737.png" align="right" height="255px" /></p>

# 👟 Trainer
An opinionated general purpose model trainer on PyTorch with a simple code base.

## Installation

From Github:

```console
git clone https://github.com/coqui-ai/Trainer
cd Trainer
make install
```

From PyPI:

```console
pip install coqui-trainer
```

Prefer installing from Github as it is more stable.

## Implementing a model
Subclass and overload the functions in the [```TrainerModel()```](trainer/model.py)

## Training a model
See the test script [here](tests/test_train_mnist.py) training a basic MNIST model.

## Training with DDP

```console
$ python -m trainer.distribute --script path/to/your/train.py --gpus "0,1"
```

We don't use ```.spawn()``` to initiate multi-gpu training since it causes certain limitations.

- Everything must the pickable.
- ```.spawn()``` trains the model in subprocesses and the model in the main process is not updated.
- DataLoader with N processes gets really slow when the N is large.

## Supported Experiment Loggers
- [Tensorboard](https://www.tensorflow.org/tensorboard) - actively maintained
- [ClearML](https://clear.ml/) - actively maintained
- [MLFlow](https://mlflow.org/)
- [Aim](https://aimstack.io/)
- [WandDB](https://wandb.ai/)

To add a new logger, you must subclass [BaseDashboardLogger](trainer/logging/base_dash_logger.py) and overload its functions.


