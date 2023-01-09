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
pip install trainer
```

Prefer installing from Github as it is more stable.

## Implementing a model
Subclass and overload the functions in the [```TrainerModel()```](trainer/model.py)


## Training a model with auto optimization
See the [MNIST example](examples/train_mnist.py).


## Training a model with advanced optimization
See the [GAN training example](examples/train_simple_gan.py) with Gradient Accumulation


## Training with Batch Size Finder
see the test script [here](tests/test_train_batch_size_finder.py) for training with batch size finder.


The batch size finder starts at a default BS(defaults to 2048 but can also be user defined) and searches for the largest batch size that can fit on your hardware. you should expect for it to run multiple trainings until it finds it. to use it instead of calling ```trainer.fit()``` youll call ```trainer.fit_with_largest_batch_size(starting_batch_size=2048)``` with ```starting_batch_size``` being the batch the size you want to start the search with. very useful if you are wanting to use as much gpu mem as possible.

## Training with DDP

```console
$ python -m trainer.distribute --script path/to/your/train.py --gpus "0,1"
```

We don't use ```.spawn()``` to initiate multi-gpu training since it causes certain limitations.

- Everything must the pickable.
- ```.spawn()``` trains the model in subprocesses and the model in the main process is not updated.
- DataLoader with N processes gets really slow when the N is large.

## Adding a callback
👟 Supports callbacks to customize your runs. You can either set callbacks in your model implementations or give them
explicitly to the Trainer.

Please check `trainer.utils.callbacks` to see available callbacks.

Here is how you provide an explicit call back to a 👟Trainer object for weight reinitialization.

```python
def my_callback(trainer):
    print(" > My callback was called.")

trainer = Trainer(..., callbacks={"on_init_end": my_callback})
trainer.fit()
```

## Profiling example

- Create the torch profiler as you like and pass it to the trainer.
    ```python
    import torch
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler/"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    prof = trainer.profile_fit(profiler, epochs=1, small_run=64)
    then run Tensorboard
    ```
- Run the tensorboard.
    ```console
    tensorboard --logdir="./profiler/"
    ```

## Supported Experiment Loggers
- [Tensorboard](https://www.tensorflow.org/tensorboard) - actively maintained
- [ClearML](https://clear.ml/) - actively maintained
- [MLFlow](https://mlflow.org/)
- [Aim](https://aimstack.io/)
- [WandDB](https://wandb.ai/)

To add a new logger, you must subclass [BaseDashboardLogger](trainer/logging/base_dash_logger.py) and overload its functions.

## Anonymized Telemetry
We constantly seek to improve 🐸 for the community. To understand the community's needs better and address them accordingly, we collect stripped-down anonymized usage stats when you run the trainer.

Of course, if you don't want, you can opt out by setting the environment variable `TRAINER_TELEMETRY=0`.
