from distutils.command.config import config

from mnist import MnistModel, MnistModelConfig

from trainer import Trainer, TrainerArgs


def main():
    """Run `MNIST` model training from scratch or from previous checkpoint."""
    # init args and config
    train_args = TrainerArgs()
    config = MnistModelConfig()

    # init the model from config
    model = MnistModel()

    # init the trainer and 🚀
    trainer = Trainer(
        train_args,
        config,
        config.output_path,
        model=model,
        train_samples=model.get_data_loader(config, None, False, None, None, None),
        eval_samples=model.get_data_loader(config, None, True, None, None, None),
        parse_command_line_args=True,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
