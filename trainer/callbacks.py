from typing import Callable, Dict


class TrainerCallback:
    def __init__(self) -> None:
        self.callbacks_on_init_start = []
        self.callbacks_on_init_end = []
        self.callbacks_on_epoch_start = []
        self.callbacks_on_epoch_end = []
        self.callbacks_on_train_step_start = []
        self.callbacks_on_train_step_end = []
        self.callbacks_on_keyboard_interrupt = []

    def parse_callbacks_dict(self, callbacks_dict: Dict[str, Callable]) -> None:
        for key, value in callbacks_dict.items():
            if key == "on_init_start":
                self.callbacks_on_init_start.append(value)
            elif key == "on_init_end":
                self.callbacks_on_init_end.append(value)
            elif key == "on_epoch_start":
                self.callbacks_on_epoch_start.append(value)
            elif key == "on_epoch_end":
                self.callbacks_on_epoch_end.append(value)
            elif key == "on_train_step_start":
                self.callbacks_on_train_step_start.append(value)
            elif key == "on_train_step_end":
                self.callbacks_on_train_step_end.append(value)
            elif key == "on_keyboard_interrupt":
                self.callbacks_on_keyboard_interrupt.append(value)
            else:
                raise ValueError(f"Invalid callback key: {key}")

    def on_init_start(self, trainer) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "on_init_start"):
                trainer.model.module.on_init_start(trainer)
        else:
            if hasattr(trainer.model, "on_init_start"):
                trainer.model.on_init_start(trainer)

        if hasattr(trainer.criterion, "on_init_start"):
            trainer.criterion.on_init_start(trainer)

        if hasattr(trainer.optimizer, "on_init_start"):
            trainer.optimizer.on_init_start(trainer)

        if self.callbacks_on_init_start:
            for callback in self.callbacks_on_init_start:
                callback(trainer)

    def on_init_end(self, trainer) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "on_init_end"):
                trainer.model.module.on_init_end(trainer)
        else:
            if hasattr(trainer.model, "on_init_end"):
                trainer.model.on_init_end(trainer)

        if hasattr(trainer.criterion, "on_init_end"):
            trainer.criterion.on_init_end(trainer)

        if hasattr(trainer.optimizer, "on_init_end"):
            trainer.optimizer.on_init_end(trainer)

        if len(self.callbacks_on_init_end) > 0:
            for callback in self.callbacks_on_init_end:
                callback(trainer)

    def on_epoch_start(self, trainer) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "on_epoch_start"):
                trainer.model.module.on_epoch_start(trainer)
        else:
            if hasattr(trainer.model, "on_epoch_start"):
                trainer.model.on_epoch_start(trainer)

        if hasattr(trainer.criterion, "on_epoch_start"):
            trainer.criterion.on_epoch_start(trainer)

        if hasattr(trainer.optimizer, "on_epoch_start"):
            trainer.optimizer.on_epoch_start(trainer)

        if self.callbacks_on_epoch_start:
            for callback in self.callbacks_on_epoch_start:
                callback(trainer)

    def on_epoch_end(self, trainer) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "on_epoch_end"):
                trainer.model.module.on_epoch_end(trainer)
        else:
            if hasattr(trainer.model, "on_epoch_end"):
                trainer.model.on_epoch_end(trainer)

        if hasattr(trainer.criterion, "on_epoch_end"):
            trainer.criterion.on_epoch_end(trainer)

        if hasattr(trainer.optimizer, "on_epoch_end"):
            trainer.optimizer.on_epoch_end(trainer)

        if self.callbacks_on_epoch_end:
            for callback in self.callbacks_on_epoch_end:
                callback(trainer)

    @staticmethod
    def before_backward_pass(trainer, loss_dict) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "before_backward_pass"):
                trainer.model.module.before_backward_pass(loss_dict, trainer.optimizer)
        else:
            if hasattr(trainer.model, "before_backward_pass"):
                trainer.model.before_backward_pass(loss_dict, trainer.optimizer)

    @staticmethod
    def before_gradient_clipping(trainer) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "before_gradient_clipping"):
                trainer.model.module.before_gradient_clipping()
        else:
            if hasattr(trainer.model, "before_gradient_clipping"):
                trainer.model.before_gradient_clipping()

    def on_train_step_start(self, trainer) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "on_train_step_start"):
                trainer.model.module.on_train_step_start(trainer)
        else:
            if hasattr(trainer.model, "on_train_step_start"):
                trainer.model.on_train_step_start(trainer)

        if hasattr(trainer.criterion, "on_train_step_start"):
            trainer.criterion.on_train_step_start(trainer)

        if hasattr(trainer.optimizer, "on_train_step_start"):
            trainer.optimizer.on_train_step_start(trainer)

        if self.callbacks_on_train_step_start:
            for callback in self.callbacks_on_train_step_start:
                callback(trainer)

    def on_train_step_end(self, trainer) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "on_train_step_end"):
                trainer.model.module.on_train_step_end(trainer)
        else:
            if hasattr(trainer.model, "on_train_step_end"):
                trainer.model.on_train_step_end(trainer)

        if hasattr(trainer.criterion, "on_train_step_end"):
            trainer.criterion.on_train_step_end(trainer)

        if hasattr(trainer.optimizer, "on_train_step_end"):
            trainer.optimizer.on_train_step_end(trainer)

        if self.callbacks_on_train_step_end:
            for callback in self.callbacks_on_train_step_end:
                callback(trainer)

    def on_keyboard_interrupt(self, trainer) -> None:
        if hasattr(trainer.model, "module"):
            if hasattr(trainer.model.module, "on_keyboard_interrupt"):
                trainer.model.module.on_keyboard_interrupt(trainer)
        else:
            if hasattr(trainer.model, "on_keyboard_interrupt"):
                trainer.model.on_keyboard_interrupt(trainer)

        if hasattr(trainer.criterion, "on_keyboard_interrupt"):
            trainer.criterion.on_keyboard_interrupt(trainer)

        if hasattr(trainer.optimizer, "on_keyboard_interrupt"):
            trainer.optimizer.on_keyboard_interrupt(trainer)

        if self.callbacks_on_keyboard_interrupt:
            for callback in self.callbacks_on_keyboard_interrupt:
                callback(trainer)
