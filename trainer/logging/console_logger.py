import datetime
import logging
from dataclasses import dataclass

from trainer.utils.distributed import rank_zero_only

logger = logging.getLogger("trainer")


@dataclass(frozen=True)
class tcolors:
    OKBLUE: str = "\033[94m"
    HEADER: str = "\033[95m"
    OKGREEN: str = "\033[92m"
    WARNING: str = "\033[93m"
    FAIL: str = "\033[91m"
    ENDC: str = "\033[0m"
    BOLD: str = "\033[1m"
    UNDERLINE: str = "\033[4m"


class ConsoleLogger:
    def __init__(self):
        # TODO: color code for value changes
        # use these to compare values between iterations
        self.old_train_loss_dict = None
        self.old_epoch_loss_dict = None
        self.old_eval_loss_dict = None

    @staticmethod
    def log_with_flush(msg: str):
        if logger is not None:
            logger.info(msg)
            for handler in logger.handlers:
                handler.flush()
        else:
            print(msg, flush=True)

    @staticmethod
    def get_time():
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")

    @rank_zero_only
    def print_epoch_start(self, epoch, max_epoch, output_path=None):
        self.log_with_flush(
            "\n{}{} > EPOCH: {}/{}{}".format(tcolors.UNDERLINE, tcolors.BOLD, epoch, max_epoch, tcolors.ENDC),
        )
        if output_path is not None:
            self.log_with_flush(f" --> {output_path}")

    @rank_zero_only
    def print_train_start(self):
        self.log_with_flush(f"\n{tcolors.BOLD} > TRAINING ({self.get_time()}) {tcolors.ENDC}")

    @rank_zero_only
    def print_train_step(self, batch_steps, step, global_step, loss_dict, avg_loss_dict):
        indent = "     | > "
        self.log_with_flush("")
        log_text = "{}   --> TIME: {} -- STEP: {}/{} -- GLOBAL_STEP: {}{}\n".format(
            tcolors.BOLD, self.get_time(), step, batch_steps, global_step, tcolors.ENDC
        )
        for key, value in loss_dict.items():
            # print the avg value if given
            if f"avg_{key}" in avg_loss_dict.keys():
                log_text += "{}{}: {}  ({})\n".format(indent, key, str(value), str(avg_loss_dict[f"avg_{key}"]))
            else:
                log_text += "{}{}: {} \n".format(indent, key, str(value))
        self.log_with_flush(log_text)

    # pylint: disable=unused-argument
    @rank_zero_only
    def print_train_epoch_end(self, global_step, epoch, epoch_time, print_dict):
        indent = "     | > "
        log_text = f"\n{tcolors.BOLD}   --> TRAIN PERFORMACE -- EPOCH TIME: {epoch_time:.2f} sec -- GLOBAL_STEP: {global_step}{tcolors.ENDC}\n"
        for key, value in print_dict.items():
            log_text += "{}{}: {}\n".format(indent, key, str(value))
        self.log_with_flush(log_text)

    @rank_zero_only
    def print_eval_start(self):
        self.log_with_flush(f"\n{tcolors.BOLD} > EVALUATION {tcolors.ENDC}\n")

    @rank_zero_only
    def print_eval_step(self, step, loss_dict, avg_loss_dict):
        indent = "     | > "
        log_text = f"{tcolors.BOLD}   --> STEP: {step}{tcolors.ENDC}\n"
        for key, value in loss_dict.items():
            # print the avg value if given
            if f"avg_{key}" in avg_loss_dict.keys():
                log_text += "{}{}: {}  ({})\n".format(indent, key, str(value), str(avg_loss_dict[f"avg_{key}"]))
            else:
                log_text += "{}{}: {} \n".format(indent, key, str(value))
        self.log_with_flush(log_text)

    @rank_zero_only
    def print_epoch_end(self, epoch, avg_loss_dict):
        indent = "     | > "
        log_text = "\n  {}--> EVAL PERFORMANCE{}\n".format(tcolors.BOLD, tcolors.ENDC)
        for key, value in avg_loss_dict.items():
            # print the avg value if given
            color = ""
            sign = "+"
            diff = 0
            if self.old_eval_loss_dict is not None and key in self.old_eval_loss_dict:
                diff = value - self.old_eval_loss_dict[key]
                if diff < 0:
                    color = tcolors.OKGREEN
                    sign = ""
                elif diff > 0:
                    color = tcolors.FAIL
                    sign = "+"
            log_text += "{}{}:{} {} {}({}{})\n".format(indent, key, color, str(value), tcolors.ENDC, sign, str(diff))
        self.old_eval_loss_dict = avg_loss_dict
        self.log_with_flush(log_text)
