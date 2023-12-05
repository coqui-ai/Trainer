import glob
import os
import shutil

from tests import run_cli


def test_continue_train():
    output_path = "output/"

    command_train = "python tests/utils/train_mnist.py"
    run_cli(command_train)

    continue_path = max(glob.glob(os.path.join(output_path, "*/")), key=os.path.getmtime)
    number_of_checkpoints = len(glob.glob(os.path.join(continue_path, "*.pth")))

    # Continue training from the best model
    command_continue = f"python tests/utils/train_mnist.py --continue_path {continue_path} --coqpit.run_eval_steps=1"
    run_cli(command_continue)

    assert number_of_checkpoints < len(glob.glob(os.path.join(continue_path, "*.pth")))

    # Continue training from the last checkpoint
    for best in glob.glob(os.path.join(continue_path, "best_model*")):
        os.remove(best)
    run_cli(command_continue)

    # Continue training from a specific checkpoint
    restore_path = os.path.join(continue_path, "checkpoint_5.pth")
    command_continue = f"python tests/utils/train_mnist.py --restore_path {restore_path}"
    run_cli(command_continue)
    shutil.rmtree(continue_path)
