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

    command_continue = f"python tests/utils/train_mnist.py --continue_path {continue_path}"
    run_cli(command_continue)

    assert number_of_checkpoints < len(glob.glob(os.path.join(continue_path, "*.pth")))
    shutil.rmtree(continue_path)
