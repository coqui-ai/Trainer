#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pathlib
import subprocess
import time

from trainer import TrainerArgs, logger


def distribute():
    """
    Call 👟Trainer training script in DDP mode.
    """
    parser = TrainerArgs().init_argparse(arg_prefix="")
    parser.add_argument("--script", type=str, help="Target training script to distibute.")
    parser.add_argument(
        "--gpus",
        type=str,
        help='GPU IDs to be used for distributed training in the format ```"0,1"```. Used if ```CUDA_VISIBLE_DEVICES``` is not set.',
    )
    args, unargs = parser.parse_known_args()

    gpus = get_gpus(args)

    group_id = time.strftime("%Y_%m_%d-%H%M%S")

    # set arguments for train.py
    folder_path = pathlib.Path(__file__).parent.absolute()
    if os.path.exists(os.path.join(folder_path, args.script)):
        command = [os.path.join(folder_path, args.script)]
    else:
        command = [args.script]

    # Pass all the TrainerArgs fields
    command.append(f"--continue_path={args.continue_path}")
    command.append(f"--restore_path={args.restore_path}")
    command.append(f"--group_id=group_{group_id}")
    command.append("--use_ddp=true")
    command += unargs
    command.append("")

    # run processes
    processes = []
    for rank, local_gpu_id in enumerate(gpus):
        my_env = os.environ.copy()
        my_env["PYTHON_EGG_CACHE"] = f"/tmp/tmp{local_gpu_id}"
        my_env["RANK"] = f"{local_gpu_id}"
        my_env["CUDA_VISIBLE_DEVICES"] = f"{','.join(gpus)}"
        command[-1] = f"--rank={rank}"
        # prevent stdout for processes with rank != 0
        stdout = None
        p = subprocess.Popen(["python3"] + command, stdout=stdout, env=my_env)  # pylint: disable=consider-using-with
        processes.append(p)
        logger.info(command)

    for p in processes:
        p.wait()


def get_gpus(args):
    # set active gpus from CUDA_VISIBLE_DEVICES or --gpus
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != "":
        gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        gpus = args.gpus
    gpus = list(map(str.strip, gpus.split(",")))
    return gpus


if __name__ == "__main__":
    distribute()
