# edited from https://github.com/fastai/imagenet-fast/blob/master/imagenet_nv/distributed.py
import os
from functools import wraps
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def is_main_process():
    return get_rank() == 0


def rank_zero_only(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        if is_main_process():
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


@rank_zero_only
def rank_zero_print(message: str, *args, **kwargs) -> None:  # pylint: disable=unused-argument
    print(message)


@rank_zero_only
def rank_zero_logger_info(message: str, logger: "Logger", *args, **kwargs) -> None:  # pylint: disable=unused-argument
    logger.info(message)


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= num_gpus
    return rt


def init_distributed(rank, num_gpus, group_name, dist_backend, dist_url):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        dist_backend,
        init_method=dist_url,
        world_size=num_gpus,
        rank=rank,
        group_name=group_name,
    )
