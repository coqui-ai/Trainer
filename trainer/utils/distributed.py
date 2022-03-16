# edited from https://github.com/fastai/imagenet-fast/blob/master/imagenet_nv/distributed.py
import os
from functools import wraps
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist


def get_rank() -> int:
    rank_keys = ("RANK", "SLURM_PROCID", "LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)

    return 0


def rank_zero_only(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


rank_zero_only.rank = getattr(rank_zero_only, "rank", get_rank())


@rank_zero_only
def rank_zero_print(message: str, *args, **kwargs) -> None:  # pylint: disable=unused-argument
    print(message)


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
