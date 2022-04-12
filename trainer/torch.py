import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler


class DistributedSamplerWrapper(DistributedSampler):
    """Wrapper over Sampler for distributed training. It allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with torch.nn.parallel.DistributedDataParallel. In such a case, each
    process can pass a torch.utils.data.DistributedSampler instance as a torch.utils.data.DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note:
        Dataset is assumed to be of constant size.

    Args:
        sampler: Sampler used for subsampling.
        num_replicas (int, optional): Number of processes participating in distributed training. By default,
            world_size is retrieved from the current distributed group.
        rank (int, optional): Rank of the current process within num_replicas. By default, rank is retrieved
            from the current distributed group.
        shuffle (bool, optional): If True, sampler will shuffle the indices. Default: True.
        seed (int, optional): random seed used to shuffle the sampler if shuffle=True. This number should be
            identical across all processes in the distributed group. Default: 0.

    Reference: https://github.com/pytorch/pytorch/issues/23430

    """

    def __init__(
        self,
        sampler,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__(
            sampler,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )

    def __iter__(self):
        indices = list(self.dataset)[: self.total_size]

        # Add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size, f"{len(indices)} != {self.total_size}"

        # Subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples, f"{len(indices)} != {self.num_samples}"

        return iter(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        elif hasattr(self.dataset, "generator"):
            self.dataset.generator = torch.Generator().manual_seed(self.seed + epoch)

    def state_dict(self):
        return self.dataset.state_dict()

    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict)


# pylint: disable=protected-access
class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps=0.1, last_epoch=-1):
        self.warmup_steps = float(warmup_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        return [
            base_lr * self.warmup_steps**0.5 * min(step * self.warmup_steps**-1.5, step**-0.5)
            for base_lr in self.base_lrs
        ]


# pylint: disable=protected-access
class StepwiseGradualLR(torch.optim.lr_scheduler._LRScheduler):
    """Hardcoded step-wise learning rate scheduling.
    Necessary for CapacitronVAE"""

    def __init__(self, optimizer, gradual_learning_rates, last_epoch=-1):
        self.gradual_learning_rates = gradual_learning_rates
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        step_thresholds = []
        rates = []
        for values in self.gradual_learning_rates:
            step_thresholds.append(values[0])
            rates.append(values[1])

        boolean_indeces = np.less_equal(step_thresholds, step)
        try:
            last_true = np.where(boolean_indeces == True)[0][-1]  # pylint: disable=singleton-comparison
        except IndexError:
            # For the steps larger than the last step in the list
            pass
        lr = rates[np.max(last_true, 0)]

        # Return last lr if step is above the set threshold
        lr = rates[-1] if step > step_thresholds[-1] else lr
        # Return first lr if step is below the second threshold - first is initial lr
        lr = rates[0] if step < step_thresholds[1] else lr

        return np.tile(lr, len(self.base_lrs))  # hack?
