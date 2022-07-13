import torch


class LargestBatchSizeFinder:
    """Batch size finder that finds the max batch size your hardware can fit

    Args:
          beginning batch size (int): a batch size to start with. set this to max batch size you think you can train
          with. Defaults to 2048
    """
    def __init__(self, beginning_batch_size: int = 2048):
        self.bs = beginning_batch_size


class OpenaiBatchSizeFinder:
    """Batch size finder based on: https://arxiv.org/pdf/2109.03784.pdf"""
    pass

