"""
DISCLAIMER:
This code is provided "as-is" without any warranty of any kind, either expressed or implied,
including but not limited to the implied warranties of merchantability and fitness for a particular purpose.
The author assumes no liability for any damages or consequences resulting from the use of this code.
Use it at your own risk.

## Authors: Piotr KAWA
## December 2024
## Code adapted from coqui.ai - https://github.com/coqui-ai/TTS/blob/dev/TTS/utils/samplers.py
"""


import random
from collections import defaultdict

from torch.utils.data.sampler import Sampler, SubsetRandomSampler


class PerfectBatchSampler(Sampler):
    """
    Samples a mini-batch of indices for a balanced class batching

    Args:
        dataset_items(list): dataset items to sample from.
        classes (list): list of classes of dataset_items to sample from.
        batch_size (int): total number of samples to be sampled in a mini-batch.
        num_gpus (int): number of GPU in the data parallel mode.
        shuffle (bool): if True, samples randomly, otherwise samples sequentially.
        drop_last (bool): if True, drops last incomplete batch.
    """

    def __init__(
        self,
        dataset_items,
        classes: list,
        batch_size: int,
        num_classes_in_batch: int,
        num_gpus: int = 1,
        drop_last: bool = False,
        label_key: str = "class_name",
    ):
        super().__init__(dataset_items)
        assert (
            batch_size % (num_classes_in_batch * num_gpus) == 0
        ), "Batch size must be divisible by number of classes times the number of data parallel devices (if enabled)."

        label_indices = defaultdict(list)
        for idx, item in enumerate(dataset_items):
            label = item[label_key]
            label_indices[label].append(idx)

        self._samplers = [SubsetRandomSampler(label_indices[key]) for key in classes]

        self._batch_size = batch_size
        self._drop_last = drop_last
        self._dp_devices = num_gpus
        self._num_classes_in_batch = num_classes_in_batch

    def __iter__(self):
        batch = []
        if self._num_classes_in_batch != len(self._samplers):
            valid_samplers_idx = random.sample(
                range(len(self._samplers)), self._num_classes_in_batch
            )
        else:
            valid_samplers_idx = None

        iters = [iter(s) for s in self._samplers]
        done = False

        while True:
            b = []
            for i, it in enumerate(iters):
                if valid_samplers_idx is not None and i not in valid_samplers_idx:
                    continue
                idx = next(it, None)
                if idx is None:
                    done = True
                    break
                b.append(idx)
            if done:
                break
            batch += b
            if len(batch) == self._batch_size:
                # yield batch
                for b in batch:
                    yield b
                batch = []
                if valid_samplers_idx is not None:
                    valid_samplers_idx = random.sample(
                        range(len(self._samplers)), self._num_classes_in_batch
                    )

        if not self._drop_last:
            if len(batch) > 0:
                groups = len(batch) // self._num_classes_in_batch
                if groups % self._dp_devices == 0:
                    yield batch
                else:
                    batch = batch[
                        : (groups // self._dp_devices)
                        * self._dp_devices
                        * self._num_classes_in_batch
                    ]
                    if len(batch) > 0:
                        yield batch

    def __len__(self):
        class_batch_size = self._batch_size // self._num_classes_in_batch
        return min(
            ((len(s) + class_batch_size - 1) // class_batch_size)
            for s in self._samplers
        )
