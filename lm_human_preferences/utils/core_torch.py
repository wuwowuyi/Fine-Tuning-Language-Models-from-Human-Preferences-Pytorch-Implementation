from dataclasses import dataclass
from typing import Any, Tuple, Optional

import torch


@dataclass
class Schema:
    dtype: Any
    shape: Tuple[Optional[int],...]


def exact_div(a: int, b: int) -> int:
    q = a // b
    if a != q * b:
        raise ValueError('Inexact division: %s / %s = %s' % (a, b, a / b))
    return q


class SampleBuffer:
    """A circular buffer for storing and sampling data.

    Data can be added to the buffer with `add`, and old data will be dropped.  If you need to
    control where the buffer is stored, wrap the constructor call in a `with tf.device` block:

        with tf.device('cpu:0'):
            buffer = SampleBuffer(...)
    """

    def __init__(self, *, capacity: int, schemas: dict[str, Schema], name=None) -> None:
        # TODO: place on CPU?
        self._capacity = capacity
        self._total = 0
        self._vars = {
            n: torch.zeros((capacity,) + s.shape, dtype=s.dtype)
            for n, s in schemas.items()
        }

    def add(self, **data):
        """Add new data to the end of the buffer, dropping old data if we exceed capacity."""
        # Check input shapes
        if data.keys() != self._vars.keys():
            raise ValueError('data.keys() = %s != %s' % (sorted(data.keys()), sorted(self._vars.keys())))
        first = next(iter(data.values()))
        pre = first.shape[:1]  # torch.Size([batch_size])
        for k, d in data.items():
            try:
                d.shape == (pre + self._vars[k].shape[1:])
            except ValueError as e:
                raise ValueError('%s, key %s' % (e, k))

        # Enqueue
        n = first.shape[0]
        capacity = self._capacity
        self._total += n
        i0 = (self._total - n) % capacity  # first index of new data item
        i0n = i0 + n  # total #items to deal with
        i1 = min(i0n, capacity)  # last index of new data item (exclusive)
        i2 = i1 % capacity  # i1 if i0n <= capacity else 0
        i3 = i0n % capacity  # i1 if i0n <= capacity else i0n % capacity
        for k, d in data.items():
            p1, p2 = torch.split(d, [i1 - i0, i3 - i2])
            self._vars[k][i0:i1] = p1
            self._vars[k][i2:i3] = p2

    def total(self):
        """Total number of entries ever added, including those already discarded."""
        return self._total

    def size(self):
        """Current number of entries."""
        return min(self.total(), self._capacity)

    def read(self, indices):
        """indices: A 1-D Tensor of indices to read from. Each index must be less than
        capacity."""
        return {k: v[indices] for k, v in self._vars.items()}

    def data(self):
        return {k: v[:self.size()] for k, v in self._vars.items()}

    def sample(self, n, seed=None):
        """Sample n entries with replacement."""
        size = self.size()
        indices = torch.randint(size, (n,))
        return self.read(indices)

    def write(self, indices, updates):
        """
        indices: A 1-D Tensor of indices to write to. Each index must be less than `capacity`.
        update: A dictionary of new values, where each entry is a tensor with the same length as `indices`.
        """
        for k, v in updates.items():
            self._vars[k][indices] = v

    def write_add(self, indices, deltas):
        for k, d in deltas.items():
            self._vars[k][indices] += d

