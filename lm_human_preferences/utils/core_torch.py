from dataclasses import dataclass
from typing import Any, Tuple, Optional

import numpy as np
import torch
from torch.nn import functional as F


def get_device(t: torch.Tensor):
    device_ord: int = t.get_device()
    return 'cpu' if device_ord == -1 else 'cuda:' + str(device_ord)


@dataclass
class Schema:
    dtype: Any
    shape: Tuple[Optional[int],...]


def exact_div(a: int, b: int) -> int:
    q = a // b
    if a != q * b:
        raise ValueError('Inexact division: %s / %s = %s' % (a, b, a / b))
    return q


def take_top_p_logits(logits: torch.Tensor, p: float):
    """Nucleus sampling.
    The implementation here is to find the minimum logits and use them to filter.

    logits.shape = (b, n_vocab) where b is batch size
    """
    sorted_logits = torch.sort(logits, descending=True, dim=-1)[0]
    probs = F.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    mask = torch.cumsum(cum_probs >= p, dim=-1) <= 1
    selected = torch.where(mask, sorted_logits, float('inf'))
    min_logits = torch.min(selected, dim=-1, keepdim=True)[0]
    return torch.where(logits >= min_logits, logits, -float('inf'))


def safe_zip(*args):
    """Zip, but require all sequences to be the same length."""
    args = tuple(map(tuple, args))
    for a in args[1:]:
        if len(args[0]) != len(a):
            raise ValueError(f'Lengths do not match: {[len(a) for a in args]}')
    return zip(*args)


def ceil_div(a, b):
    return (a - 1) // b + 1


def pearson_r(x: torch.Tensor, y: torch.Tensor):
    assert x.dim() == 1
    assert y.dim() == 1
    x_var, x_mean = torch.var_mean(x, dim=0, correction=0)
    y_var, y_mean = torch.var_mean(y, dim=0, correction=0)
    cov = torch.mean((x - x_mean)*(y - y_mean), dim=0)
    return cov / torch.sqrt(x_var * y_var)


class SampleBuffer:
    """A circular buffer for storing and sampling data.

    Data can be added to the buffer with `add`, and old data will be dropped.  If you need to
    control where the buffer is stored, wrap the constructor call in a `with tf.device` block:

        with tf.device('cpu:0'):
            buffer = SampleBuffer(...)
    """

    def __init__(self, *, capacity: int, schemas: dict[str, Schema]) -> None:
        self._capacity = capacity  # max # data items in buffer
        self._total = 0  # total # data items added
        self._vars = {
            n: torch.empty((capacity,) + s.shape, dtype=s.dtype, requires_grad=False,
                           device='cpu', pin_memory=True)
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
        i0 = (self._total - n) % capacity  # index of first new data item
        i0n = i0 + n
        i1 = min(i0n, capacity)  # max index of new data item (exclusive)
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
        """indices: A 1-D Tensor of indices to read from. Each index must be less than capacity."""
        return {k: v[indices] for k, v in self._vars.items()}

    def data(self):
        return {k: v[:self.size()] for k, v in self._vars.items()}

    def sample(self, n, seed=None):
        """Sample n entries with replacement."""
        size = self.size()
        indices = torch.randint(high=size, size=(n,))
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


class FlatStats:
    """A bunch of statistics stored as a single flat tensor."""

    def __init__(self, keys, flat):
        keys = tuple(keys)
        flat = torch.as_tensor(flat, dtype=torch.float32)
        assert [len(keys)] == list(flat.shape)
        self.keys = keys
        self.flat = flat

    @staticmethod
    def from_dict(stats):
        for k, v in stats.items():
            if v.dtype != torch.float32:
                raise ValueError('Statistic %s has dtype %r, expected %r' % (k, v.dtype, torch.float32))
        keys = tuple(sorted(stats.keys()))
        flat = torch.stack([stats[k] for k in keys])
        return FlatStats(keys, flat)

    def concat(self, more):
        dups = set(self.keys) & set(more.keys)
        if dups:
            raise ValueError('Duplicate statistics: %s' % ', '.join(dups))
        return FlatStats(self.keys + more.keys, torch.concat([self.flat, more.flat], dim=0))

    def as_dict(self):
        return dict(safe_zip(self.keys, self.flat))

    def with_values(self, flat):
        return FlatStats(self.keys, flat)

    def map_flat(self, f):
        return FlatStats(self.keys, f(self.flat))

