import numpy as np
import torch

from lm_human_preferences.utils import core_torch as utils


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def test_sample_buffer():
    set_seed(7)
    capacity = 100
    batch = 17
    lots = 100
    buffer = utils.SampleBuffer(capacity=capacity, schemas=dict(x=utils.Schema(torch.int32, ())))
    for i in range(20):
        buffer.add(x=torch.arange(batch) + batch * i)
        samples = buffer.sample(lots)['x']
        hi = batch * (i + 1)
        lo = max(0, hi - capacity)
        assert lo <= samples.min() <= lo + 3
        assert hi - 5 <= samples.max() < hi
        all_data_1 = buffer.data()
        all_data_2 = buffer.read(torch.arange(buffer.size()))
        assert torch.equal(all_data_1['x'], all_data_2['x'])
