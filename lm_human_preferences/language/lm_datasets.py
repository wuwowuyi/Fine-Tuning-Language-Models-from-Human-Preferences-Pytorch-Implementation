import os
import random
from pathlib import Path

import numpy as np
import torch

from lm_human_preferences.datasets import books, cnndm
from lm_human_preferences.datasets.tldr import tldr_generator

_registry: dict[str, "Dataset"] = {}


class Dataset:
    datasets_names = {
        'books': books.dataset_name,
        'cnndm': cnndm.dataset_name,
    }

    def __init__(
            self,
            name,
            *,
            generator=None,
    ):
        global _registry
        assert name not in _registry
        _registry[name] = self

        self.name = name
        self.generator = generator

    def tf_dataset(
            self,
            sequence_length,
            batch_size,
            *,
            mode,  # 'train' or 'test'
            encoder=None,
            # trims so that it starts right after start token
            start_token=None,
            # trims off last end_token
            end_token=None,
            padding_token=None,
    ):
        if padding_token is None:
            padding_token = encoder.padding_token

        data = np.memmap(Path(__file__).parent.parent / 'datasets' / f'{self.datasets_names[self.name]}_{mode}.bin',
                         dtype=np.uint16, mode='r')

        def _get_batch():
            batched = self.generator(data, batch_size)
            tokenized = np.empty((batch_size, sequence_length), dtype=np.int32)
            # strip off tokens before start_token and after end_token.
            # and pad tokens if len(tokens) < sequence_length
            for i, text in enumerate(batched):
                tokenized[i] = prepare_token(text, encoder, start_token, end_token, sequence_length)

            return tokenized

        return _get_batch


def prepare_token(text: str, encoder, start_token, end_token, sequence_length):
    tokens = encoder.encode(text)
    if start_token is not None:
        try:
            first_index = tokens.index(start_token) + 1
            if first_index < len(tokens):
                tokens = tokens[first_index:]
        except:
            pass

    tokens = tokens[:sequence_length]

    if end_token is not None:
        try:
            last_index = len(tokens) - tokens[::-1].index(end_token)
            tokens = tokens[:last_index]
        except:
            pass

    if len(tokens) < sequence_length:
        tokens = tokens + [encoder.padding_token] * (sequence_length - len(tokens))

    return tokens


def get_dataset(name) -> Dataset:
    global _registry
    return _registry[name]

CnnDm = Dataset(
    "cnndm",
    generator=cnndm.get_batch,
)

Tldr = Dataset(
    "tldr",
    generator=tldr_generator,
)

Books = Dataset(
    "books",
    generator=books.get_batch
)

def test_generator(mode, seed=0, shuffle=False):
    while True:
        yield ''.join([random.choice('abcdefghijklmnopqrstuvwxyz.') for _ in range(40)])

Test = Dataset(
    "test",
    generator=test_generator
)


"""
import tensorflow as tf
from lm_human_preferences.language.datasets import Books as ds
from lm_human_preferences.language.encodings import Main as encoding

e = encoding.get_encoder()
x = ds.tf_dataset(16, mode='test', encoder=e)
op = x.make_one_shot_iterator().get_next()
s = tf.Session()

while True:
    print(e.decode(s.run(op)['tokens']))
    input()
"""
