import os
import re

import ftfy
import numpy as np
import pandas as pd
import tiktoken

from lm_human_preferences.datasets import books
from lm_human_preferences.utils import azure

dataset_name = 'openai-tldr'
enc = tiktoken.get_encoding('gpt2')  # consistent with encodings.Main


def get_batch(data, batch_size):
    return books.get_batch(data, batch_size, 1000)  # most article's length is below 1000


def process(example):
    text = ftfy.fix_text(example)
    text = re.sub(r"\n{3,}", "\n\n", text)
    ids = enc.encode_ordinary(text.strip())  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    return ids, len(ids)


def cat(series: pd.Series):
    return np.concatenate(series.to_numpy())


def prepare_tldr():
    """
    Download training json files provided by openai.
    Append eot_token to every data point, and concatenate all data points into a huge 1-D numpy array.
    The array is then saved as train.bin and val.bin under the datasets directory.
    """
    train_file = azure.download_file_cached(
        f'https://openaipublic.blob.core.windows.net/lm-human-preferences/tldr/train-subset.json', cache_dir=os.path.dirname(__file__))
    val_file = azure.download_file_cached(
        f'https://openaipublic.blob.core.windows.net/lm-human-preferences/tldr/valid-subset.json', cache_dir=os.path.dirname(__file__))

    for split, file in zip(('train','val'), (train_file, val_file)):
        with open(file) as f:
            df = pd.read_json(f)

        ids, ids_size = zip(*df['content'].map(process))
        arr_len = np.sum(ids_size, dtype=np.int64)
        print(f"content average length is {round(arr_len / len(ids))}")

        filename = os.path.join(os.path.dirname(__file__), f'{dataset_name}_{split}.bin')
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        chunks = np.array_split(pd.Series(ids), 128)
        idx = 0
        for chunk in map(cat, chunks):  # 128 is arbitrary
            arr[idx: idx + len(chunk)] = chunk
            idx += len(chunk)
        arr.flush()


if __name__ == '__main__':
    prepare_tldr()
