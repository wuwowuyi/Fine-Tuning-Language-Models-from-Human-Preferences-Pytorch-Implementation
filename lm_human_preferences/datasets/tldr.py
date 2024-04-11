import os
import re

import ftfy
import numpy as np
import tiktoken
from datasets import DatasetDict
from tqdm import tqdm

from lm_human_preferences.datasets import books
from lm_human_preferences.utils import azure

dataset_name = 'openai-tldr'
enc = tiktoken.get_encoding('gpt2')  # consistent with encodings.Main


def get_batch(data, batch_size):
    return books.get_batch(data, batch_size, 10 * 2 ** 10)  # most article's length is below 10k


def prepare_tldr():
    """
    Download training json files provided by openai.
    Append eot_token to every data point, and concatenate all data points into a huge 1-D numpy array.
    The array is then saved as train.bin and val.bin under the datasets directory.
    """
    num_proc = 8  # num_core // 2
    train_file = azure.download_file_cached(
        f'https://openaipublic.blob.core.windows.net/lm-human-preferences/tldr/train-subset.json')
    val_file = azure.download_file_cached(
        f'https://openaipublic.blob.core.windows.net/lm-human-preferences/tldr/valid-subset.json')

    ds = DatasetDict.from_json({'train': train_file, 'val': val_file})

    def process(text):
        text = ftfy.fix_text(text['content'])
        text = re.sub(r"\n{3,}", "\n\n", text)
        ids = enc.encode_ordinary(text.strip())  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        return {"ids": ids, "len": len(ids)}

    # tokenize the data
    tokenized = ds.map(process, desc="tokenizing", remove_columns=['content'], num_proc=num_proc)

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.int64)
        filename = os.path.join(os.path.dirname(__file__), f'{dataset_name}_{split}.bin')
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = int((arr_len - 1) // (64 * 2 ** 20) + 1)  # 64MB per batch

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


if __name__ == '__main__':
    prepare_tldr()
