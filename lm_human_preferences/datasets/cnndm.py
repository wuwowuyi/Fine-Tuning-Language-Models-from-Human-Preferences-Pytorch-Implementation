import os
import re

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from lm_human_preferences.datasets import books

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

dataset_name = 'cnn_dailymail'  # cnn_dailymail from hugging face. 935K rows.
enc = tiktoken.get_encoding('gpt2')  # consistent with encodings.Main

def clean_up_start(text):
    if text[:2] == 'By':
        text = '\n'.join(text.split('\n')[2:])
    text = re.split(r'\(CNN\) +--', text)[-1]
    text = re.split(r"\(CNN\)", text[:100])[-1]+text[100:]
    text = re.sub(r"^and \w+\n", "", text)
    text = re.split(r".*UPDATED:\s+[0-9]{2}:[0-9]{2}.*[2011|2012|2013|2014|2015]", text)[-1]
    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    return text.strip()


def get_batch(data, batch_size):
    return map(clean_up_start, books.get_batch(data, batch_size, 10 * 2 ** 10))


def prepare_cnndm():
    """
    Load dataset from hugging face.
    Append eot_token to every data point, and concatenate all data points into a huge 1-D numpy array.
    The array is then saved as a train.bin or val.bin file under the datasets directory.
    """
    num_proc = 2  # num_cpu // 2
    dataset = load_dataset(dataset_name, '3.0.0', num_proc=num_proc)

    def process(example):
        ids = enc.encode_ordinary(example['article'])  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=['article', 'highlights'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{dataset_name}_{split}.bin')
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = int((arr_len - 1) // (16 * 2 ** 20) + 1)  # 16MB per batch

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        # generated train.bin is ~ 0.5G, val 22.6MB, test 19.7 MB.


if __name__ == '__main__':
    prepare_cnndm()
