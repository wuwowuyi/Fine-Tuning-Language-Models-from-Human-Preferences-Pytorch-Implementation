import os

import numpy as np
import tiktoken
from datasets import load_dataset  # hugging face datasets library
from tqdm import tqdm

# OpenAI's books dataset link is broken. use datasets hosted by hugging face instead.
dataset_name = 'bookcorpus'  # bookcorpus from hugging face. 74M rows.
enc = tiktoken.get_encoding('gpt2')  # consistent with encodings.Main

def get_batch(data, batch_size, read_length=100):
    batched = []
    sep = enc.decode([enc.eot_token])  # enc.eot_token was used to separate sentences. see prepare_books().
    ix = np.random.randint(len(data) - read_length * 5, size=batch_size)  # 5 is arbitrary
    for i in ix:
        data_i = data[i: i + read_length]
        xs = enc.decode(data_i).split(sep)
        while len(xs) < 3:
            data_i = np.concatenate((data_i, data[i + data_i.shape[0]: i + data_i.shape[0] + read_length]))
            xs = enc.decode(data_i).split(sep)
        batched.append(xs[1])  # the second is a complete data example
    return batched

def prepare_books():
    """
    Load books dataset from hugging face.
    Append eot_token to every data point, concatenate into a 1-D huge numpy array,
    and save as a train.bin file under the datasets directory.
    """
    num_proc = 8  # num_cpu // 2
    dataset = load_dataset(dataset_name, num_proc=num_proc)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)

    def process(example):
        ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.int64)  # int64 matches python int.
        filename = os.path.join(os.path.dirname(__file__), f'{dataset_name}_{split}.bin')
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

        # the generated train.bin is 2.3 GB, and test.bin 1.1MB.


if __name__ == '__main__':
    prepare_books()
