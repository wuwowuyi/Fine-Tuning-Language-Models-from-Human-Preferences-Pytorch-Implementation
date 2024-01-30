
import random

from datasets import load_dataset


def books_generator(mode, seed=0, shuffle=False, comm=None):
    # broken link
    # datas = [
    #     json.loads(line) for line in
    #     open(gcs.download_file_cached(f'https://openaipublic.blob.core.windows.net/lm-human-preferences/datasets/book_passages/{mode}.jsonl', comm=comm))
    # ]

    # Instead, use bookcorpus dataset from hugging face. 74M rows
    dataset = load_dataset("bookcorpus", split=mode, streaming=True)  # returns an IterableDataset
    if shuffle:
        random.seed(seed)
        dataset = dataset.shuffle(seed, buffer_size=100000)  # to review. or use flatten_indices()

    for x in dataset:
        yield x['text']
