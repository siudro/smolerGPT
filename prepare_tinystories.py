from concurrent.futures import ProcessPoolExecutor
from functools import partial
import glob
import numpy as np
import tiktoken
import json
import os
import requests
import torch
from tqdm import tqdm
import random


DATA_CACHE_DIR = "data"

def download_file(url: str, filename: str, chunk_size=1024):
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            bar.update(size)
    

def download():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print("Downloading TinyStories dataset...")
        download_file(data_url, data_filename)

    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print("Extracting TinyStories dataset...")
        os.system(f"tar -xvf {data_filename} -C {data_dir}")

    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.loads(f.read())
        print(data[0])

def process_shard(args):
    shard_id, shard = args
    tokenizer = tiktoken.get_encoding("gpt2")
    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example['story']
        text = text.strip();
        tokens = tokenizer.encode(text + '\n<|endoftext|>\n')
        all_tokens.extend(tokens)

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    tokenized_filename = shard.replace(".json", ".bin")
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())

    print(f"Saved {tokenized_filename}")

def pretokenize():
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    func = partial(process_shard)

    with ProcessPoolExecutor() as executor:
        executor.map(func, enumerate(shard_filenames))

    print("Done.")

class PreTokDataset(torch.utils.data.IterableDataset):

    def __init__(self, split: str, max_seq_len):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len

    def __iter__(self):
        bin_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
        shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1] 
        seed = 42
        rng = random.Random(seed)
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                data = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(data) // self.max_seq_len
                num_batches -= 1
                idxs = list(range(num_batches))
                rng.shuffle(idxs)
                for idx in idxs:
                    start = idx * self.max_seq_len
                    end = (idx + 1) * self.max_seq_len
                    chunk = torch.from_numpy(data[start:end].astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]

                    yield x, y
    
class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PreTokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            yield x, y


