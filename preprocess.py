import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import sentencepiece as spm
import glob

from tokenizer import Tokenizer

DATA_CACHE_DIR = Path("data")
DATA_CACHE_DIR.mkdir(exist_ok=True)

def download_file(url: str, filename: str, chunk_size: int = 1024) -> None:
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

def download() -> None:
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = DATA_CACHE_DIR / "TinyStories_all_data.tar.gz"
    
    if not data_filename.exists():
        print("Downloading TinyStories dataset...")
        download_file(data_url, str(data_filename))

    data_dir = DATA_CACHE_DIR / "TinyStories_all_data"
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
        print("Extracting TinyStories dataset...")
        os.system(f"tar -xvf {data_filename} -C {data_dir}")

def train_vocab(vocab_size: int) -> None:
    prefix = DATA_CACHE_DIR / f"tok{vocab_size}"
    tiny_file = DATA_CACHE_DIR / "tiny.txt"
    
    data_dir = DATA_CACHE_DIR / "TinyStories_all_data"
    shard_filenames = sorted(glob.glob(str(data_dir / "*.json")))
    
    with open(tiny_file, "w") as f:
        for shard in shard_filenames[:10]:
            with open(shard, "r") as g:
                data = json.load(g)
            for example in data:
                f.write(example["story"].strip() + "\n")
    
    spm.SentencePieceTrainer.train(
        input=str(tiny_file),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r"\342\201\207 ",
        normalization_rule_name="identity",
    )

def process_shard(args: tuple, vocab_size: int) -> None:
    shard_id, shard = args
    tokenizer_model = DATA_CACHE_DIR / f"tok{vocab_size}.model"
    tokenizer = Tokenizer(str(tokenizer_model))
    
    with open(shard, "r") as f:
        data = json.load(f)
    
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example['story'].strip()
        tokens = tokenizer.encode(text, bos=True, eos=True)
        all_tokens.extend(tokens)

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    tokenized_filename = str(shard).replace(".json", ".bin")
    
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())

def pretokenize(vocab_size: int) -> None:
    data_dir = DATA_CACHE_DIR / "TinyStories_all_data"
    shard_filenames = sorted(glob.glob(str(data_dir / "*.json")))
    
    func = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(func, enumerate(shard_filenames))
