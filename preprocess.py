import os
import argparse
import json
import requests
from pathlib import Path
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import sentencepiece as spm
import glob
import random

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


def process_shard(args: tuple, vocab_size: int) -> None:
    """Process a single shard with memory-efficient batching"""
    shard_id, shard = args
    tokenizer_model = DATA_CACHE_DIR / f"tok{vocab_size}.model"
    tokenizer = Tokenizer(str(tokenizer_model))

    # Use memory-efficient batch processing
    batch_size = 1000  # Process stories in smaller batches
    all_tokens = []
    
    with open(shard, "r") as f:
        data = json.load(f)
        
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_tokens = []
        
        for example in batch:
            if isinstance(example, dict) and "story" in example:
                text = example["story"].strip()
                # Handle negative tokens by mapping them to positive values within vocab range
                tokens = tokenizer.encode(text, bos=True, eos=True)
                tokens = [t % vocab_size if t < 0 else min(t, vocab_size-1) for t in tokens]
                batch_tokens.extend(tokens)
        
        # Convert batch to uint16 and store
        if batch_tokens:
            all_tokens.extend(batch_tokens)
        
        # Clear memory
        del batch_tokens
        del batch
    
    # Clear original data
    del data
    
    # Convert to numpy array efficiently
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    tokenized_filename = str(shard).replace(".json", ".bin")

    # Save efficiently using memory mapping
    with open(tokenized_filename, "wb") as f:
        all_tokens.tofile(f)


def train_vocab(vocab_size: int, num_files: int = 1, sample_rate: float = 0.1) -> None:
    """Train vocabulary with extreme memory efficiency"""
    temp_file = DATA_CACHE_DIR / "temp_stories.txt"
    chunk_size = 1000  # Process in smaller batches
    
    try:
        with open(temp_file, "w", encoding="utf-8") as outfile:
            for i in range(min(num_files, 50)):
                print(f"Processing file {i+1}/{num_files}")
                filename = f"{DATA_CACHE_DIR}/TinyStories_all_data/data{i:02d}.json"
                
                # Read and process file in chunks
                with open(filename, "r") as infile:
                    stories = []
                    data = json.load(infile)
                    
                    # Process in chunks to save memory
                    for j in range(0, len(data), chunk_size):
                        chunk = data[j:j + chunk_size]
                        num_stories = max(1, int(len(chunk) * sample_rate))
                        selected = random.sample(chunk, num_stories)
                        
                        for story in selected:
                            if isinstance(story, dict) and "story" in story:
                                outfile.write(story["story"].strip() + "\n")
                        
                        # Clear memory
                        del chunk
                    
                    # Clear memory
                    del data
        
        # Train with minimal memory usage
        spm.SentencePieceTrainer.train(
            input=str(temp_file),
            model_prefix=str(DATA_CACHE_DIR / f"tok{vocab_size}"),
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0,
            eos_id=1,
            unk_id=2,
            bos_id=-1,
            pad_piece="<|pad|>",
            eos_piece="<|endoftext|>",
            unk_piece="<|unk|>",
            input_sentence_size=50000,     # Reduced for memory efficiency
            shuffle_input_sentence=True,
            max_sentence_length=1024,      # Reduced for efficiency
            num_threads=1,                 # Single thread for stability
            train_extremely_large_corpus=False,
            max_sentencepiece_length=8,    # Reduced for efficiency
            minloglevel=1                 # Reduce logging output
        )
    finally:
        if temp_file.exists():
            try:
                os.remove(temp_file)
            except:
                pass


def process_chunk(stories: list, sample_rate: float, outfile) -> None:
    """Process a chunk of stories efficiently"""
    if not stories:
        return
        
    # Sample stories
    num_stories = max(1, int(len(stories) * sample_rate))
    sampled_indices = random.sample(range(len(stories)), num_stories)
    
    # Write sampled stories
    for idx in sampled_indices:
        story = stories[idx]
        if isinstance(story, dict) and "story" in story:
            outfile.write(story["story"].strip() + "\n")


def pretokenize(vocab_size: int, num_files: int = 1) -> None:
    """Pretokenize with extreme memory efficiency"""
    try:
        data_dir = DATA_CACHE_DIR / "TinyStories_all_data"
        shard_filenames = sorted(glob.glob(str(data_dir / "*.json")))[:num_files]

        for i, shard in enumerate(shard_filenames):
            print(f"Pretokenizing file {i+1}/{len(shard_filenames)}")
            # Process with memory mapping
            process_shard((i, shard), vocab_size)
            
    except Exception as e:
        print(f"Error during pretokenization: {str(e)}")
        raise


def prepare_dataset(vocab_size: int, num_files: int = 1, sample_rate: float = 0.1) -> None:
    """
    Prepare dataset with better error handling
    """
    try:
        print(f"Preparing dataset with {num_files} files and {sample_rate*100}% sampling rate")
        print("Step 1: Downloading dataset...")
        download()
        
        print("\nStep 2: Training vocabulary...")
        train_vocab(vocab_size, num_files, sample_rate)
        
        print("\nStep 3: Pretokenizing dataset...")
        pretokenize(vocab_size, num_files)
        
        print("\nDataset preparation complete!")
        
    except Exception as e:
        print(f"\nError during dataset preparation: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TinyStories dataset")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    download_parser = subparsers.add_parser(
        "download", help="Download TinyStories dataset"
    )

    vocab_parser = subparsers.add_parser("train-vocab", help="Train vocabulary")
    vocab_parser.add_argument(
        "--vocab-size", type=int, required=True, help="Size of vocabulary to train"
    )

    pretok_parser = subparsers.add_parser("pretokenize", help="Pretokenize the dataset")
    pretok_parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Vocabulary size to use for tokenization",
    )

    prepare_parser = subparsers.add_parser(
        "prepare-dataset", help="Run all dataset preparation steps sequentially"
    )
    prepare_parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Vocabulary size for training and tokenization",
    )

    # Add new arguments to prepare-dataset
    prepare_parser.add_argument(
        "--num-files",
        type=int,
        default=1,
        help="Number of files to process (max 50)",
    )
    prepare_parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.1,
        help="Fraction of stories to use from each file",
    )

    args = parser.parse_args()

    if args.command == "download":
        download()
    elif args.command == "train-vocab":
        train_vocab(args.vocab_size)
    elif args.command == "pretokenize":
        pretokenize(args.vocab_size)
    elif args.command == "prepare-dataset":
        prepare_dataset(
            args.vocab_size,
            num_files=args.num_files,
            sample_rate=args.sample_rate
        )
    else:
        parser.print_help()
