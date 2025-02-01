from model import GPT
from config import GPTConfig, TrainingConfig
from functools import partial
import time
import math
import os
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from dataset import Task
from tqdm import tqdm
import psutil
import GPUtil

def check_cuda_availability():
    print("\nCUDA Environment Check:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    else:
        print("No CUDA device available! Troubleshooting steps:")
        print("1. Check if CUDA toolkit is installed")
        print("2. Verify PyTorch CUDA installation")
        print("3. Check GPU drivers")

def check_environment():
    import sys
    import subprocess
    
    print("\nEnvironment Information:")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode()
        print("\nNVIDIA-SMI output:")
        print(nvidia_smi)
    except:
        print("nvidia-smi not available")

# Run environment checks first
check_cuda_availability()
check_environment()

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Clear GPU memory before starting
clear_gpu_cache()

# Set random seed for reproducibility
torch.manual_seed(42)

train_config = TrainingConfig()
out_dir = "out/"
writer = SummaryWriter(log_dir=os.path.join(out_dir, "logs"))
resume = False

# Calculate tokens per iteration
tokens_per_iter = (
    train_config.gradient_accumulation_steps
    * train_config.batch_size
    * GPTConfig.block_size
)
print(f"\nTraining Configuration:")
print(f"Tokens per iteration: {tokens_per_iter}")
print(f"Batch size: {train_config.batch_size}")
print(f"Gradient accumulation steps: {train_config.gradient_accumulation_steps}")
print(f"Effective batch size: {train_config.batch_size * train_config.gradient_accumulation_steps}")

device = train_config.device
print(f"\nUsing device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

ctx = torch.autocast(device, dtype=torch.float16 if device == "cuda" else torch.float32)

model_args = dict(
    n_layer=GPTConfig.n_layer,
    n_head=GPTConfig.n_head,
    n_embed=GPTConfig.n_embed,
    block_size=GPTConfig.block_size,
    bias=GPTConfig.bias,
    vocab_size=GPTConfig.vocab_size,
    dropout=GPTConfig.dropout,
)

iter_batches = partial(
    Task.iter_batches,
    batch_size=train_config.batch_size,
    max_seq_len=GPTConfig.block_size,
    device=device,
    num_workers=0,
)

best_val_loss = 1e9
if resume:
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    best_val_loss = checkpoint["best_val_loss"]

    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    print("Loaded checkpoint")
else:
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(device)

scaler = torch.GradScaler(enabled=(train_config.dtype == "float16"))


train_config.learning_rate = 1e-4  # Lower base learning rate
train_config.min_lr = 1e-5        # Lower minimum learning rate
train_config.warmup_iters = 1000    # Longer warmup for stability
train_config.lr_decay_iters = train_config.max_iters * 0.9  # Decay over 90% of training

optimizer = model.configure_optimizers(
    train_config.weight_decay,
    train_config.learning_rate,
    (train_config.beta1, train_config.beta2),
    device,
)

# Increase batch sizes and optimize GPU memory usage
train_config.batch_size = 64  # Increase from 32
train_config.gradient_accumulation_steps = 4  # Decrease from 8 to maintain similar effective batch size
train_config.eval_iters = 50  # Increase evaluation iterations

# Add CUDA optimization settings
torch.backends.cuda.matmul.allow_tf32 = True  # Already set, but confirm it's there
torch.backends.cudnn.benchmark = True  # Add this for potential speed improvement
torch.backends.cudnn.deterministic = False  # Allow for optimizations

# Optimize memory allocation
def optimize_memory():
    if torch.cuda.is_available():
        # Reserve more GPU memory upfront
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Pre-allocate memory
        torch.cuda.set_per_process_memory_fraction(0.85)  # Use up to 85% of GPU memory
        # Enable memory efficient attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

# Update the model compilation for better performance
if train_config.compile:
    model = torch.compile(model, mode='max-autotune')

# Add this before the training loop to monitor memory usage
print("\nGPU Memory Configuration:")
print(f"Batch size: {train_config.batch_size}")
print(f"Gradient accumulation steps: {train_config.gradient_accumulation_steps}")
print(f"Effective batch size: {train_config.batch_size * train_config.gradient_accumulation_steps}")

params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print("Number of params:", params)

train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)

# Add debugging code before training loop
print("\nInput tensor shapes:")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"Max index in X: {X.max().item()}")
print(f"Block size: {GPTConfig.block_size}")
print(f"Vocab size: {GPTConfig.vocab_size}")

# Add shape checks
if X.max() >= GPTConfig.vocab_size:
    raise ValueError(f"Input contains token indices >= vocab_size ({GPTConfig.vocab_size})")
if X.shape[1] > GPTConfig.block_size:
    raise ValueError(f"Input sequence length {X.shape[1]} exceeds block_size {GPTConfig.block_size}")

# Add this after getting X, Y from iter_batches
def clamp_tokens(tokens, vocab_size):
    return torch.clamp(tokens, 0, vocab_size - 1)

# Clamp token values to be within vocabulary size
X = clamp_tokens(X, GPTConfig.vocab_size)
Y = clamp_tokens(Y, GPTConfig.vocab_size)

# Keep the existing debug prints
print("\nInput tensor shapes:")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"Max index in X: {X.max().item()}")
print(f"Block size: {GPTConfig.block_size}")
print(f"Vocab size: {GPTConfig.vocab_size}")

iter_num = 0
t0 = time.time()

# Move these functions before the training loop starts
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(train_config.eval_iters)
        batch_iter = iter_batches(split=split)
        for k in range(train_config.eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < train_config.warmup_iters:
        return train_config.learning_rate * (it + 1) / (train_config.warmup_iters + 1)
    if it > train_config.lr_decay_iters:
        return train_config.min_lr
    decay_ratio = (it - train_config.warmup_iters) / (
        train_config.lr_decay_iters - train_config.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (
        train_config.learning_rate - train_config.min_lr
    )

# Move these functions before the training loop too
def log_memory_usage():
    gpu = GPUtil.getGPUs()[0]
    gpu_memory = f"GPU Memory: {gpu.memoryUsed:.0f}MB/{gpu.memoryTotal:.0f}MB ({gpu.memoryUtil*100:.1f}%)"
    ram = psutil.Process().memory_info().rss / 1024**2
    print(f"RAM Usage: {ram:.0f}MB, {gpu_memory}")

def monitor_memory():
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print("\nGPU Memory Status:")
        print(f"Allocated: {allocated:.0f}MB")
        print(f"Reserved: {reserved:.0f}MB")
        print(f"Total GPU Memory: {gpu.memoryTotal:.0f}MB")
        print(f"GPU Utilization: {gpu.memoryUtil*100:.1f}%")

# Then start the training loop
print("\nInitial memory status:")
monitor_memory()
optimize_memory()

pbar = tqdm(range(train_config.max_iters), desc="Training")
for iter_num in pbar:
    lr = get_lr(iter_num) if train_config.decay_lr else train_config.learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % train_config.eval_interval == 0:
        losses = estimate_loss()
        log_memory_usage()
        pbar.set_description(
            f"iter {iter_num}: train={losses['train']:.4f}, val={losses['val']:.4f}, lr={lr:.2e}"
        )
        writer.add_scalar("train_loss", losses["train"], iter_num)
        writer.add_scalar("val_loss", losses["val"], iter_num)
        writer.add_scalar("lr", lr, iter_num)

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            checkpoint = {
                "model": model.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
            pbar.write(f"Saved checkpoint at iter {iter_num}")

    for micro_step in range(train_config.gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / train_config.gradient_accumulation_steps
        X, Y = next(train_batch_iter)
        scaler.scale(loss).backward()

    if train_config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % train_config.log_interval == 0:
        lossf = loss.item() * train_config.gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1

    if iter_num > train_config.max_iters:
        break

    # In the training loop, add print for debugging
    if iter_num % 100 == 0:
        print(f"\nLearning rate at iter {iter_num}: {lr:.2e}")

writer.close()
