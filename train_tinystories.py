from model import GPTConfig, GPT
from functools import partial
import time
import math
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from prepare_tinystories import Task

out_dir = "out/v2"
writer = SummaryWriter(log_dir=os.path.join(out_dir, "logs"))
eval_interval = 100
log_interval = 10
eval_iters = 200

gradient_accumulation_steps = 4
batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
n_embed = 288 
dropout = 0.2
bias = False

learning_rate = 6e-4
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 100
lr_decay_iters = max_iters

min_lr = learning_rate / 10

device = "cuda"
dtype = "bfloat16"
compile = True
resume = True 

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size

print("Tokens per iteration: ", tokens_per_iter)

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.set_float32_matmul_precision('high')

device_type = "cuda"

ctx = torch.autocast("cuda", dtype=torch.bfloat16)

data_dir = "data/shakespeare/"
meta_vocab_size = 50304

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embed=n_embed,
    block_size=block_size,
    bias=bias,
    vocab_size=meta_vocab_size,
    dropout=dropout,
)

iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=block_size,
    device=device,
    num_workers=0
)

best_val_loss = 1e9
if resume:
    ckpt_path = os.path.join(out_dir, "ckpt-new.pt")
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

scaler = torch.GradScaler(enabled=(dtype == "float16"))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    model = torch.compile(model)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        batch_iter = iter_batches(split=split)
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (learning_rate - min_lr)

params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print("Number of params:", params)

train_batch_iter = iter_batches(split="train")


X, Y = next(train_batch_iter)

iter_num = 0
t0 = time.time()
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}"
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
            print(f"Saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = next(train_batch_iter)
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps

        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1

    if iter_num > max_iters:
        break

writer.close()
