from model import GPTConfig, GPT
import time
import math
import pickle 
import os
import numpy as np
import torch

out_dir = 'out'
eval_interval = 250 
log_interval = 10
eval_iters = 200

gradient_accumulation_steps = 5 * 8
batch_size = 64 
block_size = 1024
n_layer = 6
n_head = 6
n_embed = 384
dropout = 0.2
bias = False

learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 100
lr_decay_iters = max_iters

min_lr = learning_rate / 10

device = 'cuda'
dtype = 'bfloat16'
compile = False 

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size

print("Tokens per iteration: ", tokens_per_iter)

torch.manual_seed(42)

device_type = 'cuda'

ctx = torch.autocast("cuda")

data_dir = 'data/shakespeare/'

def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1: i+1+block_size]).astype(np.int64)) for i in ix])

    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

    return x, y

iter_num = 0
best_val_loss = 1e9


meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = 50304
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
        meta_vocab_size = meta['meta_vocab_size']

model_args = dict(n_layer=n_layer, n_head=n_head, n_embed=n_embed, block_size=block_size, bias=bias, vocab_size=meta_vocab_size, dropout=dropout)

gptconf = GPTConfig(**model_args)
model = GPT(gptconf).to(device)

scaler = torch.GradScaler(enabled=(dtype=='float16'))

optimizer = torch.optim.AdamW(params=model.parameters(), betas=(beta1,beta2))

if compile:
    model = torch.compile(model)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate  * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (learning_rate - min_lr)


X, Y = get_batch('train')

t0 = time.time()
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}")
    
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
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

            iter_num+=1

        if iter_num > max_iters:
            break
