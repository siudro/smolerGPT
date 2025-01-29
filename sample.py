import os
import torch
from tokenizer import Tokenizer
from model import GPT
from config import GPTConfig

ckpt_dir = "out/"
tokenizer_path = "tok4096.model"
ckpt_path = os.path.join(ckpt_dir, "ckpt-v1.pt")

prompt = 'The direction from which the sun rises is'

num_samples = 3
max_new_tokens = 500
temperature = 0.5 
top_k = None 
top_p = None 
min_p = 0.05 
seed = 42

device = "cuda"
dtype = torch.bfloat16
compile = False

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda"
ctx = torch.autocast(device, dtype=torch.bfloat16)

checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."

for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()

model.to(device)

enc = Tokenizer(tokenizer_path) 
encode = lambda s: enc.encode(s, bos=True, eos=False)
decode = lambda l: enc.decode(l)
x = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, min_p=min_p, top_p=top_p)
            print(decode(y[0].tolist()))
            print("------------------")
