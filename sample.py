import os
import torch
import argparse
from tokenizer import Tokenizer
from model import GPT
from config import GPTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="GPT Inference Script")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=False,
        help="Full path to the checkpoint file",
        default="out/ckpt.pt"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=os.path.join("data", "tok4096.model"),
        help="Path to tokenizer model",
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Input prompt for generation"
    )
    parser.add_argument(
        "--num_samples", type=int, default=3, help="Number of samples to generate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top_p", type=float, default=None, help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--min_p", type=float, default=0.05, help="Minimum probability for sampling"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for inference",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model"
    )
    return parser.parse_args()


def setup_device(args):
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    ctx = torch.autocast(args.device, dtype=dtype)
    return ctx


def load_model(args):
    checkpoint = torch.load(args.ckpt_path, map_location=args.device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(args.device)
    if args.compile:
        model = torch.compile(model)
    return model


def main():
    args = parse_args()
    ctx = setup_device(args)
    model = load_model(args)

    enc = Tokenizer(args.tokenizer_path)
    encode = lambda s: enc.encode(s, bos=True, eos=False)
    decode = lambda l: enc.decode(l)

    x = torch.tensor(
        encode(args.prompt), dtype=torch.long, device=args.device
    ).unsqueeze(0)

    with torch.no_grad():
        with ctx:
            for k in range(args.num_samples):
                y = model.generate(
                    x,
                    args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    min_p=args.min_p,
                )
                print(decode(y[0].tolist()))
                print("------------------")


if __name__ == "__main__":
    main()