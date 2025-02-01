import os
import torch
from model import GPT
from config import GPTConfig
from tokenizer import Tokenizer

OUT_DIR = "out"
CHECKPOINT_PATH = os.path.join(OUT_DIR, "ckpt.pt")
# Use vocab size from config to construct tokenizer path
TOKENIZER_PATH = os.path.join("data", f"tok{GPTConfig.vocab_size}.model")

def check_files():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}. Make sure training has saved at least one checkpoint.")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"No tokenizer found at {TOKENIZER_PATH}. Run preprocess.py first to generate the tokenizer.")

def load_latest_model():
    check_files()  # Add this check
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cuda')
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    # Handle unwanted prefix in state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to('cuda')
    model.eval()
    return model

def chat():
    check_files()  # Add this check
    # Load tokenizer and model
    tokenizer = Tokenizer(TOKENIZER_PATH)
    model = load_latest_model()
    print("\nModel loaded! Chat with the AI (type 'quit' to exit)")
    
    def clamp_tokens(tokens, vocab_size):
        return torch.clamp(tokens, 0, vocab_size - 1)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break
                
            # Encode input and clamp tokens
            tokens = tokenizer.encode(user_input, bos=True, eos=False)
            tokens = torch.tensor(tokens).unsqueeze(0).to('cuda')
            tokens = clamp_tokens(tokens, model.config.vocab_size)
            
            # Generate response with safer parameters
            with torch.no_grad():
                output_tokens = model.generate(
                    tokens, 
                    max_new_tokens=32,        # Shorter responses to avoid spiraling
                    temperature=0.6,          # More conservative
                    top_k=20,                # More focused selection
                    top_p=0.9,
                    min_p=0.1                # Higher min_p to avoid rare/nonsense tokens
                )[0].tolist()
            
            # Decode and print response
            response = tokenizer.decode(output_tokens)
            print("\nAI:", response)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    chat() 