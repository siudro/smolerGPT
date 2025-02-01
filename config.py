from dataclasses import dataclass
import torch
  

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 4096
    n_layer: int = 6
    n_head: int = 8
    n_embed: int = 384
    dropout: float = 0.2
    bias: bool = False


@dataclass
class TrainingConfig:
    learning_rate: float = 6e-4
    max_iters: int = 30000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    decay_lr: bool = True
    warmup_iters: int = 1000
    lr_decay_iters: int = 30000
    min_lr: float = 6e-5

    eval_interval: int = 50
    log_interval: int = 1
    eval_iters: int = 100
    gradient_accumulation_steps: int = 8
    batch_size: int = 32

    @property
    def device(self) -> str:
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, using CPU")
            return "cpu"
        return "cuda"
    
    @property
    def dtype(self) -> torch.dtype:
        if self.device == "cpu":
            return torch.float32  # Use float32 for CPU
        return torch.float16     # Use float16 for GPU

    compile: bool = False
