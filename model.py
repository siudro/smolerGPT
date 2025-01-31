import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

from config import GPTConfig


class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10_000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, q, k):
        seq_len = q.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, device=q.device) / self.dim)
            )
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().type_as(q)
            self.sin_cached = freqs.sin().type_as(q)
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        q_ = self.apply_rotary_emb(q, cos, sin)
        k_ = self.apply_rotary_emb(k, cos, sin)
        return q_, k_

    def apply_rotary_emb(self, x, cos, sin):
        assert x.ndim == 4  # multihead attention
        d = x.shape[3] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        assert config.n_embed % config.n_head == 0
        self.head_dim = config.n_embed // config.n_head
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if not self.flash:
            print("Not using flash attention")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

        if config.use_rotary:
            self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.config.n_embed, dim=2)
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)

        # Apply rotary embeddings if enabled
        if self.config.use_rotary:
            q, k = self.rotary(q, k)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            attn_pattern = (q @ k.transpose(-2, -1)) * (
                1.0 / math.sqrt(k.shape[-1])
            )  # B, nh, T, T
            attn_pattern = attn_pattern.masked_fill(
                self.bias[:, :, :T, :T] == 0, float("-inf")
            )
            attn = F.softmax(attn_pattern, dim=-1)
            y = attn @ v  # B, nh, T, T @ B, nh, T, hs -> B, nh, T, hs

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embed
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(config.n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embed, bias=False)
        self.w3 = nn.Linear(config.n_embed, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embed)
        self.ffd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffd(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create base transformer components
        transformer_dict = {
            "wte": nn.Embedding(config.vocab_size, config.n_embed),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.RMSNorm(config.n_embed),
        }

        # Only add positional embeddings if not using rotary
        if not config.use_rotary:
            transformer_dict["wpe"] = nn.Embedding(config.block_size, config.n_embed)

        self.transformer = nn.ModuleDict(transformer_dict)

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.transformer.wte(idx)

        # Add learnable positional embeddings
        if not self.config.use_rotary:
            device = idx.device
            b, t = idx.shape
            pos_emb = self.transformer.wpe(
                torch.arange(0, t, dtype=torch.long, device=device)
            )
            x = x + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, min_p=None
    ):
        for _ in range(max_new_tokens):
            context = (
                idx
                if idx.size(1) < self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(context)

            logits = logits[:, -1, :] / temperature

            if top_p is not None and top_p > 0.0:
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                mask = cumulative_probs >= top_p
                mask[..., 0] = True

                cutoff_indices = mask.int().argmax(dim=-1, keepdim=True)

                top_p_mask = torch.zeros_like(logits, dtype=torch.bool)
                for b in range(logits.size(0)):
                    cut = cutoff_indices[b].item()
                    kept_indices = sorted_indices[b, : cut + 1]
                    top_p_mask[b, kept_indices] = True
                logits[~top_p_mask] = float("-inf")

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if min_p is not None and min_p > 0.0:
                logit_max = logits.max(dim=-1, keepdim=True).values
                threshold = logit_max + torch.log(
                    torch.tensor(min_p, device=logits.device, dtype=logits.dtype)
                )
                logits[logits < threshold] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == 2:
                break
            idx = torch.cat([idx, idx_next], dim=-1)

        return idx
