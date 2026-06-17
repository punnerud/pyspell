"""llama2 architecture, math-identical to the on-device `tinyllm` runtime.

Critical parity points (so the exported v2 checkpoint runs correctly on the dongle
and in the browser-WASM): RMSNorm eps=1e-5, **interleaved-pair RoPE** (run.c style,
freq = 10000^-(i/head_size) over adjacent pairs — NOT the rotate-half convention),
SwiGLU FFN w2(silu(w1 x) * w3 x), GQA via head grouping, attention scale
1/sqrt(head_size), and a tied (shared) classifier = token embedding.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    dim: int = 256
    hidden_dim: int = 768
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 8
    vocab_size: int = 259      # byte-level: <unk>,<s>,</s> + 256 raw bytes
    seq_len: int = 512
    tie_classifier: bool = True  # False -> separate learned wcls
    frozen_emb: bool = False     # True -> install + freeze a precomputed input embedding
    #   frozen_emb + tie  -> output is a RAG lookup against the frozen dict (no wcls)
    #   frozen_emb + untie-> separate learned classifier over the same vocab

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    @property
    def group_size(self) -> int:
        """Q8 group size: largest power-of-two divisor of gcd(dim, hidden_dim), <=64."""
        g = math.gcd(self.dim, self.hidden_dim)
        gs = 1
        while gs * 2 <= g and g % (gs * 2) == 0:
            gs *= 2
        return min(gs, 64)


PRESETS = {
    # ~0.46M params -> ~0.49 MB int8 (under 500 kB) with a ~1024 word/subword vocab.
    # Short sequences (word-level), so a tiny model learns the strict grammar well.
    "full": Config(dim=128, hidden_dim=256, n_layers=2, n_heads=4, n_kv_heads=4,
                   vocab_size=1024, seq_len=128),
    # Curated 512 vocab + frozen semantic+POS input embedding + untied learned
    # classifier. ~0.46M params -> ~479 kB int8 (under 500 kB).
    "full512": Config(dim=128, hidden_dim=256, n_layers=2, n_heads=4, n_kv_heads=4,
                      vocab_size=512, seq_len=128, tie_classifier=False, frozen_emb=True),
    # Same, but the classifier IS the frozen embedding (tied): the last layer is a RAG
    # lookup against the dict — model emits an embedding, nearest word wins. Smallest
    # (no separate wcls), fully "embedding in -> model -> embedding lookup".
    "full512tied": Config(dim=128, hidden_dim=256, n_layers=2, n_heads=4, n_kv_heads=4,
                          vocab_size=512, seq_len=128, tie_classifier=True, frozen_emb=True),
    # Tiny: pipeline smoke test only.
    "smoke": Config(dim=64, hidden_dim=128, n_layers=2, n_heads=4, n_kv_heads=4,
                    vocab_size=512, seq_len=64),
}


def rmsnorm(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def precompute_rope(head_size, seq_len, device):
    # i = 0,2,4,... (adjacent-pair indices); freq = 1/10000^(i/head_size).
    idx = torch.arange(0, head_size, 2, dtype=torch.float32, device=device)
    freqs = 1.0 / (10000.0 ** (idx / head_size))            # [head_size/2]
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    ang = torch.outer(t, freqs)                              # [seq_len, head_size/2]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    # x: [B, T, H, head_size]; rotate adjacent pairs (2p, 2p+1).
    T = x.shape[1]
    cos = cos[:T].view(1, T, 1, -1)
    sin = sin[:T].view(1, T, 1, -1)
    xe = x[..., 0::2]
    xo = x[..., 1::2]
    re = xe * cos - xo * sin
    ro = xe * sin + xo * cos
    return torch.stack((re, ro), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        hd = c.head_size
        self.wq = nn.Linear(c.dim, c.n_heads * hd, bias=False)
        self.wk = nn.Linear(c.dim, c.n_kv_heads * hd, bias=False)
        self.wv = nn.Linear(c.dim, c.n_kv_heads * hd, bias=False)
        self.wo = nn.Linear(c.n_heads * hd, c.dim, bias=False)
        self.w1 = nn.Linear(c.dim, c.hidden_dim, bias=False)
        self.w2 = nn.Linear(c.hidden_dim, c.dim, bias=False)
        self.w3 = nn.Linear(c.dim, c.hidden_dim, bias=False)
        self.att_norm = nn.Parameter(torch.ones(c.dim))
        self.ffn_norm = nn.Parameter(torch.ones(c.dim))

    def forward(self, x, cos, sin):
        c = self.c
        B, T, _ = x.shape
        hd = c.head_size
        h = rmsnorm(x, self.att_norm)
        q = self.wq(h).view(B, T, c.n_heads, hd)
        k = self.wk(h).view(B, T, c.n_kv_heads, hd)
        v = self.wv(h).view(B, T, c.n_kv_heads, hd)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if c.n_kv_heads != c.n_heads:
            rep = c.n_heads // c.n_kv_heads
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        q = q.transpose(1, 2)  # [B, H, T, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # scale = 1/sqrt(hd)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.wo(y)
        h = rmsnorm(x, self.ffn_norm)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x


class Llama(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.dim)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.final_norm = nn.Parameter(torch.ones(c.dim))
        self.lm_head = nn.Linear(c.dim, c.vocab_size, bias=False)
        self._rope = {}
        self.apply(self._init)
        if c.tie_classifier:
            self.lm_head.weight = self.tok.weight  # shared classifier
        # else: separate learned classifier (kept from _init above)

    def set_frozen_embedding(self, emb):
        """Install a precomputed [vocab, dim] input embedding and freeze it.
        Use with tie_classifier=False so the (untied) classifier still learns."""
        assert emb.shape == (self.c.vocab_size, self.c.dim), emb.shape
        with torch.no_grad():
            self.tok.weight.copy_(emb)
        self.tok.weight.requires_grad_(False)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope_tables(self, device):
        key = str(device)
        if key not in self._rope:
            self._rope[key] = precompute_rope(self.c.head_size, self.c.seq_len, device)
        return self._rope[key]

    def forward(self, idx, targets=None):
        x = self.tok(idx)
        cos, sin = self._rope_tables(idx.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = rmsnorm(x, self.final_norm)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def num_params(self):
        n = sum(p.numel() for p in self.parameters())
        if self.lm_head.weight.data_ptr() == self.tok.weight.data_ptr():
            n -= self.lm_head.weight.numel()  # tied head shares tok.weight; count once
        return n
