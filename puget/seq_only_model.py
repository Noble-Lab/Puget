from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ATTN_IMPLS = ("sdpa", "flex", "math")
INPUT_NORMS = ("layernorm", "none")
OUTPUT_ACTIVATIONS = ("linear", "softplus")

# Preference order, not a hard requirement: flash is used when the tensor layout
# and dtype allow it, and the efficient/math kernels catch fp32 and biased calls.
_SDPA_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]

_FLEX_FN = None


def _flex_fn():
    global _FLEX_FN
    if _FLEX_FN is None:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX_FN = torch.compile(flex_attention, dynamic=False)
    return _FLEX_FN


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    assert embed_dim % 2 == 0, "embed_dim must be even"
    half = embed_dim // 2
    pos = np.arange(length, dtype=np.float32)[:, None]
    dims = np.arange(half, dtype=np.float32)[None, :]
    rates = 1.0 / (10000 ** (2 * dims / embed_dim))
    ang = pos * rates
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


class SeqAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.2, attn_impl: str = "sdpa"):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        if attn_impl not in ATTN_IMPLS:
            raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}, got {attn_impl!r}")
        self.h = int(heads)
        self.dk = dim // int(heads)
        self.scale = 1.0 / math.sqrt(self.dk)
        self.attn_impl = str(attn_impl)

        self.attn_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.ffn_norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.drop = nn.Dropout(dropout)

    def _attend(self, q, k, v, bias: Optional[torch.Tensor]):
        if self.attn_impl == "flex":
            if bias is None:
                return _flex_fn()(q, k, v, scale=self.scale)

            def score_mod(score, b, h, q_idx, kv_idx):
                return score + bias[b, h, q_idx, kv_idx]

            return _flex_fn()(q, k, v, score_mod=score_mod, scale=self.scale)

        if self.attn_impl == "sdpa":
            # attn_mask=None is what lets the flash kernel be selected; with a
            # bias SDPA falls back to the memory-efficient kernel, which still
            # beats materializing softmax over an explicit (B, h, N, N) tensor.
            with sdpa_kernel(_SDPA_BACKENDS):
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=bias, dropout_p=0.0, scale=self.scale)

        logits = (q @ k.transpose(-2, -1)) * self.scale
        if bias is not None:
            logits = logits + bias
        return logits.softmax(dim=-1) @ v

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        h = self.attn_norm(x)
        q = self.q(h).view(B, N, self.h, self.dk).transpose(1, 2)
        k = self.k(h).view(B, N, self.h, self.dk).transpose(1, 2)
        v = self.v(h).view(B, N, self.h, self.dk).transpose(1, 2)

        ctx = self._attend(q, k, v, bias)
        ctx = ctx.transpose(1, 2).reshape(B, N, D)

        x = x + self.drop(self.o(ctx))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SeqBaselineTransformer(nn.Module):
    def __init__(
        self,
        *,
        seq_input_dim: int,
        n_tokens: int,
        output_dim: int = 14,
        proj_dim: int = 512,
        heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout_p: float = 0.2,
        hidden: int = 512,
        promoter_tokens: int = 4,
        input_norm: str = "layernorm",
        attn_impl: str = "sdpa",
        output_activation: str = "softplus",
    ):
        super().__init__()
        if input_norm not in INPUT_NORMS:
            raise ValueError(f"input_norm must be one of {INPUT_NORMS}, got {input_norm!r}")
        if promoter_tokens > n_tokens:
            raise ValueError(f"promoter_tokens {promoter_tokens} exceeds n_tokens {n_tokens}")
        if output_activation not in OUTPUT_ACTIVATIONS:
            raise ValueError(
                f"output_activation must be one of {OUTPUT_ACTIVATIONS}, "
                f"got {output_activation!r}"
            )
        self.N = int(n_tokens)
        self.promoter_tokens = int(promoter_tokens)
        self.p0 = (self.N - self.promoter_tokens) // 2
        self.attn_impl = str(attn_impl)
        self.input_norm_kind = str(input_norm)
        self.output_activation = str(output_activation)

        self.input_norm = (nn.LayerNorm(int(seq_input_dim))
                           if input_norm == "layernorm" else nn.Identity())
        self.seq_mlp = nn.Sequential(
            nn.Linear(int(seq_input_dim), hidden), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(hidden, proj_dim))
        pe = get_1d_sincos_pos_embed(proj_dim, self.N)
        self.register_buffer("pos_embed", torch.from_numpy(pe).float(), persistent=True)

        self.blocks = nn.ModuleList([
            SeqAttentionBlock(proj_dim, int(heads), mlp_ratio, dropout_p, attn_impl)
            for _ in range(int(num_layers))])

        self.out_norm = nn.LayerNorm(proj_dim)
        self.head = nn.Linear(proj_dim, int(output_dim))

    def forward(self, seq: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if seq.shape[1] != self.N:
            raise ValueError(f"expected {self.N} tokens, got {seq.shape[1]}")
        # fp32 before the input LayerNorm: Enformer post_transformer reaches
        # |x| ~ 2.4e4, whose square overflows fp16 in the variance reduction.
        x = self.input_norm(seq.float())
        x = self.seq_mlp(x) + self.pos_embed.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, bias)
        pooled = self.out_norm(x[:, self.p0:self.p0 + self.promoter_tokens, :].mean(dim=1))
        output = self.head(pooled)
        if self.output_activation == "softplus":
            output = F.softplus(output)
        return output


def build_model(*, seq_input_dim, n_tokens, output_dim=14, proj_dim=512, num_heads=8,
                num_layers=4, mlp_ratio=4.0, dropout_p=0.2, promoter_tokens=4,
                input_norm="layernorm", attn_impl="sdpa",
                output_activation="softplus") -> SeqBaselineTransformer:
    return SeqBaselineTransformer(
        seq_input_dim=int(seq_input_dim), n_tokens=int(n_tokens),
        output_dim=int(output_dim), proj_dim=int(proj_dim), heads=int(num_heads),
        num_layers=int(num_layers), mlp_ratio=float(mlp_ratio),
        dropout_p=float(dropout_p), promoter_tokens=int(promoter_tokens),
        input_norm=str(input_norm), attn_impl=str(attn_impl),
        output_activation=str(output_activation))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------- selftest
def _selftest():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  torch={torch.__version__}")

    for D_in, N, tag in ((1536, 512, "borzoi/alphagenome 524k"),
                         (3072, 192, "enformer 196k post_pointwise")):
        m = build_model(seq_input_dim=D_in, n_tokens=N)
        print(f"  D_in={D_in:5d} N={N:4d}  params={count_parameters(m)/1e6:7.3f} M   ({tag})")
    print("  reference: Puget pointwise bias 524k = 13.671 M, seq-only 14-head uses the same trunk")

    # The control head remains unconstrained; Softplus must be elementwise and
    # nonnegative without changing parameter shapes or counts.
    linear = build_model(seq_input_dim=12, n_tokens=8, output_dim=3,
                         proj_dim=16, num_heads=2, num_layers=1,
                         output_activation="linear").eval()
    softplus = build_model(seq_input_dim=12, n_tokens=8, output_dim=3,
                           proj_dim=16, num_heads=2, num_layers=1,
                           output_activation="softplus").eval()
    softplus.load_state_dict(linear.state_dict())
    test_x = torch.randn(2, 8, 12)
    with torch.no_grad():
        raw = linear(test_x)
        constrained = softplus(test_x)
    assert torch.equal(constrained, F.softplus(raw))
    assert bool((constrained >= 0).all())
    print("  output activation parity/nonnegativity: ok")

    # Backend parity, fp32 on identical weights and identical inputs.
    base = build_model(seq_input_dim=256, n_tokens=192, proj_dim=128, num_heads=4,
                       num_layers=2).to(dev).eval()
    x = torch.randn(4, 192, 256, device=dev)
    bias = torch.randn(4, 4, 192, 192, device=dev) * 0.1
    outs = {}
    for impl in ATTN_IMPLS:
        for blk in base.blocks:
            blk.attn_impl = impl
        with torch.no_grad():
            outs[impl] = {"nobias": base(x).clone()}
            if dev == "cuda" or impl != "flex":
                outs[impl]["bias"] = base(x, bias).clone()
    for impl in ("sdpa", "flex"):
        for key in outs[impl]:
            d = (outs[impl][key] - outs["math"][key]).abs().max().item()
            status = "ok" if d < 2e-4 else "MISMATCH"
            print(f"  parity {impl:5s} vs math ({key:6s}): max|diff| = {d:.2e}  {status}")

    if dev == "cuda":
        m = build_model(seq_input_dim=1536, n_tokens=512).cuda()
        xb = torch.randn(32, 512, 1536, device="cuda", dtype=torch.float16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = m(xb)
            y.float().pow(2).mean().backward()
        print(f"  bf16 flash fwd+bwd ok: out={tuple(y.shape)} dtype={y.dtype}")


if __name__ == "__main__":
    _selftest()
