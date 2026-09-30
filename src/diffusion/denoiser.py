"""The denoising network: a residual MLP with adaLN-Zero conditioning.

Why not OmegAMP's 1D TransUNet. The paper diffuses an (M x K) residue-level
embedding, where axis M is sequence position -- so convolution and downsampling
encode a real inductive bias (local motifs, helical periodicity). Here the
diffusion target is a single 64-d attention-pooled vector whose coordinates are
the output of `to_mu: Linear(192, 64)`: an arbitrary basis with no ordering,
locality or translation structure. A 1D conv over those channels would assert
that dim 17 is "near" dim 18, and downsampling 64 -> 32 -> 16 would destroy
capacity for no inductive benefit. Attention is out too: it needs >= 2 tokens and
there is exactly one.

adaLN-Zero is the faithful analogue of the paper's "inject conditioning at every
layer" (Rombach-style concatenation into every feature map): a per-block affine
(shift, scale, gate) computed from (t, c) does the same job, multiplicatively
rather than by concatenation. The -Zero part matters more here than anywhere --
each block's modulation projection is zero-initialized, so the network starts as
an exact identity map. With 6,690 training examples that is the difference
between converging and memorizing.

adaLN-single (PixArt): one shared modulation projection plus a small per-block
offset table, rather than an independent Linear(d -> 3d) per block. Saves ~1.0M
parameters (~35% of the model) at no documented cost, which matters at ~300
parameters per training example.
"""
import torch
import torch.nn as nn

from src.diffusion.conditioning import PROPERTIES, ConditionEmbedder
from src.diffusion.schedule import timestep_embedding


def modulate(h: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return h * (1 + scale) + shift


class ResBlock(nn.Module):
    def __init__(self, d_model: int, ffn_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, h, shift, scale, gate):
        return h + gate * self.ff(modulate(self.norm(h), shift, scale))


class LatentDenoiser(nn.Module):
    def __init__(self, d_z: int = 64, d_model: int = 256, n_blocks: int = 6,
                 ffn_mult: int = 2, dropout: float = 0.1, d_cond: int = 256,
                 n_fourier: int = 16, length_mean: float = 21.75,
                 length_std: float = 9.73, min_len: int = 8, max_len: int = 50,
                 self_cond: bool = False):
        super().__init__()
        self.d_z, self.d_model, self.n_blocks = d_z, d_model, n_blocks
        self.self_cond = self_cond

        self.in_proj = nn.Linear(d_z, d_model)
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.cond_embed = ConditionEmbedder(
            length_mean=length_mean, length_std=length_std, d_cond=d_cond,
            n_fourier=n_fourier, min_len=min_len, max_len=max_len,
        )
        self.cond_proj = nn.Linear(d_cond, d_model) if d_cond != d_model else nn.Identity()

        # self-conditioning: a zero-init projection of the previous x0 estimate, so
        # enabling it starts as an exact no-op relative to the trained model
        if self_cond:
            self.sc_proj = nn.Linear(d_z, d_model)
            nn.init.zeros_(self.sc_proj.weight)
            nn.init.zeros_(self.sc_proj.bias)

        self.blocks = nn.ModuleList(
            [ResBlock(d_model, ffn_mult, dropout) for _ in range(n_blocks)]
        )
        # adaLN-single: shared projection + per-block learned offsets
        self.ada_shared = nn.Linear(d_model, 3 * d_model)
        self.ada_offset = nn.Parameter(torch.zeros(n_blocks, 3 * d_model))
        nn.init.zeros_(self.ada_shared.weight)
        nn.init.zeros_(self.ada_shared.bias)

        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.out_ada = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.out_ada.weight)
        nn.init.zeros_(self.out_ada.bias)
        self.out_proj = nn.Linear(d_model, d_z)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor,
                mask: torch.Tensor, is_target: torch.Tensor,
                x0_prev: torch.Tensor | None = None) -> torch.Tensor:
        h = self.in_proj(x_t)
        if self.self_cond:
            prev = x0_prev if x0_prev is not None else torch.zeros_like(x_t)
            h = h + self.sc_proj(prev)

        cvec = self.t_mlp(timestep_embedding(t, self.d_model))
        cvec = cvec + self.cond_proj(self.cond_embed(c, mask, is_target))

        mods = self.ada_shared(nn.functional.silu(cvec))          # (B, 3d)
        for i, block in enumerate(self.blocks):
            shift, scale, gate = (mods + self.ada_offset[i]).chunk(3, dim=-1)
            h = block(h, shift, scale, gate)

        shift, scale = self.out_ada(nn.functional.silu(cvec)).chunk(2, dim=-1)
        return self.out_proj(modulate(self.out_norm(h), shift, scale))


def build_denoiser(cfg: dict, length_mean: float, length_std: float) -> LatentDenoiser:
    return LatentDenoiser(
        d_z=cfg["d_z"],
        d_model=cfg["d_model"],
        n_blocks=cfg["n_blocks"],
        ffn_mult=cfg.get("ffn_mult", 2),
        dropout=cfg.get("dropout", 0.1),
        d_cond=cfg.get("d_cond", cfg["d_model"]),
        n_fourier=cfg.get("n_fourier", 16),
        length_mean=length_mean,
        length_std=length_std,
        min_len=cfg.get("min_len", 8),
        max_len=cfg.get("max_len", 50),
        self_cond=cfg.get("self_cond", False),
    )


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
