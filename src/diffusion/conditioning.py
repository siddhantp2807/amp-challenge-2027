"""Stage-2 conditioning: the property vector, its masking, and its embedding.

Follows OmegAMP (paper/7531_OmegAMP_Targeted_AMP_Disc.pdf, Sec. 3.1):
`cond(s) = (1_AMP(s), |s|, Charge(s), Hydroph.(s))`, with a binary mask sampled
per training example so the model learns every subset of the properties at once.
Two deliberate departures:

  - `1_AMP` becomes `is_target`: every sequence here is already an AMP, so the
    always-on bit is repurposed to mark the target set against the broader
    pretraining corpus. Like `1_AMP` it is never masked.
  - hydrophobicity is GRAVY, not hydrophobic moment. Both were probed against
    the frozen latent; GRAVY is recoverable from `z` at R^2 0.930 and hmoment at
    only 0.639 (ridge 0.277), so hmoment carries an information floor the
    diffusion model could not have overcome. It stays a reported diagnostic.

Property values come from data/ld-processed/, and src/eval/properties.py
reproduces them exactly (tests/test_ld_properties.py) so generated sequences can
be scored on the same definitions the labels use.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.eval.properties import charge_bjellqvist, gravy

# Order is load-bearing: it fixes the column order of the condition matrix, the
# bit order of the mask, and therefore the meaning of a trained checkpoint's
# conditioning weights. Appending is safe; reordering invalidates checkpoints.
PROPERTIES = ("length", "charge", "gravy")

# Recomputed from a generated sequence at eval time. Must match the label columns.
PROPERTY_FNS = {
    "length": lambda s: float(len(s)),
    "charge": charge_bjellqvist,
    "gravy": gravy,
}

LD_DIR = Path("data/ld-processed")


def compute_properties(sequences: list[str]) -> np.ndarray:
    """(N, len(PROPERTIES)) property matrix for generated (unlabelled) sequences."""
    return np.array(
        [[PROPERTY_FNS[p](s) for p in PROPERTIES] for s in sequences], dtype=np.float64
    )


def load_conditions(which: str, split_csv: str | None = None) -> pd.DataFrame:
    """Load sequences + property labels, optionally joined to a split column.

    `which` is "finetuning" or "pretraining". data/ld-processed/finetuning.csv has
    no split column of its own, so the target splits are joined on `sequence` from
    the cleaned corpus (verified exact set equality, 6690/372/371).
    """
    df = pd.read_csv(LD_DIR / f"{which}.csv")
    missing = [p for p in PROPERTIES if p not in df.columns]
    if missing:
        raise ValueError(f"{which}.csv is missing conditioning columns {missing}")

    if split_csv is not None:
        splits = pd.read_csv(split_csv)[["sequence", "split"]]
        n_before = len(df)
        df = df.merge(splits, on="sequence", how="inner")
        if len(df) != n_before:
            raise ValueError(
                f"{which}.csv <-> {split_csv} join dropped {n_before - len(df)} rows; "
                "the label file and the cleaned corpus have diverged."
            )
    return df


class PropertyStandardizer:
    """Per-property mean/std, fitted on the training split only.

    Stored in the diffusion checkpoint so sampling standardizes user-specified
    targets exactly the way training did.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    @classmethod
    def fit(cls, c: np.ndarray) -> "PropertyStandardizer":
        return cls(c.mean(axis=0), c.std(axis=0).clip(min=1e-8))

    def transform(self, c: np.ndarray) -> np.ndarray:
        return (np.asarray(c, dtype=np.float64) - self.mean) / self.std

    def inverse(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(c, dtype=np.float64) * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "properties": list(PROPERTIES)}

    @classmethod
    def from_state_dict(cls, d: dict) -> "PropertyStandardizer":
        if tuple(d["properties"]) != PROPERTIES:
            raise ValueError(
                f"checkpoint was trained on {d['properties']}, but this build uses "
                f"{list(PROPERTIES)} -- the conditioning weights are not transferable."
            )
        return cls(np.array(d["mean"]), np.array(d["std"]))


def sample_masks(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """(n, K) bool mask, True = property is ACTIVE. The paper's D_mask exactly.

    Draw k ~ Uniform{0..K}, then keep a uniformly random size-k subset. Two
    properties of this scheme matter downstream:

      - P(all masked) = 1/(K+1) = 0.25, so a quarter of training steps train the
        unconditional model. Classifier-free guidance is therefore free and needs
        no separate p_uncond knob.
      - P(a *specific* size-j subset is all-active) = 1/(j+1), independent of K.
        So adding a fourth property would not reduce how often the three we care
        about are jointly trained. Verified by simulation; see the plan.
    """
    k_dim = len(PROPERTIES)
    k = torch.randint(0, k_dim + 1, (n,), generator=generator)
    # random permutation per row; keep the first k columns of each
    order = torch.argsort(torch.rand(n, k_dim, generator=generator), dim=1)
    ranks = torch.argsort(order, dim=1)
    return ranks < k.unsqueeze(1)


class ConditionEmbedder(nn.Module):
    """(condition values, mask, is_target) -> a single d_cond vector.

    Length gets an embedding table rather than Fourier features: it is discrete
    with 43 possible values and is the property the VAE encodes most sharply
    (length_exact_rate 1.0 from z), so exact lookup beats a smooth function of a
    scalar. Charge and GRAVY are continuous and must be matched to ~0.1 std at
    sampling time, which a raw scalar fed to a Linear resolves poorly -- hence
    random Fourier features.

    Each maskable property has its own learned mask embedding, substituted for
    the value embedding. The mask *pattern* is additionally embedded, because the
    sum of per-property embeddings alone is ambiguous about which combination
    produced it.
    """

    def __init__(self, length_mean: float, length_std: float, d_cond: int = 256,
                 n_fourier: int = 16, min_len: int = 8, max_len: int = 50,
                 fourier_scale: float = 1.0):
        super().__init__()
        self.d_cond = d_cond
        self.min_len, self.max_len = min_len, max_len
        # needed to turn the standardized length column back into a table index;
        # constructor args (not a later setter) so forward can never see them unset
        self.register_buffer("_len_mean", torch.tensor(float(length_mean)))
        self.register_buffer("_len_std", torch.tensor(float(length_std)))
        self.properties = PROPERTIES
        self.length_idx = PROPERTIES.index("length")

        self.length_embed = nn.Embedding(max_len - min_len + 1, d_cond)
        self.is_target_embed = nn.Embedding(2, d_cond)
        self.mask_pattern_embed = nn.Embedding(2 ** len(PROPERTIES), d_cond)

        # continuous branches, one per non-length property
        self.continuous = [p for p in PROPERTIES if p != "length"]
        self.register_buffer(
            "fourier_freqs",
            torch.randn(len(self.continuous), n_fourier) * fourier_scale,
        )
        self.continuous_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * n_fourier, d_cond), nn.SiLU(),
                          nn.Linear(d_cond, d_cond))
            for _ in self.continuous
        ])
        self.mask_embed = nn.Parameter(torch.randn(len(PROPERTIES), d_cond) * 0.02)
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(d_cond, d_cond))

    def forward(self, c: torch.Tensor, mask: torch.Tensor,
                is_target: torch.Tensor) -> torch.Tensor:
        """c: (B, K) standardized values. mask: (B, K) bool, True = active.
        is_target: (B,) long. Returns (B, d_cond)."""
        b = c.shape[0]
        h = self.is_target_embed(is_target)

        # length arrives standardized like every other column; recover the integer
        # residue count for the table lookup, clamped so an out-of-range request
        # cannot index past the table
        raw_len = (c[:, self.length_idx] * self._len_std + self._len_mean).round()
        len_idx = (raw_len - self.min_len).clamp(0, self.max_len - self.min_len).long()
        len_emb = self.length_embed(len_idx)
        m = mask[:, self.length_idx].unsqueeze(-1)
        h = h + torch.where(m, len_emb, self.mask_embed[self.length_idx].expand(b, -1))

        for j, name in enumerate(self.continuous):
            idx = PROPERTIES.index(name)
            v = c[:, idx].unsqueeze(-1) * self.fourier_freqs[j].unsqueeze(0)
            feats = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
            emb = self.continuous_mlp[j](feats)
            m = mask[:, idx].unsqueeze(-1)
            h = h + torch.where(m, emb, self.mask_embed[idx].expand(b, -1))

        weights = (2 ** torch.arange(len(PROPERTIES), device=mask.device))
        h = h + self.mask_pattern_embed((mask.long() * weights).sum(-1))
        return self.out(h)
