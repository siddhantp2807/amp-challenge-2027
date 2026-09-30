"""The potency scorer: a shared trunk with one Tobit head per species.

One architecture for every rung of the feature ladder -- only `d_in` changes.

Width 192, not 256: there are 2,099 training sequences over 495 clusters, and the
trunk is the only thing standing between that and memorization.

TRAIN ON CPU. `torch.special.log_ndtr` -- the numerically stable censored
likelihood primitive -- has patchy MPS coverage, and a 2.1k x ~100 problem at this
width is faster on CPU anyway. MPS is used only for the one-shot VAE encode.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.special import log_ndtr

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def tobit_nll(mu, log_s, y, mask, censor, head_weight, s_min=0.05, s_max=3.0):
    """Masked Tobit negative log-likelihood.

    `censor` is 0 exact / -1 left / +1 right and is ALREADY IN pMIC SPACE -- the
    MIC->pMIC flip happens once, in the label build. A left-censored pMIC (from a
    right-censored MIC, i.e. "no activity up to X") says the true value lies at or
    below the bound, so the model is free to predict anything lower and is
    penalized only for predicting higher. That is what keeps the inactive tail
    informative instead of either discarded or mistaken for measurements.

    `s` is clamped rather than left free: with ~240 cells a head will happily drive
    s -> 0 on the rows it memorizes, which makes the loss look excellent and the
    downstream confidence bound meaningless. s_min=0.05 pMIC sits far below the
    ~0.8 label sd, so the clamp binds only pathologically.
    """
    s = torch.clamp(log_s.exp(), s_min, s_max)
    z = (y - mu) / s

    ll_exact = -0.5 * z * z - torch.log(s) - LOG_SQRT_2PI
    # log_ndtr, never log(1 - Phi(z)): the naive form underflows to -inf around
    # |z| ~ 6 and NaNs the whole batch.
    ll = torch.where(
        censor < 0, log_ndtr(z),
        torch.where(censor > 0, log_ndtr(-z), ll_exact),
    )

    w = mask.float() * head_weight.unsqueeze(0)
    return -(w * ll).sum() / w.sum().clamp_min(1.0)


class PotencyScorer(nn.Module):
    def __init__(self, d_in, n_heads, width=192, dropout=0.3):
        super().__init__()
        self.d_in, self.n_heads, self.width = d_in, n_heads, width
        self.trunk = nn.Sequential(
            nn.Linear(d_in, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
        )
        # One Linear per head rather than a single (n_heads * 2) projection: it
        # keeps per-head weight decay and per-head init independent, and makes a
        # collapsed head visible in the parameters rather than buried in a slice.
        self.heads = nn.ModuleList(nn.Linear(width, 2) for _ in range(n_heads))
        for h in self.heads:
            nn.init.zeros_(h.bias)

    def embed(self, x):
        return self.trunk(x)

    def forward(self, x):
        h = self.trunk(x)
        out = torch.stack([head(h) for head in self.heads], dim=1)  # (B, J, 2)
        return out[..., 0], out[..., 1]                             # mu, log_s

    def config(self):
        return {"d_in": self.d_in, "n_heads": self.n_heads, "width": self.width}
