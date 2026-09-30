"""Featurization for the potency scorer, as a ladder of named blocks.

Two blocks, added one at a time so each has to earn its place in an ablation:

    f1   descriptors only  -- peptidy's 48 plus the four the repo defines itself
    f2   f1 || the frozen VAE's 64-d latent mu

Everything here is deterministic and order-preserving: `build(seqs, "f2")` returns
rows in the order the sequences were passed, so a cached matrix can be indexed by
position.

The descriptor block is filtered once at fit time -- columns that are constant on
the training rows carry no signal and are dropped, which removes peptidy's
non-canonical `freq_*` entries automatically (this corpus is canonical-only).
The surviving column names are stored in the bundle and asserted on load, because
a silent column-order change between training and generate time is
determinism-preserving and completely wrong.
"""
from __future__ import annotations

import numpy as np
import torch
from peptidy.descriptors import compute_descriptors

from src.eval.common import encode_sequences
from src.eval.properties import charge_bjellqvist, gravy, hydrophobic_moment, net_charge

# Fixed so the encode pass cannot depend on how many sequences are being scored.
ENCODE_BATCH = 256

FEATURE_SETS = ("f1", "f2")

# Computed here rather than taken from peptidy: gravy and the hydrophobic moment
# have no peptidy equivalent, and charge_bjellqvist is the exact definition stage 2
# conditioned on, so the scorer and the generator agree on what "charge" means.
REPO_DESCRIPTORS = {
    "repo_charge_bjellqvist": charge_bjellqvist,
    "repo_net_charge": net_charge,
    "repo_gravy": gravy,
    "repo_hydrophobic_moment": hydrophobic_moment,
}


def descriptor_frame(sequences):
    """(N, D) descriptors plus the ordered column names, before any filtering."""
    rows = []
    for s in sequences:
        d = compute_descriptors(s)
        d.update({name: fn(s) for name, fn in REPO_DESCRIPTORS.items()})
        rows.append(d)
    names = sorted(rows[0])
    x = np.array([[r[n] for n in names] for r in rows], dtype=np.float64)
    return x, names


def latent_features(vae, sequences, device):
    """(N, 64) posterior means from the frozen VAE.

    `mu`, not a sample: the measured posterior width ratio is 0.116, below the 0.15
    bar at which resampling is meaningful augmentation, so sampling would add noise
    and buy nothing.
    """
    with torch.no_grad():
        z = encode_sequences(vae, list(sequences), device, batch_size=ENCODE_BATCH)
    return z.cpu().numpy().astype(np.float64)


def build(sequences, feature_set, vae=None, device=None):
    """Raw feature matrix for a rung, plus column names. No scaling, no filtering."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}, expected {FEATURE_SETS}")

    x, names = descriptor_frame(sequences)
    if feature_set == "f2":
        if vae is None:
            raise ValueError("feature set f2 needs a VAE")
        z = latent_features(vae, sequences, device)
        x = np.hstack([x, z])
        names = names + [f"z{i:02d}" for i in range(z.shape[1])]
    return x, names


class FeatureScaler:
    """Standardize, and drop columns that are constant on the fit rows.

    Mirrors `LatentStats` in src/diffusion/latents.py: fit, transform, and a
    state_dict that round-trips through a checkpoint.
    """

    def __init__(self, keep=None, mean=None, std=None, names=None):
        self.keep, self.mean, self.std, self.names = keep, mean, std, names

    @classmethod
    def fit(cls, x, names, eps=1e-8):
        std = x.std(axis=0)
        keep = np.flatnonzero(std > eps)
        return cls(
            keep=keep,
            mean=x[:, keep].mean(axis=0),
            std=np.clip(x[:, keep].std(axis=0), eps, None),
            names=[names[i] for i in keep],
        )

    def transform(self, x):
        return ((x[:, self.keep] - self.mean) / self.std).astype(np.float32)

    @property
    def dim(self):
        return len(self.keep)

    def state_dict(self):
        return {
            "keep": self.keep.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "names": list(self.names),
        }

    @classmethod
    def from_state_dict(cls, d):
        return cls(
            keep=np.asarray(d["keep"], dtype=int),
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
            names=list(d["names"]),
        )
