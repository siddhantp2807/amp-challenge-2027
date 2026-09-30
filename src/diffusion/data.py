"""Assembling the stage-2 training set: latents, conditions, and the two phases.

Phase A pretrains on the deduplicated union of the broad corpus and target-train
(30,938 latents); phase B fine-tunes on target-train alone (7,609). This mirrors
stage 1's own two-phase plan, and matters more here than there: 7,609 latents
against a ~2.3M-parameter denoiser is the dominant overfitting risk.

Leakage: data/ld-processed/pretraining.csv shares 4,762 sequences with
target-train but 0 with target-val and 0 with target-test (the corpus was already
CD-HIT cluster-filtered against val/test in stage 1). The union is therefore safe
to train on, and is deduplicated by sequence with is_target=1 winning.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.diffusion.conditioning import PROPERTIES, PropertyStandardizer, load_conditions
from src.diffusion.latents import encode_posterior


@dataclass
class LatentSet:
    sequences: list[str]
    mu: torch.Tensor          # (N, d_z)
    logvar: torch.Tensor      # (N, d_z)
    conditions: np.ndarray    # (N, K) raw property values
    is_target: torch.Tensor   # (N,) long

    def __len__(self):
        return len(self.sequences)


def build_split_frames(data_cfg: dict) -> dict:
    """target train/val/test frames plus the phase-A union, all with conditions."""
    target = load_conditions("finetuning", data_cfg["finetune_clean"])
    broad = load_conditions("pretraining")

    frames = {s: target[target.split == s].reset_index(drop=True)
              for s in ("train", "val", "test")}

    broad = broad.copy()
    broad["is_target"] = 0
    tr = frames["train"].copy()
    tr["is_target"] = 1
    # dedup with is_target=1 winning: concat target first, then drop later dupes
    union = pd.concat([tr, broad], ignore_index=True)
    union = union.drop_duplicates(subset="sequence", keep="first").reset_index(drop=True)

    held_out = set(frames["val"].sequence) | set(frames["test"].sequence)
    leaked = held_out & set(union.sequence)
    if leaked:
        raise RuntimeError(
            f"{len(leaked)} target val/test sequences leaked into the phase-A union "
            "-- the pretraining corpus's cluster exclusion has been broken."
        )
    frames["union"] = union
    return frames


def encode_frame(model, frame: pd.DataFrame, device) -> LatentSet:
    seqs = frame["sequence"].tolist()
    mu, logvar = encode_posterior(model, seqs, device)
    is_target = torch.tensor(
        frame["is_target"].to_numpy() if "is_target" in frame else np.ones(len(frame)),
        dtype=torch.long,
    )
    return LatentSet(
        sequences=seqs,
        mu=mu,
        logvar=logvar,
        conditions=frame[list(PROPERTIES)].to_numpy(dtype=np.float64),
        is_target=is_target,
    )


def standardized_conditions(ls: LatentSet, std: PropertyStandardizer) -> torch.Tensor:
    return torch.tensor(std.transform(ls.conditions), dtype=torch.float32)


def sample_latents(ls: LatentSet, stats, indices: torch.Tensor,
                   resample: bool = True, jitter_sigma: float = 0.0) -> torch.Tensor:
    """Normalized latents for a minibatch.

    resample=True draws z ~ q(z|x) rather than returning mu. The encoder's logvar
    is a calibrated statement of how far z can move without changing the decoded
    sequence, so this is the VAE's own augmentation, and it teaches the model the
    conditional spread p(z|c) instead of one point per condition. Evaluation uses
    mu so the val loss is not noise-dominated.

    But measure before trusting it: preflight's G6 reports the mean posterior std
    relative to the marginal spread of mu, and on this checkpoint it is 0.116 --
    below the 0.15 bar, i.e. free_bits=2.5 made the posteriors tight enough that
    resampling barely moves anything. `jitter_sigma` adds explicit noise in
    *normalized* latent units on top, which is the prescribed fallback; sweep it
    over {0, 0.05, 0.1} rather than assuming a value.
    """
    mu = ls.mu[indices]
    z = mu + torch.randn_like(mu) * torch.exp(0.5 * ls.logvar[indices]) if resample else mu
    z = stats.normalize(z)
    if jitter_sigma > 0:
        z = z + torch.randn_like(z) * jitter_sigma
    return z
