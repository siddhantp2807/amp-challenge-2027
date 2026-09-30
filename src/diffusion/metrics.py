"""Shared scoring for anything that produces latents: baselines and the sampler.

The headline metric is per-property MAE in units of the *training* standard
deviation, which is what OmegAMP reports. It is uninterpretable on its own, so
every table that carries it must also carry two reference rows:

  decoder ceiling  -- decode the true val latents and re-measure. Nothing using
                      this decoder can score better. Measured: 0.000 / 0.549 /
                      0.451 for length / charge / GRAVY.
  cond. Gaussian   -- the closed-form baseline: 0.487 / 0.744 / 0.582.

So the real headroom is ~0.49 std on length and only ~0.20 / ~0.13 on charge and
GRAVY. A model landing at 0.60 charge is near the ceiling, not failing.
"""
import numpy as np
import torch

from src.diffusion.conditioning import PROPERTIES, compute_properties
from src.eval.common import decode_with_predicted_length, is_degenerate

COPY_THRESHOLD = 0.1          # normalized edit distance below which it's a near-copy
LENGTH_BUCKETS = ((8, 16), (17, 24), (25, 35), (36, 50))


def decode_latents(vae, z: torch.Tensor, device) -> list[str]:
    if not torch.is_tensor(z):
        z = torch.tensor(z, dtype=torch.float32)
    return decode_with_predicted_length(vae, z.float(), device)


def property_mae(sequences: list[str], targets: np.ndarray,
                 train_std: np.ndarray) -> dict:
    """MAE per property, in train-std units. Empty decodes are excluded from the
    property comparison but counted, since len('') = 0 would silently dominate."""
    keep = np.array([len(s) > 0 for s in sequences])
    got = compute_properties([s for s in sequences if s])
    want = np.asarray(targets)[keep]
    mae = np.abs(got - want).mean(axis=0) / np.asarray(train_std)
    return {
        **{p: float(mae[i]) for i, p in enumerate(PROPERTIES)},
        "n_scored": int(keep.sum()),
        "n_empty": int((~keep).sum()),
    }


def diversity_metrics(sequences: list[str], train_sequences: list[str]) -> dict:
    """Uniqueness, degeneracy (overall and bucketed), novelty and copy rate.

    copy_frac here is a *memorization detector*, not the 23-27% figure quoted in
    SUMMARY.md -- that one is real_latent_copy_frac, i.e. what decoding genuine
    training latents returns. The conditional-Gaussian baseline samples at 0.000,
    so any rise on generated samples is the overfitting alarm.
    """
    from rapidfuzz.distance import Levenshtein as RFLev
    from rapidfuzz.process import cdist

    nonempty = [s for s in sequences if s]
    out = {
        "n": len(sequences),
        "unique_frac": len(set(sequences)) / max(len(sequences), 1),
        "degenerate_frac": float(np.mean([is_degenerate(s) for s in sequences])),
    }
    for lo, hi in LENGTH_BUCKETS:
        sub = [s for s in sequences if lo <= len(s) <= hi]
        out[f"degenerate_frac_{lo}-{hi}"] = (
            float(np.mean([is_degenerate(s) for s in sub])) if sub else None
        )
        out[f"n_{lo}-{hi}"] = len(sub)

    if nonempty:
        nn = cdist(nonempty, train_sequences, scorer=RFLev.normalized_distance,
                   workers=-1).min(axis=1)
        out["novelty_median"] = float(np.median(nn))
        out["copy_frac"] = float(np.mean(nn < COPY_THRESHOLD))
    return out


def score_latents(vae, z, targets: np.ndarray, train_std: np.ndarray,
                  train_sequences: list[str], device) -> dict:
    seqs = decode_latents(vae, z, device)
    return {
        "property_mae": property_mae(seqs, targets, train_std),
        **diversity_metrics(seqs, train_sequences),
        "examples": seqs[:8],
    }


def mmd2(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    """Unbiased-ish RBF MMD^2, for comparing generated latents to real ones."""
    x, y = np.asarray(x), np.asarray(y)
    if sigma is None:
        sub = np.vstack([x[:500], y[:500]])
        d = np.linalg.norm(sub[:, None] - sub[None], axis=-1)
        sigma = np.median(d[d > 0]) or 1.0

    def k(a, b):
        d2 = ((a[:, None] - b[None]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * sigma ** 2))

    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())
