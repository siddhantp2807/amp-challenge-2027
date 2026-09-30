"""Extra smoothness diagnostics beyond approach/step-1.md's §5 minimum.

- interpolation walks: linearly interpolate in latent space between held-out
  pairs, decode each step, and check that consecutive decodes are close
  (small, roughly monotonic edit distance) and non-degenerate -- a smooth
  path, not sudden jumps into garbage.
- nearest-neighbor decode-consistency: latent-space nearness among train
  points should predict decoded-sequence nearness -- a second, independent
  way to catch "smooth in some regions, cratered in others" that a handful of
  interpolation walks might miss.

`interpolation_monotonicity_proxy` is the cheap subset of this reused by
train/finetune.py as a training-time early-stopping signal.
"""
import numpy as np
import torch
from scipy.stats import spearmanr

from src.eval.common import decode_with_predicted_length, encode_sequences, is_degenerate, normalized_edit_distance

N_INTERP_STEPS = 8


def interpolation_walks(model, sequences: list[str], device, n_pairs: int = 100, seed: int = 0) -> dict:
    mu = encode_sequences(model, sequences, device, sample=False)
    rng = np.random.RandomState(seed)
    n = mu.shape[0]
    pairs = [tuple(rng.choice(n, size=2, replace=False)) for _ in range(min(n_pairs, n * (n - 1) // 2))]

    alphas = torch.linspace(0, 1, N_INTERP_STEPS)
    step_edit_dists = []
    nondegenerate_flags = []
    for i, j in pairs:
        z_i, z_j = mu[i], mu[j]
        path = torch.stack([z_i * (1 - a) + z_j * a for a in alphas])
        decoded = decode_with_predicted_length(model, path, device)
        for a, b in zip(decoded[:-1], decoded[1:]):
            step_edit_dists.append(normalized_edit_distance(a, b))
        nondegenerate_flags.extend(not is_degenerate(s) for s in decoded)

    return {
        "n_pairs": len(pairs),
        "n_interp_steps": N_INTERP_STEPS,
        "mean_consecutive_edit_distance": float(np.mean(step_edit_dists)),
        "max_consecutive_edit_distance": float(np.max(step_edit_dists)),
        "nondegenerate_rate": float(np.mean(nondegenerate_flags)),
        # smooth path: small average step size, no pathological jumps, and
        # intermediate points still decode to plausible (non-degenerate) peptides
        "pass": (
            float(np.mean(step_edit_dists)) < 0.35
            and float(np.mean(nondegenerate_flags)) >= 0.9
        ),
    }


def interpolation_monotonicity_proxy(model, sequences: list[str], device, n_pairs: int = 20, seed: int = 0) -> float:
    """Cheap version for use as a training-time early-stopping signal during
    fine-tuning: mean consecutive-step edit distance along a handful of
    interpolation walks. Lower is smoother; used as one of two OR-conditions
    for early stopping (the other being recon-on-val plateau).
    """
    result = interpolation_walks(model, sequences, device, n_pairs=n_pairs, seed=seed)
    return result["mean_consecutive_edit_distance"]


def nearest_neighbor_consistency(
    model, val_sequences: list[str], train_sequences: list[str], device, k: int = 5, seed: int = 0
) -> dict:
    val_mu = encode_sequences(model, val_sequences, device, sample=False)
    train_mu = encode_sequences(model, train_sequences, device, sample=False)
    val_decoded = decode_with_predicted_length(model, val_mu, device)

    dists = torch.cdist(val_mu, train_mu)  # (n_val, n_train)
    knn_idx = dists.topk(k, largest=False).indices  # (n_val, k)

    train_decoded_cache: dict[int, str] = {}
    latent_dists = []
    seq_dists = []
    for vi in range(val_mu.shape[0]):
        for ti in knn_idx[vi].tolist():
            if ti not in train_decoded_cache:
                train_decoded_cache[ti] = decode_with_predicted_length(
                    model, train_mu[ti : ti + 1], device
                )[0]
            latent_dists.append(dists[vi, ti].item())
            seq_dists.append(normalized_edit_distance(val_decoded[vi], train_decoded_cache[ti]))

    corr, pvalue = spearmanr(latent_dists, seq_dists)
    return {
        "k": k,
        "n_val": val_mu.shape[0],
        "spearman_corr": float(corr),
        "spearman_pvalue": float(pvalue),
        "pass": bool(corr > 0.2 and pvalue < 0.05),
    }


def smoothness_check(model, val_sequences: list[str], train_sequences: list[str], device) -> dict:
    interp = interpolation_walks(model, val_sequences, device)
    nn_consistency = nearest_neighbor_consistency(model, val_sequences, train_sequences, device)
    return {
        "interpolation_walks": interp,
        "nearest_neighbor_consistency": nn_consistency,
        "pass": interp["pass"] and nn_consistency["pass"],
    }
