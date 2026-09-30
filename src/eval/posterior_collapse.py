"""§5.2 Posterior collapse check.

(a) distinct-z decode diversity: decode N independent prior samples, confirm
    the decoder isn't producing near-identical output regardless of z.
(b) perturbation sensitivity: perturb encoded mu at increasing magnitudes,
    confirm output edit distance increases roughly monotonically with
    perturbation magnitude (proportionate changes, not none / not chaotic).
"""
import numpy as np
import torch
from scipy.stats import spearmanr

from src.eval.common import decode_with_predicted_length, encode_sequences, normalized_edit_distance

PERTURBATION_FRACS = [0.1, 0.3, 1.0, 3.0]  # multiples of empirical latent std


def prior_decode_diversity(model, device, n_samples: int = 500, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n_samples, model.d_z, generator=gen)
    decoded = decode_with_predicted_length(model, z, device)

    n_identical_pairs = 0
    n_pairs = 0
    # sample a bounded number of pairs rather than all O(n^2) for large n_samples
    rng = np.random.RandomState(seed)
    pair_idx = rng.choice(n_samples, size=(min(2000, n_samples * 2), 2))
    for i, j in pair_idx:
        if i == j:
            continue
        n_pairs += 1
        if decoded[i] == decoded[j]:
            n_identical_pairs += 1

    identical_frac = n_identical_pairs / max(1, n_pairs)
    return {
        "n_samples": n_samples,
        "identical_pair_frac": identical_frac,
        "pass": identical_frac <= 0.05,
        "decoded_samples": decoded[:20],  # small preview for the report
    }


def perturbation_sensitivity(model, sequences: list[str], device, seed: int = 0) -> dict:
    mu = encode_sequences(model, sequences, device, sample=False)
    latent_std = mu.std(dim=0, keepdim=True) + 1e-6

    baseline_decoded = decode_with_predicted_length(model, mu, device)

    gen = torch.Generator().manual_seed(seed)
    magnitudes = []
    edit_dists = []
    for frac in PERTURBATION_FRACS:
        delta = torch.randn(mu.shape, generator=gen) * latent_std * frac
        perturbed_decoded = decode_with_predicted_length(model, mu + delta, device)
        for base, pert in zip(baseline_decoded, perturbed_decoded):
            magnitudes.append(frac)
            edit_dists.append(normalized_edit_distance(base, pert))

    corr, pvalue = spearmanr(magnitudes, edit_dists)
    return {
        "n_sequences": len(sequences),
        "perturbation_fracs": PERTURBATION_FRACS,
        "spearman_corr": float(corr),
        "spearman_pvalue": float(pvalue),
        "mean_edit_distance_by_frac": {
            frac: float(np.mean([e for m, e in zip(magnitudes, edit_dists) if m == frac]))
            for frac in PERTURBATION_FRACS
        },
        "pass": bool(corr > 0.3 and pvalue < 0.05),
    }


def posterior_collapse_check(model, sequences: list[str], device) -> dict:
    diversity = prior_decode_diversity(model, device)
    sensitivity = perturbation_sensitivity(model, sequences, device)
    return {
        "prior_decode_diversity": diversity,
        "perturbation_sensitivity": sensitivity,
        "pass": diversity["pass"] and sensitivity["pass"],
    }
