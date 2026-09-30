"""Decoder faithfulness: does `decode` render `z` correctly?

This replaces exact-match reconstruction as the freeze gate. Exact match
conflates two different failures -- the encoder discarding information, and
the decoder mis-rendering information it was given -- and only the second is
"the decoder is wrong". It is also unreachable for this architecture: a
non-autoregressive decoder painting up to 50 positions from one broadcast `z`
tops out around 3.5% exact match even on training data, so gating on 80%
gates on the architecture, not on the checkpoint.

What stage 2 actually needs is that `z -> sequence` be well-defined, faithful
and injective, which these four checks measure directly:

(a) latent round trip: z -> decode -> encode -> z_hat. Asks whether the
    decoder emits a sequence the encoder reads back as the *same* latent,
    whatever its edit distance to the original input. Scored as per-dimension
    R^2, plus a whitened distance reported against the whitened distance
    between two random real peptides so the scale is interpretable.
(b) non-collapse: (a) is gameable -- a decoder collapsing onto a handful of
    sequences that the encoder maps back consistently scores perfectly and is
    useless -- so round-trip fidelity only counts paired with coverage.
(c) z-utilization: decode from the mean latent of all training sequences of
    the same length, i.e. all length information and no per-sequence
    information, and require true `z` to beat that baseline by a wide margin.
    Guards against a decoder that is really just a length-conditioned prior.
(d) noise survival: a diffusion sampler hands the decoder `z + eps`, never an
    exact `z`, so decodes must stay non-degenerate under perturbation on the
    scale the sampler will leave behind.

Perturbations here are scaled by each dimension's own empirical std, so the
numbers are comparable across checkpoints whose latent scale differs (which
it does -- see free_bits' effect on per-dim variance).
"""
import collections

import numpy as np
import torch

from src.eval.common import decode_with_predicted_length, encode_sequences, is_degenerate

ROUND_TRIP_R2_GATE = 0.80
UNIQUE_FRAC_GATE = 0.98
DEGENERATE_FRAC_GATE = 0.05
Z_UTILIZATION_RATIO_GATE = 3.0
NOISE_SURVIVAL_FRAC = 0.25
NOISE_LEVELS = (0.05, 0.1, 0.25, 0.5)


def _latent_std(model, train_sequences: list[str], device) -> torch.Tensor:
    mu = encode_sequences(model, train_sequences, device, sample=False)
    return mu.std(dim=0) + 1e-6


def position_accuracy(references: list[str], decodes: list[str]) -> float:
    """Fraction of residues placed correctly, denominator = total reference
    length (so a short decode is penalized for what it omits).
    """
    total = hits = 0
    for ref, dec in zip(references, decodes):
        total += len(ref)
        hits += sum(1 for i in range(min(len(ref), len(dec))) if ref[i] == dec[i])
    return hits / max(total, 1)


def latent_round_trip(model, sequences: list[str], device, std: torch.Tensor, seed: int = 0) -> dict:
    """z -> decode -> encode -> z_hat, scored in whitened latent space."""
    mu = encode_sequences(model, sequences, device, sample=False)
    decoded = decode_with_predicted_length(model, mu, device)
    z_hat = encode_sequences(model, decoded, device, sample=False)

    r2 = float(1.0 - ((mu - z_hat).var(dim=0) / mu.var(dim=0).clamp(min=1e-8)).mean())
    rt_dist = ((mu - z_hat) / std).norm(dim=1)

    # reference scale: how far apart two random real peptides are, same metric
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(mu.shape[0], generator=gen)
    pair_dist = ((mu - mu[perm]) / std).norm(dim=1)

    mean_rt, mean_pair = float(rt_dist.mean()), float(pair_dist.mean())
    return {
        "n": len(sequences),
        "per_dim_r2": r2,
        "mean_whitened_distance": mean_rt,
        "random_real_pair_distance": mean_pair,
        "distance_ratio": mean_rt / max(mean_pair, 1e-8),
        "pass": r2 >= ROUND_TRIP_R2_GATE,
    }


def decode_non_collapse(model, sequences: list[str], device) -> dict:
    """Coverage check that closes the round trip's degenerate-solution loophole."""
    mu = encode_sequences(model, sequences, device, sample=False)
    decoded = decode_with_predicted_length(model, mu, device)
    unique_frac = len(set(decoded)) / max(len(decoded), 1)
    degenerate_frac = float(np.mean([is_degenerate(s) for s in decoded]))
    return {
        "n": len(decoded),
        "unique_frac": unique_frac,
        "degenerate_frac": degenerate_frac,
        "pass": unique_frac >= UNIQUE_FRAC_GATE and degenerate_frac <= DEGENERATE_FRAC_GATE,
    }


def z_utilization(model, sequences: list[str], train_sequences: list[str], device, min_per_length: int = 5) -> dict:
    """True `z` vs a length-matched mean latent carrying no per-sequence information."""
    mu_train = encode_sequences(model, train_sequences, device, sample=False)
    by_length = collections.defaultdict(list)
    for seq, z in zip(train_sequences, mu_train):
        by_length[len(seq)].append(z)
    length_mean = {
        length: torch.stack(zs).mean(dim=0)
        for length, zs in by_length.items()
        if len(zs) >= min_per_length
    }

    scored = [s for s in sequences if len(s) in length_mean]
    if not scored:
        return {"n": 0, "pass": False, "note": "no sequences with a length-matched train baseline"}

    mu = encode_sequences(model, scored, device, sample=False)
    acc_true = position_accuracy(scored, decode_with_predicted_length(model, mu, device))

    z_baseline = torch.stack([length_mean[len(s)] for s in scored])
    acc_baseline = position_accuracy(scored, decode_with_predicted_length(model, z_baseline, device))

    ratio = acc_true / max(acc_baseline, 1e-8)
    return {
        "n": len(scored),
        "position_accuracy_from_z": acc_true,
        "position_accuracy_from_length_mean": acc_baseline,
        "ratio": ratio,
        "pass": ratio >= Z_UTILIZATION_RATIO_GATE,
    }


def noise_survival(model, sequences: list[str], device, std: torch.Tensor, seed: int = 0) -> dict:
    """Degeneracy under the perturbation scale a diffusion sampler will leave behind."""
    mu = encode_sequences(model, sequences, device, sample=False)
    gen = torch.Generator().manual_seed(seed)

    by_level = {}
    for frac in NOISE_LEVELS:
        z = mu + torch.randn(mu.shape, generator=gen) * std * frac
        decoded = decode_with_predicted_length(model, z, device)
        z_hat = encode_sequences(model, decoded, device, sample=False)
        by_level[str(frac)] = {
            "degenerate_frac": float(np.mean([is_degenerate(s) for s in decoded])),
            "unique_frac": len(set(decoded)) / max(len(decoded), 1),
            "round_trip_whitened_distance": float(((z - z_hat) / std).norm(dim=1).mean()),
        }

    at_gate = by_level[str(NOISE_SURVIVAL_FRAC)]
    return {
        "n": len(sequences),
        "gate_noise_frac": NOISE_SURVIVAL_FRAC,
        "by_noise_level": by_level,
        "pass": at_gate["degenerate_frac"] <= DEGENERATE_FRAC_GATE,
    }


def faithfulness_check(model, sequences: list[str], train_sequences: list[str], device) -> dict:
    std = _latent_std(model, train_sequences, device)
    round_trip = latent_round_trip(model, sequences, device, std)
    non_collapse = decode_non_collapse(model, sequences, device)
    utilization = z_utilization(model, sequences, train_sequences, device)
    noise = noise_survival(model, sequences, device, std)
    result = {
        "latent_round_trip": round_trip,
        "decode_non_collapse": non_collapse,
        "z_utilization": utilization,
        "noise_survival": noise,
    }
    result["pass"] = all(sub["pass"] for sub in result.values())
    return result
