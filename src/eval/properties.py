"""Property-level checks: what a conditional stage-2 model actually depends on.

Exact match asks whether a decode reproduces the input residue for residue. Stage 2
(a latent diffusion model, conditioned on peptide properties) instead needs `z` to
carry properties such as length, charge and hydrophobicity, and prior samples to
look like real peptides. These are simple, transparent approximations (net charge
at neutral pH, Kyte-Doolittle GRAVY), not activity predictors.
"""
import math

import numpy as np
import torch

from src.eval.common import decode_with_predicted_length, encode_sequences, is_degenerate

KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
          L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
AA = sorted(KD)

# --- stage-2 conditioning properties -----------------------------------------
# These reproduce data/ld-processed/{pretraining,finetuning}.csv exactly (verified
# to atol 1e-9 on all 35,446 rows). Those labels were produced by
# data-curation/scripts/feature_add/add_pc_features.py, so the tables below are
# copied from the generating libraries verbatim rather than from a textbook --
# both libraries differ from the published values, and the difference is large
# enough to bias reported property MAE. See test_ld_properties.py.

# peptidy.descriptors.charge's pKa sets. NOT the textbook Bjellqvist values:
# using those instead costs ~0.21 charge units of systematic error.
POS_PK = {"Nterm": 7.5, "K": 10, "R": 12, "H": 5.98}
NEG_PK = {"Cterm": 3.55, "D": 4.05, "E": 4.45, "C": 9, "Y": 10}

# modlamp's 'eisenberg' scale: the consensus scale rounded to 2 significant
# figures (I=1.4 not 1.38, R=-2.5 not -2.53, V/L both 1.1, F=1.2 not 1.19).
EISENBERG = {"I": 1.4, "F": 1.2, "V": 1.1, "L": 1.1, "W": 0.81, "M": 0.64, "A": 0.62,
             "G": 0.48, "C": 0.29, "Y": 0.26, "P": 0.12, "T": -0.05, "S": -0.18,
             "H": -0.4, "E": -0.74, "N": -0.78, "Q": -0.85, "D": -0.9, "K": -1.5,
             "R": -2.5}


def net_charge(s: str) -> float:
    return s.count("K") + s.count("R") + 0.1 * s.count("H") - s.count("D") - s.count("E")


def charge_bjellqvist(s: str, pH: float = 7.0) -> float:
    """Henderson-Hasselbalch net charge -- the stage-2 conditioning definition.

    Distinct from net_charge above, which is the rule-of-thumb the freeze report
    has always used. Stage 2 must condition and evaluate on *this* one, since it
    is what data/ld-processed/ holds; mixing the two measures the gap between two
    charge definitions rather than model error.
    """
    pos = sum((1 if a == "Nterm" else s.count(a)) * (10 ** (pk - pH) / (10 ** (pk - pH) + 1))
              for a, pk in POS_PK.items())
    neg = sum((1 if a == "Cterm" else s.count(a)) * (10 ** (pH - pk) / (10 ** (pH - pk) + 1))
              for a, pk in NEG_PK.items())
    return pos - neg


def gravy(s: str) -> float:
    return float(np.mean([KD[c] for c in s])) if s else 0.0


def hydrophobic_moment(s: str, angle: float = 100.0) -> float:
    """Eisenberg hydrophobic moment over the whole sequence, normalized by length.

    Whole-sequence rather than windowed: modlamp's calculate_moment() defaults to
    window=1000, which exceeds every peptide here (<=50 AA), so its modality='max'
    over windows never actually engages.

    Not a stage-2 conditioning property -- reported as a diagnostic, to show
    whether steering length/charge/GRAVY incidentally moves amphipathicity.
    """
    if not s:
        return 0.0
    d = math.radians(angle)
    h = [EISENBERG[c] for c in s]
    re = sum(x * math.cos(i * d) for i, x in enumerate(h))
    im = sum(x * math.sin(i * d) for i, x in enumerate(h))
    return math.hypot(re, im) / len(s)


def composition(s: str) -> np.ndarray:
    return np.array([s.count(a) for a in AA], dtype=float) / max(len(s), 1)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def reconstruct(model, sequences: list[str], device) -> list[str]:
    return decode_with_predicted_length(model, encode_sequences(model, sequences, device), device)


def property_preservation(model, sequences: list[str], device, reconstructions: list[str] | None = None) -> dict:
    """Input vs. its own reconstruction: Pearson r, mean abs error and the input's
    spread for length / net charge / GRAVY, plus composition cosine similarity
    against a shuffled-pair baseline."""
    rec = reconstructions if reconstructions is not None else reconstruct(model, sequences, device)
    out = {}
    for name, f in [("length", len), ("net_charge", net_charge), ("gravy", gravy)]:
        a = np.array([f(s) for s in sequences], dtype=float)
        b = np.array([f(s) for s in rec], dtype=float)
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        out[name] = {"pearson_r": r, "mean_abs_error": float(np.abs(a - b).mean()), "input_std": float(a.std())}
    perm = np.random.RandomState(0).permutation(len(rec))
    out["composition_cosine"] = {
        "matched": float(np.mean([_cos(composition(a), composition(b)) for a, b in zip(sequences, rec)])),
        "shuffled_baseline": float(np.mean([_cos(composition(a), composition(rec[j])) for a, j in zip(sequences, perm)])),
    }
    return out


def prior_sample_properties(model, real_sequences: list[str], device, n: int = 500, seed: int = 0) -> dict:
    """Decode z ~ N(0, I) and compare length / charge / GRAVY distributions with real peptides."""
    z = torch.randn(n, model.d_z, generator=torch.Generator().manual_seed(seed))
    samples = [s for s in decode_with_predicted_length(model, z, device) if s]

    def stats(x):
        return {
            "length": [float(np.mean([len(s) for s in x])), float(np.std([len(s) for s in x]))],
            "net_charge": [float(np.mean([net_charge(s) for s in x])), float(np.std([net_charge(s) for s in x]))],
            "gravy": [float(np.mean([gravy(s) for s in x])), float(np.std([gravy(s) for s in x]))],
        }

    return {
        "n": len(samples),
        "unique_frac": len(set(samples)) / max(len(samples), 1),
        "degenerate_frac": float(np.mean([is_degenerate(s) for s in samples])),
        "samples": stats(samples),
        "real": stats(real_sequences),
        "examples": samples[:8],
    }


def aggregate_posterior_sample_properties(model, train_sequences: list[str], device, n: int = 500, seed: int = 0) -> dict:
    """Sample from where the encoder actually puts real peptides, not from N(0, I).

    Fits a Gaussian (mean and full covariance) to the encoder's mu over the training peptides,
    samples n latents, decodes them and measures charge / GRAVY / composition distance to the
    real composition / novelty. This is a cheap stand-in for what a diffusion model trained on
    those latents would produce. Also reports how often decoding *real* training latents returns
    a near-copy (normalised edit distance < 0.1) of a training peptide: the headroom a diffusion
    model would have to memorise. See notebooks/001.ipynb.
    """
    from rapidfuzz.distance import Levenshtein as RFLev
    from rapidfuzz.process import cdist

    mu = encode_sequences(model, train_sequences, device)
    m = mu.double().mean(0)
    S = torch.cov(mu.double().T) + 1e-6 * torch.eye(mu.shape[1], dtype=torch.float64)
    torch.manual_seed(seed)
    z = torch.distributions.MultivariateNormal(m, covariance_matrix=S).sample((n,)).float()
    samples = [s for s in decode_with_predicted_length(model, z, device) if s]
    real_comp = np.mean([composition(s) for s in train_sequences], axis=0)

    def nn_dist(seqs):
        return cdist(seqs, train_sequences, scorer=RFLev.normalized_distance, workers=-1).min(axis=1)

    idx = torch.randperm(mu.shape[0], generator=torch.Generator().manual_seed(seed))[:n]
    recon = [s for s in decode_with_predicted_length(model, mu[idx], device) if s]
    ch = np.array([net_charge(s) for s in samples])
    gv = np.array([gravy(s) for s in samples])
    nn = nn_dist(samples)
    return {
        "n": len(samples),
        "unique_frac": len(set(samples)) / max(len(samples), 1),
        "degenerate_frac": float(np.mean([is_degenerate(s) for s in samples])),
        "length_mean": float(np.mean([len(s) for s in samples])),
        "charge": [float(ch.mean()), float(ch.std())],
        "gravy": [float(gv.mean()), float(gv.std())],
        "comp_L1": float(np.abs(np.mean([composition(s) for s in samples], axis=0) - real_comp).sum()),
        "novelty_median": float(np.median(nn)),
        "copy_frac": float(np.mean(nn < 0.1)),
        "real_latent_copy_frac": float(np.mean(nn_dist(recon) < 0.1)),
        "examples": samples[:8],
    }
