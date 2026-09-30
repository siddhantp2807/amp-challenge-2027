"""§5.3 Density / holes check.

(a) non-degeneracy of broad prior samples: decoded sequences should be
    reasonable (not degenerate/repetitive) across the space, not just near
    training data's specific latent locations.
(b) aggregate posterior q(z) = mean_i q(z|x_i) vs prior N(0,I): a large
    divergence means the aggregate posterior occupies a narrow sub-region of
    the prior -- exactly the "holes" failure mode, since diffusion trained to
    match q(z) would undersample regions the prior nominally covers.
"""
import numpy as np
import torch

from src.eval.common import decode_with_predicted_length, encode_sequences, is_degenerate


def prior_sample_nondegeneracy(model, device, n_samples: int = 500, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n_samples, model.d_z, generator=gen)
    decoded = decode_with_predicted_length(model, z, device)
    degenerate = [is_degenerate(s) for s in decoded]
    nondegenerate_rate = 1.0 - float(np.mean(degenerate))
    return {
        "n_samples": n_samples,
        "nondegenerate_rate": nondegenerate_rate,
        "pass": nondegenerate_rate >= 0.95,
    }


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float | None = None) -> float:
    """Unbiased-ish MMD^2 estimate with an RBF kernel, median-heuristic
    bandwidth if sigma is not given.
    """
    xy = torch.cat([x, y], dim=0)
    if sigma is None:
        with torch.no_grad():
            dists = torch.cdist(xy, xy)
            sigma = dists[dists > 0].median().item() + 1e-6

    def kernel(a, b):
        d2 = torch.cdist(a, b).pow(2)
        return torch.exp(-d2 / (2 * sigma**2))

    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)
    m, n = x.shape[0], y.shape[0]
    return float(
        (kxx.sum() - kxx.trace()) / (m * (m - 1))
        + (kyy.sum() - kyy.trace()) / (n * (n - 1))
        - 2 * kxy.mean()
    )


def aggregate_posterior_vs_prior(model, sequences: list[str], device, seed: int = 0) -> dict:
    mu = encode_sequences(model, sequences, device, sample=False)

    per_dim_mean_dev = mu.mean(dim=0).abs().mean().item()
    per_dim_var_dev = (mu.var(dim=0) - 1.0).abs().mean().item()

    gen = torch.Generator().manual_seed(seed)
    n = min(mu.shape[0], 500)
    idx = torch.randperm(mu.shape[0], generator=gen)[:n]
    prior_sample = torch.randn(n, model.d_z, generator=gen)
    mmd2 = _mmd_rbf(mu[idx], prior_sample)

    return {
        "n_sequences": len(sequences),
        "per_dim_mean_abs_deviation": per_dim_mean_dev,
        "per_dim_var_abs_deviation": per_dim_var_dev,
        "mmd2_vs_prior": mmd2,
        # thresholds are heuristic starting points; the freeze report surfaces
        # the raw numbers regardless so they can be recalibrated empirically.
        "pass": mmd2 < 0.1 and per_dim_mean_dev < 0.5 and per_dim_var_dev < 0.5,
    }


def density_holes_check(model, train_sequences: list[str], device) -> dict:
    nondegeneracy = prior_sample_nondegeneracy(model, device)
    posterior_vs_prior = aggregate_posterior_vs_prior(model, train_sequences, device)
    return {
        "prior_sample_nondegeneracy": nondegeneracy,
        "aggregate_posterior_vs_prior": posterior_vs_prior,
        "pass": nondegeneracy["pass"] and posterior_vs_prior["pass"],
    }
