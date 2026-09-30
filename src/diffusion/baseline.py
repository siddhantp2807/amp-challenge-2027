"""Closed-form baselines the diffusion model has to beat.

These are not strawmen. MMD^2 between the aggregate posterior q(z) and a fitted
full-covariance Gaussian is ~0.000-0.0025 (SUMMARY.md Sec. 6), i.e. q(z) is very
nearly Gaussian -- and the conditionals of a Gaussian are then also nearly right.
Measured on target-val, the conditional linear-Gaussian already reaches property
MAE 0.487 / 0.744 / 0.582 (length / charge / GRAVY, in train-std units) with
perfect uniqueness and zero near-copies, against a decoder ceiling of
0.000 / 0.549 / 0.451.

So a 2M-parameter diffusion model trained on 6,690 latents that does not beat
~2,000 closed-form parameters has no justification, and reporting this is the
only thing that makes the diffusion number interpretable.

The GMM variant answers the follow-up: if diffusion only ties the linear-Gaussian,
is p(z|c) genuinely unimodal (a legitimate finding) or is the model failing? A
mixture is the cheapest way to tell.
"""
import numpy as np


class ConditionalGaussian:
    """Joint Gaussian over [z, c]; samples from the exact conditional p(z|c)."""

    def __init__(self, mean, cov, d_z: int):
        self.d_z = d_z
        self.mean_z, self.mean_c = mean[:d_z], mean[d_z:]
        szz, szc, scc = cov[:d_z, :d_z], cov[:d_z, d_z:], cov[d_z:, d_z:]
        self.k = szc @ np.linalg.inv(scc)
        cond_cov = szz - self.k @ szc.T
        # symmetrize before factorizing: cond_cov is symmetric in exact arithmetic
        # but the inverse above leaves asymmetry that can fail Cholesky
        cond_cov = 0.5 * (cond_cov + cond_cov.T)
        self.chol = np.linalg.cholesky(cond_cov + 1e-8 * np.eye(d_z))

    @classmethod
    def fit(cls, z: np.ndarray, c: np.ndarray) -> "ConditionalGaussian":
        joint = np.hstack([z, c])
        return cls(joint.mean(0), np.cov(joint, rowvar=False), z.shape[1])

    def conditional_mean(self, c: np.ndarray) -> np.ndarray:
        return self.mean_z + (np.asarray(c) - self.mean_c) @ self.k.T

    def sample(self, c: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mu = self.conditional_mean(c)
        return mu + rng.standard_normal((len(mu), self.d_z)) @ self.chol.T


class UnconditionalGaussian:
    """Full-covariance fit to the training latents -- what src/generate.py samples."""

    def __init__(self, mean, cov):
        self.mean, self.cov = mean, cov

    @classmethod
    def fit(cls, z: np.ndarray) -> "UnconditionalGaussian":
        return cls(z.mean(0), np.cov(z, rowvar=False))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(self.mean, self.cov, size=n)


class ConditionalGMM:
    """Gaussian mixture over [z, c]; p(z|c) by responsibility-weighted conditionals.

    Distinguishes "the conditional really is unimodal" from "the diffusion model
    underfit" when diffusion fails to beat the linear-Gaussian.
    """

    def __init__(self, gmm, d_z: int):
        self.gmm, self.d_z = gmm, d_z
        self.components = []
        for mean, cov in zip(gmm.means_, gmm.covariances_):
            self.components.append(ConditionalGaussian(mean, cov, d_z))

    @classmethod
    def fit(cls, z: np.ndarray, c: np.ndarray, n_components: int = 8,
            seed: int = 0) -> "ConditionalGMM":
        from sklearn.mixture import GaussianMixture
        joint = np.hstack([z, c])
        gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                              random_state=seed, reg_covar=1e-6).fit(joint)
        return cls(gmm, z.shape[1])

    def _responsibilities(self, c: np.ndarray) -> np.ndarray:
        """p(component | c), from each component's marginal over c."""
        from scipy.stats import multivariate_normal
        d_z = self.d_z
        logp = np.stack([
            np.log(w + 1e-300) + multivariate_normal.logpdf(
                c, mean=m[d_z:], cov=cv[d_z:, d_z:], allow_singular=True)
            for w, m, cv in zip(self.gmm.weights_, self.gmm.means_, self.gmm.covariances_)
        ], axis=1)
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(axis=1, keepdims=True)

    def sample(self, c: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        resp = self._responsibilities(np.asarray(c))
        picks = np.array([rng.choice(len(self.components), p=r) for r in resp])
        out = np.empty((len(c), self.d_z))
        for j, comp in enumerate(self.components):
            sel = picks == j
            if sel.any():
                out[sel] = comp.sample(np.asarray(c)[sel], rng)
        return out
