"""Out-of-fold metrics for the potency scorer.

Three, each answering something the others cannot:

1. `per_head_spearman` -- rank correlation on the EXACT cells only. Censored cells
   have no true value to correlate against, so including them would score noise.
2. `weighted_mean` -- the primary scalar. Per-head Spearman averaged by cell count,
   NOT a Spearman pooled across heads: pooling is inflated by the between-species
   offsets (P. aeruginosa is simply harder than E. coli), which the model gets for
   free from the head bias and which say nothing about ranking within a species.
3. `censored_concordance` -- the only metric that scores the censored 25%, and the
   second independent detector of a flipped censoring sign. If the sign is
   inverted this collapses toward 0.5 while per-head Spearman barely moves,
   because the exact cells are still fit correctly.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def per_head_spearman(Y, M, C, P):
    """(J,) Spearman per head on exact cells, and (J,) the cell counts used."""
    j = Y.shape[1]
    rhos = np.full(j, np.nan)
    ns = np.zeros(j, dtype=int)
    for k in range(j):
        sel = M[:, k] & (C[:, k] == 0)
        ns[k] = sel.sum()
        if ns[k] >= 20:
            rhos[k] = spearmanr(P[sel, k], Y[sel, k]).statistic
    return rhos, ns


def weighted_mean(rhos, ns):
    """Cell-count-weighted mean of the per-head correlations."""
    ok = np.isfinite(rhos)
    if not ok.any():
        return np.nan
    return float(np.average(rhos[ok], weights=ns[ok]))


def censored_concordance(Y, M, C, P):
    """Fraction of DETERMINED pairs the prediction orders correctly, per head.

    A pair's true ordering is known when:
      - both cells are exact (order by pMIC); or
      - one is exact and the other is a bound that the exact value clears. A
        left-censored pMIC means "true value is at or below this bound", so an
        exact cell ABOVE that bound is definitely more potent. Symmetrically for
        right-censored.
    Two bounds are never comparable, and a bound is uninformative against an exact
    value on the wrong side of it. Those pairs are skipped rather than guessed.
    """
    j = Y.shape[1]
    out = np.full(j, np.nan)
    for k in range(j):
        sel = M[:, k]
        y, c, p = Y[sel, k], C[sel, k], P[sel, k]
        if len(y) < 20:
            continue

        yi, yj = y[:, None], y[None, :]
        ci, cj = c[:, None], c[None, :]

        # i is truly more potent than j when...
        both_exact = (ci == 0) & (cj == 0) & (yi > yj)
        # i exact, j left-censored at yj: j's true value <= yj, so yi > yj settles it
        i_beats_bound = (ci == 0) & (cj == -1) & (yi > yj)
        # i right-censored at yi (true value >= yi), j exact below it
        bound_beats_j = (ci == +1) & (cj == 0) & (yi > yj)
        determined = both_exact | i_beats_bound | bound_beats_j

        if not determined.any():
            continue
        correct = (p[:, None] > p[None, :]) & determined
        out[k] = correct.sum() / determined.sum()
    return out


def summary(Y, M, C, P, species):
    """One row per head plus the headline scalars, as a dict for JSON/reporting."""
    rhos, ns = per_head_spearman(Y, M, C, P)
    conc = censored_concordance(Y, M, C, P)
    return {
        "per_head": {
            s: {"spearman": None if np.isnan(r) else round(float(r), 4),
                "n_exact": int(n),
                "concordance": None if np.isnan(cc) else round(float(cc), 4)}
            for s, r, n, cc in zip(species, rhos, ns, conc)
        },
        "weighted_spearman": round(weighted_mean(rhos, ns), 4),
        "mean_concordance": round(float(np.nanmean(conc)), 4),
    }


def format_summary(summ, label=""):
    lines = [f"{label:<26s} weighted rho {summ['weighted_spearman']:.3f}   "
             f"concordance {summ['mean_concordance']:.3f}"]
    for s, d in summ["per_head"].items():
        rho = "  n/a" if d["spearman"] is None else f"{d['spearman']:+.3f}"
        cc = "  n/a" if d["concordance"] is None else f"{d['concordance']:.3f}"
        lines.append(f"    {s:<26s} rho {rho}  conc {cc}  n={d['n_exact']:>4d}")
    return "\n".join(lines)
