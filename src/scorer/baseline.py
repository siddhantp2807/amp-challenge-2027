"""Rung F0 -- the reference bar, not the model.

One `HistGradientBoostingRegressor` per species, descriptors only, EXACT cells
only, no censoring, no shared trunk. Deliberately the simplest thing that could
work, run through the identical cluster-grouped CV as every later rung.

This exists so the neural multi-head Tobit model has something honest to beat. If
F1 cannot clear this, the machinery is not buying anything and F0 is what ships.

    python -m src.scorer.baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from src.scorer import data as sdata
from src.scorer import metrics as smetrics
from src.scorer.features import descriptor_frame


def run_cv(labels_path, n_splits=5, seed=0):
    train = sdata.load(labels_path, split="train")
    x, _ = descriptor_frame(train.sequences)
    print(f"F0: {len(train)} sequences, {train.n_obs} observed cells, "
          f"{x.shape[1]} raw descriptors")

    # Out-of-fold predictions, filled fold by fold.
    P = np.full_like(train.Y, np.nan, dtype=np.float64)

    for fold, (tr, te) in enumerate(sdata.folds(train, n_splits)):
        for k in range(train.Y.shape[1]):
            # exact cells only -- HGB has no censored likelihood
            fit_rows = tr[train.M[tr, k] & (train.C[tr, k] == 0)]
            if len(fit_rows) < 50:
                continue
            model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.06, max_depth=4, random_state=seed
            )
            model.fit(x[fit_rows], train.Y[fit_rows, k])
            P[te, k] = model.predict(x[te])
        print(f"  fold {fold + 1}/{n_splits} done")

    P = np.nan_to_num(P, nan=np.nanmean(P))
    return smetrics.summary(train.Y, train.M, train.C, P, sdata.SPECIES)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", type=Path, default=sdata.DEFAULT_LABELS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=Path("reports/scorer_f0.json"))
    args = ap.parse_args()

    runs = []
    for seed in args.seeds:
        print(f"\nseed {seed}")
        summ = run_cv(args.labels, args.folds, seed)
        print(smetrics.format_summary(summ, f"F0 seed {seed}"))
        runs.append(summ)

    w = [r["weighted_spearman"] for r in runs]
    c = [r["mean_concordance"] for r in runs]
    print(f"\nF0 REFERENCE BAR   weighted rho {np.mean(w):.3f} +/- {np.std(w):.3f}"
          f"   concordance {np.mean(c):.3f} +/- {np.std(c):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"runs": runs, "weighted_spearman_mean": float(np.mean(w)),
         "weighted_spearman_sd": float(np.std(w)),
         "concordance_mean": float(np.mean(c))}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
