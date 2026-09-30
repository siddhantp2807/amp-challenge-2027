"""Fit the shipping ensemble and write checkpoints/release/scorer_v1.pt.

The members ARE the cross-validation refits -- one per fold, per family. That is
deliberate: the folds are already cluster-disjoint, so the spread between members
is genuine disagreement on held-out structure rather than seed noise, and it costs
nothing beyond the CV that had to run anyway.

    python -m src.scorer.export
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from src.scorer import data as sdata
from src.scorer.features import descriptor_frame
from src.scorer.train import DEFAULT_CFG, fit_one

DEFAULT_OUT = Path("checkpoints/release/scorer_v1.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", type=Path, default=sdata.DEFAULT_LABELS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    train = sdata.load(args.labels, split="train")
    x, names = descriptor_frame(train.sequences)
    j = train.Y.shape[1]
    print(f"fitting on {len(train)} sequences, {train.n_obs} cells, {x.shape[1]} features")

    cfg = {**DEFAULT_CFG, "lr": args.lr, "_names": names}
    hgb, mlp_state, mlp_cfg, scalers = [], [], [], []

    for fold, (tr, _) in enumerate(sdata.folds(train, args.folds)):
        per_species = []
        for k in range(j):
            rows = tr[train.M[tr, k] & (train.C[tr, k] == 0)]
            m = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.06, max_depth=4, random_state=args.seed
            )
            m.fit(x[rows], train.Y[rows, k])
            per_species.append(m)
        hgb.append(pickle.dumps(per_species))

        model, scaler, nll = fit_one(x, train, tr, cfg, args.seed * 100 + fold)
        mlp_state.append({k: v.cpu() for k, v in model.state_dict().items()})
        mlp_cfg.append(model.config())
        scalers.append(scaler.state_dict())
        print(f"  fold {fold + 1}/{args.folds}  inner-val {nll:.4f}")

    payload = {
        "species": sdata.SPECIES,
        "feature_names": names,
        "hgb": hgb,
        "mlp_state": mlp_state,
        "mlp_cfg": mlp_cfg,
        "scalers": scalers,
        "train_cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "n_train_sequences": len(train),
        "n_train_cells": train.n_obs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    mb = args.out.stat().st_size / 1048576
    print(f"\nwrote {args.out}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
