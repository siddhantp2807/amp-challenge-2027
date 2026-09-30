"""Train and cross-validate the potency scorer.

    python -m src.scorer.train --features f1 --mode cv
    python -m src.scorer.train --features f2 --mode cv
    python -m src.scorer.train --features f2 --mode fit    # the shipping refits

CV is `GroupKFold` over CD-HIT clusters on the TRAIN split only. The 133-sequence
val and 110-sequence test splits are far too small to select on -- a probe on 67
of them moved 0.13 Spearman between two near-identical models -- so they are never
touched here.

Inside each outer fold, a further 15% of that fold's clusters is held out to early
stop on. Grouping there too, not at random: an inner-val peptide homologous to an
inner-train one would stop training on a memorization signal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.scorer import data as sdata
from src.scorer import metrics as smetrics
from src.scorer.features import FeatureScaler, build
from src.scorer.model import PotencyScorer, tobit_nll

CACHE = Path("data/processed/scorer-cache")


def features_for(sequences, feature_set, cache_key):
    """Build (and disk-cache) the raw feature matrix.

    The VAE encode is the only expensive step and it is identical across folds and
    seeds, so it happens exactly once per feature set.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    xpath, npath = CACHE / f"{cache_key}_x.npy", CACHE / f"{cache_key}_names.json"
    if xpath.exists() and npath.exists():
        return np.load(xpath), json.loads(npath.read_text())

    vae = device = None
    if feature_set == "f2":
        from src.diffusion.latents import load_vae
        from src.train.utils import get_device
        device = get_device()
        vae = load_vae("checkpoints/release/finetune_fb3p0_lip0p1_v1_best.pt", device)
        print(f"  encoding {len(sequences)} sequences on {device}")

    x, names = build(sequences, feature_set, vae=vae, device=device)
    np.save(xpath, x)
    npath.write_text(json.dumps(names))
    return x, names


def _inner_split(clusters, idx, frac, rng):
    """Hold out `frac` of the clusters present in `idx`, for early stopping."""
    uniq = np.unique(clusters[idx])
    rng.shuffle(uniq)
    n_val = max(1, int(round(frac * len(uniq))))
    val_clusters = set(uniq[:n_val].tolist())
    is_val = np.array([clusters[i] in val_clusters for i in idx])
    return idx[~is_val], idx[is_val]


def fit_one(x, train, tr_idx, cfg, seed, verbose=False):
    """Fit one model on `tr_idx`, early stopping on an inner cluster holdout."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    fit_idx, stop_idx = _inner_split(train.clusters, tr_idx, cfg["inner_val_frac"], rng)

    scaler = FeatureScaler.fit(x[fit_idx], cfg["_names"])
    Xf = torch.from_numpy(scaler.transform(x[fit_idx]))
    Xs = torch.from_numpy(scaler.transform(x[stop_idx]))

    def tensors(idx):
        return (torch.from_numpy(train.Y[idx]),
                torch.from_numpy(train.M[idx]),
                torch.from_numpy(train.C[idx].astype(np.int8)))

    Yf, Mf, Cf = tensors(fit_idx)
    Ys, Ms, Cs = tensors(stop_idx)

    # Head weights from THIS fold's training cells only -- computing them on the
    # full label set would leak the held-out fold's class balance.
    hw = torch.from_numpy(sdata.head_weights(Mf.numpy().sum(axis=0)))

    model = PotencyScorer(scaler.dim, train.Y.shape[1],
                          width=cfg["width"], dropout=cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    best, best_state, since = float("inf"), None, 0
    n = len(fit_idx)
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.from_numpy(rng.permutation(n))
        for i in range(0, n, cfg["batch_size"]):
            b = perm[i : i + cfg["batch_size"]]
            mu, log_s = model(Xf[b])
            loss = tobit_nll(mu, log_s, Yf[b], Mf[b], Cf[b], hw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            mu, log_s = model(Xs)
            if cfg["stop_on"] == "nll":
                val = tobit_nll(mu, log_s, Ys, Ms, Cs, hw).item()
            else:
                # The deliverable is a RANKING, and Tobit NLL is not a ranking
                # loss -- it is dominated by the sigma term and by the censored
                # mass, so its minimum sits well away from the best-ranking model.
                # Selecting on inner-val rank quality directly costs nothing and
                # is still strictly inside the training fold.
                rhos, ns = smetrics.per_head_spearman(
                    Ys.numpy(), Ms.numpy(), Cs.numpy(), mu.numpy())
                val = -smetrics.weighted_mean(rhos, ns)
                if not np.isfinite(val):
                    val = float("inf")
        if val < best - 1e-4:
            best, since = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg["patience"]:
                break
        if verbose and epoch % 50 == 0:
            print(f"      epoch {epoch:3d}  inner-val NLL {val:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    return model, scaler, best


def predict(model, scaler, x, idx):
    with torch.no_grad():
        mu, log_s = model(torch.from_numpy(scaler.transform(x[idx])))
    return mu.numpy(), log_s.numpy()


def run_cv(x, names, train, cfg, seeds, n_splits=5):
    runs, oof_all = [], []
    for seed in seeds:
        P = np.full_like(train.Y, np.nan, dtype=np.float64)
        S = np.full_like(train.Y, np.nan, dtype=np.float64)
        for fold, (tr, te) in enumerate(sdata.folds(train, n_splits)):
            cfg = {**cfg, "_names": names}
            model, scaler, nll = fit_one(x, train, tr, cfg, seed * 100 + fold)
            mu, log_s = predict(model, scaler, x, te)
            P[te], S[te] = mu, np.exp(np.clip(log_s, np.log(0.05), np.log(3.0)))
            print(f"    fold {fold + 1}/{n_splits}  inner-val NLL {nll:.4f}")
        summ = smetrics.summary(train.Y, train.M, train.C, P, sdata.SPECIES)
        print(smetrics.format_summary(summ, f"  seed {seed}"))
        runs.append(summ)
        oof_all.append((P, S))
    return runs, oof_all


DEFAULT_CFG = {
    "width": 192,
    "dropout": 0.3,
    "lr": 3e-3,
    "weight_decay": 1e-2,
    "batch_size": 128,
    "epochs": 400,
    "patience": 40,
    "inner_val_frac": 0.15,
    "stop_on": "spearman",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", choices=("f1", "f2"), required=True)
    ap.add_argument("--mode", choices=("cv", "fit"), default="cv")
    ap.add_argument("--labels", type=Path, default=sdata.DEFAULT_LABELS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--stop-on", choices=("nll", "spearman"), default=None)
    ap.add_argument("--tag", default="", help="suffix for the report filename")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    for key in ("lr", "dropout", "width", "patience", "stop_on"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    train = sdata.load(args.labels, split="train")
    x, names = features_for(train.sequences, args.features,
                            f"train_{args.features}")
    print(f"{args.features}: {len(train)} sequences, {train.n_obs} cells, "
          f"{x.shape[1]} raw features")

    runs, _ = run_cv(x, names, train, cfg, args.seeds, args.folds)

    w = [r["weighted_spearman"] for r in runs]
    c = [r["mean_concordance"] for r in runs]
    print(f"\n{args.features.upper()}   weighted rho {np.mean(w):.3f} +/- {np.std(w):.3f}"
          f"   concordance {np.mean(c):.3f} +/- {np.std(c):.3f}")

    out = args.out or Path(f"reports/scorer_{args.features}{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"features": args.features, "cfg": {k: v for k, v in cfg.items()
                                            if not k.startswith("_")},
         "runs": runs,
         "weighted_spearman_mean": float(np.mean(w)),
         "weighted_spearman_sd": float(np.std(w)),
         "concordance_mean": float(np.mean(c))}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
