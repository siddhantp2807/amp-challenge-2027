"""Reproduce every headline number in SCORER.md, and flag drift from the recorded ones.

    python -m src.scorer.report                     # all three sections
    python -m src.scorer.report --section cv        # just the CV table
    python -m src.scorer.report --section heldout
    python -m src.scorer.report --section library

This exists because the two numbers SCORER.md leads with -- CV rho 0.533 and
held-out test rho 0.518 -- were originally produced by throwaway scripts. A claim
nobody can re-derive from the repo is a claim with nothing behind it.

Each section prints EXPECTED (what SCORER.md records) beside GOT, and the run
exits non-zero if anything has moved by more than `--tol`. So this is a
regression test on the shipped artifact, not just a printout: if a refactor
quietly changes the featurization, the ranking, or the label build, this is what
says so.

Runtime is about 5 minutes end to end -- the CV section refits 5 folds x (6 HGB +
3 MLP seeds), which is the same work the original selection did.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from src.scorer import data as sdata
from src.scorer import metrics as smetrics
from src.scorer.train import DEFAULT_CFG, features_for, fit_one, predict

LIBRARY = Path("generate/library.fasta")
ANTIBACTERIAL = Path("data/antibacterial.fasta")
TOP_MAX_IDENTITY = 0.8
TOP_K = 100

# What SCORER.md records. Update these ONLY together with that document, and only
# with a reason -- they are the point of comparison, not a convenience.
EXPECTED = {
    "cv_hgb_rho": 0.509,
    "cv_mlp_rho": 0.475,
    "cv_ensemble_rho": 0.533,
    "cv_ensemble_conc": 0.716,
    "test_rho": 0.518,
    "test_conc": 0.718,
    "val_rho": 0.395,
    "val_conc": 0.673,
    "library_r2_linear": 0.555,
    "library_r2_quadratic": 0.582,
    "placeholder_r2_quadratic": 1.000,
    "spearman_breadth_charge": 0.600,
    "top100_overlap_with_placeholder": 0,
}


def read_fasta(path):
    return [ln.strip() for ln in open(path) if ln.strip() and not ln.startswith(">")]


# --------------------------------------------------------------------------- CV


def section_cv(seeds=(0, 1, 2), n_splits=5, lr=1e-3):
    """Out-of-fold predictions for both families and their rank average.

    Reproduces SCORER.md section 4. The two families are combined by rank
    averaging per head, which is what ships -- averaging their raw outputs would
    be meaningless, since a boosted tree's conditional mean and a Tobit location
    parameter under a clamped sigma are not on a comparable scale.
    """
    train = sdata.load(split="train")
    x, names = features_for(train.sequences, "f1", "train_f1")
    j = train.Y.shape[1]
    print(f"CV: {len(train)} sequences, {train.n_obs} cells, "
          f"{x.shape[1]} raw features, {len(set(train.clusters))} clusters")

    hgb_oof = np.full_like(train.Y, np.nan, dtype=np.float64)
    mlp_oof = np.zeros_like(hgb_oof)
    cfg = {**DEFAULT_CFG, "lr": lr, "_names": names}

    for fold, (tr, te) in enumerate(sdata.folds(train, n_splits)):
        for k in range(j):
            rows = tr[train.M[tr, k] & (train.C[tr, k] == 0)]
            if len(rows) < 50:
                continue
            m = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.06, max_depth=4, random_state=0
            )
            m.fit(x[rows], train.Y[rows, k])
            hgb_oof[te, k] = m.predict(x[te])

        # The seeds are averaged inside the fold, matching what export.py ships.
        per_seed = [predict(*fit_one(x, train, tr, cfg, s * 100 + fold)[:2], x, te)[0]
                    for s in seeds]
        mlp_oof[te] = np.mean(per_seed, axis=0)
        print(f"  fold {fold + 1}/{n_splits} done")

    hgb_oof = np.nan_to_num(hgb_oof, nan=np.nanmean(hgb_oof))

    # Rank-average over the full out-of-fold column, per head.
    ens = np.column_stack([
        rankdata(hgb_oof[:, k]) + rankdata(mlp_oof[:, k]) for k in range(j)
    ])

    out = {}
    for tag, P in [("hgb", hgb_oof), ("mlp", mlp_oof), ("ensemble", ens)]:
        summ = smetrics.summary(train.Y, train.M, train.C, P, sdata.SPECIES)
        print(smetrics.format_summary(summ, {"hgb": "HGB (F0)",
                                             "mlp": "Tobit MLP x3 seeds",
                                             "ensemble": "rank-average (SHIPS)"}[tag]))
        out[tag] = summ
    return {
        "per_model": out,
        "checks": {
            "cv_hgb_rho": out["hgb"]["weighted_spearman"],
            "cv_mlp_rho": out["mlp"]["weighted_spearman"],
            "cv_ensemble_rho": out["ensemble"]["weighted_spearman"],
            "cv_ensemble_conc": out["ensemble"]["mean_concordance"],
        },
    }


# ---------------------------------------------------------------------- HELDOUT


def section_heldout(bundle):
    """The one-shot measurement on val and test. SCORER.md section 6.

    Spent once, after the bundle was frozen. Read `test` as the honest estimate:
    CV was contaminated by roughly six configuration choices made against it, so
    it is optimistic by construction. `val` is reported beside it not as a
    contradiction but as the scale of the noise -- it is the same size as test and
    disagrees by 0.12.
    """
    out, checks = {}, {}
    for split in ("val", "test"):
        d = sdata.load(split=split)
        # mu is the ensemble's rank-space consensus; the metrics only need a
        # monotone score, so the LCB and breadth steps are not involved here.
        P = bundle.score(d.sequences)["mu"]
        summ = smetrics.summary(d.Y, d.M, d.C, P, sdata.SPECIES)
        print(smetrics.format_summary(summ, f"{split.upper()} (n={len(d)}, {d.n_obs} cells)"))
        print()
        out[split] = summ
        checks[f"{split}_rho"] = summ["weighted_spearman"]
        checks[f"{split}_conc"] = summ["mean_concordance"]
    return {"per_split": out, "checks": checks}


# ---------------------------------------------------------------------- LIBRARY


def placeholder_score(sequences):
    """The scorer this replaced: distance to a fixed point in (charge, GRAVY)."""
    from src.eval.properties import charge_bjellqvist, gravy

    c = np.array([charge_bjellqvist(s) for s in sequences])
    g = np.array([gravy(s) for s in sequences])
    return -(((c - 6.0) / 3.35) ** 2 + ((g - (-0.30)) / 1.00) ** 2)


def section_library(bundle):
    """Does the ranking carry anything the generator did not already impose?

    This is the check that decides whether the scorer was worth building. The
    library was BUILT high-charge -- generate.py conditions on charge and sweeps
    +3..+10 -- so a charge-dominated scorer would just be re-sorting the
    conditioning variable under a new name.

    The placeholder is the reference: regressed on (charge, GRAVY) in a quadratic
    basis it scores R^2 = 1.000, because it IS a quadratic in charge and GRAVY.
    It carried exactly zero independent information.
    """
    from src.amp_challenge_2027.generate import screen_identity
    from src.eval.properties import charge_bjellqvist, gravy

    lib = read_fasta(LIBRARY)
    print(f"library: {len(lib):,} sequences")
    r = bundle.score(lib)
    breadth = r["breadth"]

    length = np.array([len(s) for s in lib], dtype=float)
    charge = np.array([charge_bjellqvist(s) for s in lib])
    grav = np.array([gravy(s) for s in lib])
    G = np.column_stack([length, charge, grav])
    Q = np.column_stack([G, charge ** 2, grav ** 2, charge * grav])

    r2_lin = LinearRegression().fit(G, breadth).score(G, breadth)
    r2_quad = LinearRegression().fit(Q, breadth).score(Q, breadth)
    ph = placeholder_score(lib)
    ph_r2 = LinearRegression().fit(Q, ph).score(Q, ph)
    rho_charge = spearmanr(breadth, charge).statistic

    # Both top-100s, each gated the way generate.py gates: rank first, then take
    # the first 100 that clear the 80% identity bar.
    refs = read_fasta(ANTIBACTERIAL)
    def top100(score_vec):
        order = np.argsort(-np.round(score_vec, 6), kind="stable")
        return screen_identity([lib[i] for i in order], refs, TOP_MAX_IDENTITY, TOP_K)

    top_new, top_old = top100(breadth), top100(ph)
    overlap = len(set(top_new) & set(top_old))

    head_sd = r["mu"].std(axis=0)
    print(f"\n  R2(breadth ~ length, charge, GRAVY)   linear {r2_lin:.3f}   "
          f"quadratic {r2_quad:.3f}   (gate < 0.70)")
    print(f"  placeholder, same quadratic basis      {ph_r2:.3f}")
    print(f"  spearman(breadth, charge)              {rho_charge:+.3f}")
    print(f"  per-head predicted rank sd             {np.round(head_sd, 3)}  "
          f"(collapse if < 0.05)")
    print(f"  top-100 overlap with the placeholder   {overlap}/100")

    def profile(seqs, name):
        idx = [lib.index(s) for s in seqs]
        print(f"    {name:<12s} charge {charge[idx].mean():.2f}  "
              f"GRAVY {grav[idx].mean():+.3f}  length {length[idx].mean():.1f}")
    print("\n  selection profile:")
    profile(top_new, "top-100")
    profile(top_old, "placeholder")
    print(f"    {'library':<12s} charge {charge.mean():.2f}  "
          f"GRAVY {grav.mean():+.3f}  length {length.mean():.1f}")

    return {
        "checks": {
            "library_r2_linear": round(float(r2_lin), 4),
            "library_r2_quadratic": round(float(r2_quad), 4),
            "placeholder_r2_quadratic": round(float(ph_r2), 4),
            "spearman_breadth_charge": round(float(rho_charge), 4),
            "top100_overlap_with_placeholder": int(overlap),
        },
        "per_head_rank_sd": [round(float(v), 4) for v in head_sd],
        "gates": {
            "discriminativeness_r2_below_0.70": bool(r2_quad < 0.70),
            "no_head_collapse": bool((head_sd > 0.05).all()),
        },
    }


# ------------------------------------------------------------------------- MAIN


def compare(checks, tol):
    """EXPECTED vs GOT for everything the sections produced."""
    print("\n" + "=" * 72)
    print(f"{'metric':<38s} {'expected':>10s} {'got':>10s} {'delta':>9s}")
    print("-" * 72)
    drifted = []
    for key, got in checks.items():
        exp = EXPECTED.get(key)
        if exp is None:
            print(f"{key:<38s} {'--':>10s} {got:>10.3f}")
            continue
        delta = got - exp
        flag = "" if abs(delta) <= tol else "   <-- DRIFT"
        print(f"{key:<38s} {exp:>10.3f} {got:>10.3f} {delta:>+9.3f}{flag}")
        if abs(delta) > tol:
            drifted.append((key, exp, got))
    print("=" * 72)
    return drifted


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--section", choices=("cv", "heldout", "library", "all"), default="all")
    ap.add_argument("--bundle", default="checkpoints/release/scorer_v1.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tol", type=float, default=0.02,
                    help="tolerance before a metric counts as drifted")
    ap.add_argument("--out", type=Path, default=Path("reports/scorer_report.json"))
    args = ap.parse_args()

    want = ("cv", "heldout", "library") if args.section == "all" else (args.section,)
    report, checks = {}, {}

    bundle = None
    if {"heldout", "library"} & set(want):
        if not Path(args.bundle).exists():
            raise SystemExit(f"{args.bundle} not found -- run `python -m src.scorer.export` first")
        from src.scorer.predict import ScorerBundle
        bundle = ScorerBundle.load(args.bundle)

    if "cv" in want:
        print("\n### CV -- cluster-grouped, train split only (SCORER.md section 4)\n")
        report["cv"] = section_cv(seeds=tuple(args.seeds))
        checks.update(report["cv"]["checks"])

    if "heldout" in want:
        print("\n### HELD-OUT -- spent once (SCORER.md section 6)\n")
        report["heldout"] = section_heldout(bundle)
        checks.update(report["heldout"]["checks"])

    if "library" in want:
        print("\n### LIBRARY DIAGNOSTICS (SCORER.md section 7)\n")
        if not LIBRARY.exists():
            print(f"  {LIBRARY} not found -- run `uv run generate` first; skipping")
        else:
            report["library"] = section_library(bundle)
            checks.update(report["library"]["checks"])

    drifted = compare(checks, args.tol)
    report["checks"] = checks
    report["expected"] = EXPECTED
    report["tolerance"] = args.tol
    report["drifted"] = [{"metric": k, "expected": e, "got": g} for k, e, g in drifted]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")

    if drifted:
        print(f"\n{len(drifted)} metric(s) moved by more than {args.tol}. Either the "
              f"pipeline changed or SCORER.md is stale -- reconcile before shipping.")
        raise SystemExit(1)
    print("\nall reproduced within tolerance.")


if __name__ == "__main__":
    main()
