#!/usr/bin/env python3
"""
01-build-mic-labels.py
======================
STEP 1 of the scorer pipeline.

Turns the standardised DBAASP MIC frame into the long label table the potency
scorer trains on: one row per (sequence, species), pMIC instead of MIC, censoring
expressed in pMIC space, and the existing CD-HIT homology-aware split attached.

    sequence, cluster, split, species, y_pmic, censor, n_rep

MHB ONLY
--------
Input is data/processed-data/dbaasp-std/filtered-dbaasp.csv, which is 100% MHB.
That is asserted at load, so a future re-run of 05-standardize-mic.py with a wider
--medium cannot silently mix broths into a label set that has no medium covariate.

THE CENSORING SIGN FLIP
-----------------------
This is the one thing in the whole pipeline that is silent when wrong, so it is
done here, once, and nothing downstream reasons about it again.

DBAASP records `censor_direction` in MIC space: ">128 µM" is a RIGHT-censored MIC,
meaning "no activity up to 128" -- an inactive peptide. pMIC = -log10(MIC in M) is
DECREASING in MIC, so that same observation is a LEFT-censored pMIC: the true pMIC
is somewhere at or below the bound.

Get this backwards and the model learns that the 2,660 inactive rows are the most
potent ones. It will not announce itself -- CV Spearman on the exact rows barely
moves, because those are still fit correctly. The tripwires are the assertion at
the bottom of this file and tests/test_censoring_sign.py.

    censor = 0   exact
           = -1  left-censored pMIC  (from a right-censored MIC: inactive)
           = +1  right-censored pMIC (from a left-censored MIC: potent, 28 rows)

`y_pmic` holds the BOUND for censored rows, not a value.

USAGE
-----
    python scripts/mic/01-build-mic-labels.py
    python scripts/mic/01-build-mic-labels.py --out data/processed-data/mic-labels/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# build_cluster_dataframe lives in the clustering pipeline; reuse it rather than
# writing a second .clstr parser that can drift from the first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cluster_and_segregate"))
from importlib import import_module  # noqa: E402

build_cluster_dataframe = import_module("filter-pretrain").build_cluster_dataframe

DEFAULT_MIC = Path("data/processed-data/dbaasp-std/filtered-dbaasp.csv")
DEFAULT_SEGREGATED = Path(
    "data/processed-data/cluster_and_segregate/segregate/finetune_segregated.csv"
)
DEFAULT_CLSTR = Path(
    "data/processed-data/cluster_and_segregate/cd-hit-output/finetune50_output.clstr"
)
DEFAULT_OUT = Path("data/processed-data/mic-labels")

# The six best-supported organisms become heads. The four below ~300 sequences
# (S. enterica x2, E. faecium, E. cloacae) are dropped rather than pooled --
# pooling them would inject strain heterogeneity into a head that has none.
HEAD_SPECIES = [
    "Escherichia coli",
    "Staphylococcus aureus",
    "Pseudomonas aeruginosa",
    "Klebsiella pneumoniae",
    "Acinetobacter baumannii",
    "Enterococcus faecalis",
]

# Expected shape of the input, from the committed filtered-dbaasp.csv.
EXPECT_ROWS = 10_663
EXPECT_SEQS = 2_344
EXPECT_SPLITS = {"train": 2_101, "val": 133, "test": 110}
MIN_TRAIN_CLUSTERS = 496


def load_mic(path):
    """The standardised MIC frame, with medium and column choice checked."""
    df = pd.read_csv(path)

    media = sorted(df["medium"].dropna().unique())
    if media != ["MHB"]:
        raise SystemExit(
            f"{path} contains media {media}, expected ['MHB'] only.\n"
            "This label set has no medium covariate, so mixing broths would "
            "silently widen the target. Re-run 05-standardize-mic.py without a "
            "wider --medium, or add a medium feature first."
        )

    # mod_concentration is the unit-converted column (µM). cleaned_concentration
    # is pre-conversion and 48% of its rows are µg/ml -- thresholding that one
    # mixes two unit systems (see notebooks/002).
    conc = pd.to_numeric(df["mod_concentration"], errors="coerce")
    df = df.assign(mod_concentration=conc)
    df = df[df["mod_concentration"] > 0]
    return df


def to_pmic(df):
    """MIC in µM -> pMIC in M, and censoring flipped into pMIC space."""
    # µM -> M is exactly -6 in log space.
    df = df.assign(y_pmic=6.0 - np.log10(df["mod_concentration"]))

    # The flip. MIC-right ("no activity up to X") becomes pMIC-left.
    flip = {"right": -1, "left": +1}
    df = df.assign(
        censor=df["censor_direction"].map(flip).fillna(0).astype("int8")
    )
    return df


def collapse(group):
    """One row per (sequence, species).

    An exact measurement always beats a bound, so if any exact rows exist the cell
    is their median and the bounds are discarded. A cell that is only ever bounded
    keeps the single weakest-but-still-true bound: for a left-censored pMIC that is
    the LOWEST bound (the highest concentration actually tested), which is the
    claim that over-punishes the peptide least.

    A value and a bound are never mixed into one number.
    """
    exact = group[group["censor"] == 0]
    if len(exact):
        return pd.Series(
            {"y_pmic": exact["y_pmic"].median(), "censor": np.int8(0),
             "n_rep": len(exact)}
        )

    left = group[group["censor"] == -1]
    if len(left):
        return pd.Series(
            {"y_pmic": left["y_pmic"].min(), "censor": np.int8(-1),
             "n_rep": len(left)}
        )

    right = group[group["censor"] == +1]
    return pd.Series(
        {"y_pmic": right["y_pmic"].max(), "censor": np.int8(+1), "n_rep": len(right)}
    )


def attach_clusters(df, segregated_path, clstr_path):
    """Sequence -> FT_n -> cluster, plus the homology-aware split."""
    seg = pd.read_csv(segregated_path)
    clusters = build_cluster_dataframe(clstr_path)[["id", "cluster"]]
    seg = seg.merge(clusters, on="id", how="left")
    if seg["cluster"].isna().any():
        raise SystemExit(
            f"{seg['cluster'].isna().sum()} sequences in {segregated_path} have no "
            f"cluster in {clstr_path}; the two files are out of sync."
        )

    out = df.merge(seg[["sequence", "split", "cluster"]], on="sequence", how="left")
    missing = out["split"].isna().sum()
    if missing:
        raise SystemExit(
            f"{missing} labelled rows have no split. Every DBAASP sequence is "
            "expected to be in the finetune set."
        )
    return out


def check(labels, raw_seqs):
    """Tripwires. Each one has cost a run somewhere, so they fail hard."""
    if raw_seqs != EXPECT_SEQS:
        raise SystemExit(f"expected {EXPECT_SEQS} input sequences, got {raw_seqs}")

    per_split = labels.groupby("split")["sequence"].nunique().to_dict()
    for split, n in EXPECT_SPLITS.items():
        if per_split.get(split, 0) > n:
            raise SystemExit(
                f"split {split} has {per_split.get(split)} sequences, more than the "
                f"{n} in the finetune split -- the join has duplicated rows."
            )

    train_clusters = labels[labels["split"] == "train"]["cluster"].nunique()
    if train_clusters < MIN_TRAIN_CLUSTERS * 0.9:
        raise SystemExit(
            f"only {train_clusters} train clusters (expected ~{MIN_TRAIN_CLUSTERS}); "
            "5-fold GroupKFold needs the cluster count to hold up."
        )

    # THE sign-flip tripwire. Censored rows are the inactive tail, so in pMIC
    # space (higher = more potent) their median must sit BELOW the exact rows'.
    med_exact = labels.loc[labels["censor"] == 0, "y_pmic"].median()
    med_cens = labels.loc[labels["censor"] == -1, "y_pmic"].median()
    if not med_cens < med_exact:
        raise SystemExit(
            f"CENSORING SIGN LOOKS INVERTED: left-censored median pMIC {med_cens:.3f} "
            f"is not below exact median {med_exact:.3f}. Left-censored rows are the "
            "INACTIVE peptides and must be the less potent group."
        )
    return train_clusters, med_exact, med_cens


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--mic", type=Path, default=DEFAULT_MIC)
    parser.add_argument("--segregated", type=Path, default=DEFAULT_SEGREGATED)
    parser.add_argument("--clstr", type=Path, default=DEFAULT_CLSTR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = load_mic(args.mic)
    raw_seqs = df["sequence"].nunique()
    print(f"{args.mic}: {len(df):,} rows, {raw_seqs:,} sequences, medium MHB")

    df = to_pmic(df)
    df = df[df["species_clean"].isin(HEAD_SPECIES)]
    print(f"  {len(df):,} rows on the {len(HEAD_SPECIES)}-species head panel")

    labels = (
        df.groupby(["sequence", "species_clean"], sort=True)
        .apply(collapse, include_groups=False)
        .reset_index()
        .rename(columns={"species_clean": "species"})
    )
    labels["censor"] = labels["censor"].astype("int8")
    labels["n_rep"] = labels["n_rep"].astype("int32")

    labels = attach_clusters(labels, args.segregated, args.clstr)
    labels = labels[
        ["sequence", "cluster", "split", "species", "y_pmic", "censor", "n_rep"]
    ].sort_values(["sequence", "species"], ignore_index=True)

    train_clusters, med_exact, med_cens = check(labels, raw_seqs)

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "mic_labels.csv"
    labels.to_csv(dest, index=False)

    print(f"\nwrote {len(labels):,} cells -> {dest}")
    print(f"  {labels['sequence'].nunique():,} sequences x {labels['species'].nunique()} species")
    print(f"  splits: {labels.groupby('split')['sequence'].nunique().to_dict()}")
    print(f"  train clusters: {train_clusters}")
    print("\n  censoring (pMIC space):")
    for code, name in [(0, "exact"), (-1, "left  (inactive)"), (+1, "right (potent)")]:
        n = int((labels["censor"] == code).sum())
        print(f"    {code:+d} {name:18s} {n:6,}")
    print(f"  median pMIC  exact {med_exact:.3f}  vs left-censored {med_cens:.3f}  OK")
    print("\n  cells per species:")
    print(labels.groupby("species").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
