"""Build the stage-2 conditioning labels from the cleaned corpora.

Writes data/ld-processed/{pretraining,finetuning}.csv with the columns
src/diffusion/conditioning.py expects: sequence, charge, length,
hydrophobic_moment, gravy.

Sources, deliberately the external libraries rather than src/eval/properties.py:
    charge             peptidy.descriptors.charge(seq, pH=7)
    gravy              modlamp PeptideDescriptor(seqs, 'gravy').calculate_global()
    hydrophobic_moment modlamp PeptideDescriptor(seqs, 'eisenberg').calculate_moment(
                           window=1000, angle=100, modality='max')

properties.py reimplements all three so generated sequences can be scored on the
same definitions the labels use, and tests/test_ld_properties.py pins the two
against each other. Generating the labels from the libraries is what keeps that
test a real cross-check instead of a tautology.

Note on modlamp scales: its 'gravy' scale is the raw Kyte-Doolittle table
(verified identical on all 20 residues). Its similarly-named 'kytedoolittle'
scale is a *normalized* variant spanning roughly [-1.3, 1.7] and is NOT GRAVY --
using it would silently condition on a different quantity.

Reads the cleaned corpora, not the segregated CSVs: load_conditions() joins on
sequence and asserts the merge drops no rows, so labels must be built after
clean_corpus.py has deduplicated.

Run: uv run python -m src.data.build_ld_labels
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from modlamp.descriptors import PeptideDescriptor
from peptidy.descriptors import charge as peptidy_charge

OUT_DIR = Path("data/ld-processed")
COLUMNS = ["sequence", "charge", "length", "hydrophobic_moment", "gravy"]


def compute_labels(sequences: list[str]) -> pd.DataFrame:
    gravy = PeptideDescriptor(sequences, "gravy")
    gravy.calculate_global()

    moment = PeptideDescriptor(sequences, "eisenberg")
    moment.calculate_moment(window=1000, angle=100, modality="max")

    return pd.DataFrame(
        {
            "sequence": sequences,
            "charge": [peptidy_charge(s, pH=7) for s in sequences],
            "length": [len(s) for s in sequences],
            "hydrophobic_moment": np.asarray(moment.descriptor).ravel(),
            "gravy": np.asarray(gravy.descriptor).ravel(),
        }
    )[COLUMNS]


def build(clean_csv: str, out_name: str) -> None:
    sequences = pd.read_csv(clean_csv)["sequence"].tolist()
    df = compute_labels(sequences)

    assert len(df) == len(sequences), "row count changed while computing labels"
    assert df["sequence"].is_unique, (
        f"{clean_csv} contains duplicate sequences; conditioning joins on sequence, "
        "so clean_corpus.py must run first"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{out_name}.csv"
    df.to_csv(out, index=False)
    print(f"{out_name}: {len(df)} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/data.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    build(cfg["pretrain_clean"], "pretraining")
    build(cfg["finetune_clean"], "finetuning")


if __name__ == "__main__":
    main()
