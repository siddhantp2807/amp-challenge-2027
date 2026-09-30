"""Stage-1 data cleaning: canonical filter, length filter, dedup, column-select.

Mechanically enforces "sequence-only, no biophysical conditioning" by dropping
every column except id/sequence(/split) at the pipeline boundary, rather than
leaving that as a convention downstream code has to remember.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import yaml


def clean_sequences(
    df: pd.DataFrame,
    min_length: int,
    max_length: int,
    canonical_regex: str,
    keep_columns: list[str],
    dedup_within_split: bool = False,
) -> pd.DataFrame:
    pattern = re.compile(canonical_regex)
    df = df.copy()

    mask = df["sequence"].str.match(pattern)
    df = df[mask]

    lengths = df["sequence"].str.len()
    df = df[(lengths >= min_length) & (lengths <= max_length)]

    if dedup_within_split:
        # never let a dedup collapse merge a val/test row into train: dedup
        # independently per split, then assert no exact duplicate crosses splits
        parts = [g.drop_duplicates(subset="sequence") for _, g in df.groupby("split")]
        df = pd.concat(parts, ignore_index=True)
        dup_across = df["sequence"].duplicated(keep=False)
        assert not dup_across.any(), (
            f"{dup_across.sum()} sequences duplicated across splits after dedup"
        )
    else:
        df = df.drop_duplicates(subset="sequence")

    return df[keep_columns].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/data.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    pretrain_raw = pd.read_csv(cfg["pretrain_raw"])
    pretrain_clean = clean_sequences(
        pretrain_raw,
        min_length=cfg["min_length"],
        max_length=cfg["max_length"],
        canonical_regex=cfg["canonical_regex"],
        keep_columns=["id", "sequence"],
        dedup_within_split=False,
    )
    Path(cfg["pretrain_clean"]).parent.mkdir(parents=True, exist_ok=True)
    pretrain_clean.to_csv(cfg["pretrain_clean"], index=False)
    print(f"pretrain: {len(pretrain_raw)} -> {len(pretrain_clean)} sequences")

    finetune_raw = pd.read_csv(cfg["finetune_raw"])
    finetune_clean = clean_sequences(
        finetune_raw,
        min_length=cfg["min_length"],
        max_length=cfg["max_length"],
        canonical_regex=cfg["canonical_regex"],
        keep_columns=["id", "sequence", "split"],
        dedup_within_split=True,
    )
    finetune_clean.to_csv(cfg["finetune_clean"], index=False)
    print(f"finetune: {len(finetune_raw)} -> {len(finetune_clean)} sequences")
    print(finetune_clean["split"].value_counts())


if __name__ == "__main__":
    main()
