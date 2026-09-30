"""The long label file as dense per-sequence matrices, plus the CV splits.

The label table is one row per (sequence, species) and is ~31% filled. The model
has one head per species and a masked loss, so what it needs is a dense
(N, J) layout with an observation mask rather than the long form:

    Y (N, J) float32   pMIC, or the BOUND where censored
    M (N, J) bool      cell observed
    C (N, J) int8      0 exact / -1 left / +1 right, already in pMIC space

Splitting is by CD-HIT cluster, not by sequence: two homologous peptides in the
same fold on opposite sides of the split would make the held-out number measure
memorization. All species-rows of a sequence move together, which the (N, J)
layout enforces structurally -- a sequence is one row.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DEFAULT_LABELS = Path("data/processed-data/mic-labels/mic_labels.csv")

# Fixed order. Stored in the bundle and asserted on load -- the heads are indexed
# by position, so a reordering here silently permutes every prediction.
SPECIES = [
    "Escherichia coli",
    "Staphylococcus aureus",
    "Pseudomonas aeruginosa",
    "Klebsiella pneumoniae",
    "Acinetobacter baumannii",
    "Enterococcus faecalis",
]


class LabelMatrix:
    """Dense (N, J) view of the label table for one split."""

    def __init__(self, sequences, clusters, Y, M, C):
        self.sequences = sequences
        self.clusters = clusters
        self.Y, self.M, self.C = Y, M, C

    def __len__(self):
        return len(self.sequences)

    @property
    def n_obs(self):
        return int(self.M.sum())

    def subset(self, idx):
        return LabelMatrix(
            [self.sequences[i] for i in idx],
            self.clusters[idx],
            self.Y[idx],
            self.M[idx],
            self.C[idx],
        )

    def head_counts(self):
        """Observed cells per head -- the input to the 1/sqrt(n_j) rebalancing."""
        return self.M.sum(axis=0)


def load(path=DEFAULT_LABELS, split="train", species=SPECIES):
    df = pd.read_csv(path)
    df = df[df["split"] == split]
    df = df[df["species"].isin(species)]

    seqs = sorted(df["sequence"].unique())
    row = {s: i for i, s in enumerate(seqs)}
    col = {s: j for j, s in enumerate(species)}

    n, j = len(seqs), len(species)
    Y = np.zeros((n, j), dtype=np.float32)
    M = np.zeros((n, j), dtype=bool)
    C = np.zeros((n, j), dtype=np.int8)

    ri = df["sequence"].map(row).to_numpy()
    ci = df["species"].map(col).to_numpy()
    Y[ri, ci] = df["y_pmic"].to_numpy(dtype=np.float32)
    M[ri, ci] = True
    C[ri, ci] = df["censor"].to_numpy(dtype=np.int8)

    clusters = np.zeros(n, dtype=np.int64)
    clusters[ri] = df["cluster"].to_numpy(dtype=np.int64)

    return LabelMatrix(seqs, clusters, Y, M, C)


def folds(data, n_splits=5):
    """Cluster-grouped folds. Yields (train_idx, test_idx) index arrays."""
    idx = np.arange(len(data))
    splitter = GroupKFold(n_splits=n_splits)
    for tr, te in splitter.split(idx, groups=data.clusters):
        yield tr, te


def head_weights(counts):
    """1/sqrt(n_j), normalized to mean 1.

    E. coli has ~5.7x the cells of E. faecalis. Unweighted, the trunk optimizes
    E. coli and the rare heads ride along; 1/sqrt(n) takes that to ~2.4:1, which
    keeps them alive without letting a 330-cell head steer the representation.
    """
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    return (w / w.mean()).astype(np.float32)
