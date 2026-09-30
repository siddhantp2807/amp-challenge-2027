#!/usr/bin/env python3
"""Homology-aware train/val/test split for CD-HIT clustered peptide sets."""

import argparse
import re
from collections import defaultdict

import pandas as pd


def extract_cluster_sizes(clstr_path):
    """Parse a CD-HIT .clstr file into per-cluster member lists and lengths.

    Returns:
        clusters: dict[int, list[str]]  cluster_id -> [seq_id, ...]
        seq_lengths: dict[str, int]     seq_id -> length (aa), from CD-HIT's own annotation
    """
    clusters = defaultdict(list)
    seq_lengths = {}
    cluster_id = None
    member_re = re.compile(r">(\S+?)\.{3}")
    length_re = re.compile(r"(\d+)aa,")

    with open(clstr_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">Cluster"):
                cluster_id = int(line.split()[-1])
                continue
            m_id, m_len = member_re.search(line), length_re.search(line)
            if not m_id or not m_len:
                continue
            seq_id = m_id.group(1)
            clusters[cluster_id].append(seq_id)
            seq_lengths[seq_id] = int(m_len.group(1))

    return dict(clusters), seq_lengths


def check_length_stratification(clusters, seq_lengths, bin_edges=(8, 21, 36, 51)):
    """Compute per-cluster length-bin histograms.

    bin_edges are half-open: default (8,21,36,51) gives bins 8-20, 21-35, 36-50 aa.

    Returns:
        cluster_bin_counts: dict[int, list[int]]  cluster_id -> counts per bin
        overall_bin_counts: list[int]             total sequences per bin
        n_bins: int
    """
    n_bins = len(bin_edges) - 1

    def bin_of(length):
        for i in range(n_bins):
            if bin_edges[i] <= length < bin_edges[i + 1]:
                return i
        return n_bins - 1  # clamp (handles length == top edge)

    cluster_bin_counts = {}
    overall_bin_counts = [0] * n_bins
    for cid, members in clusters.items():
        counts = [0] * n_bins
        for seq_id in members:
            b = bin_of(seq_lengths[seq_id])
            counts[b] += 1
            overall_bin_counts[b] += 1
        cluster_bin_counts[cid] = counts

    return cluster_bin_counts, overall_bin_counts, n_bins


def greedy_bin_pack_split(clusters, seq_lengths, val_frac=0.05, test_frac=0.05,
                           bin_edges=(8, 21, 36, 51)):
    """Assign whole clusters to train/val/test, balancing total sequence count
    and length-bin distribution across splits. Never splits a cluster.

    Returns:
        assignment: dict[str, str]   seq_id -> "train" | "val" | "test"
        summary: dict                per-split sequence counts and per-bin counts
    """
    cluster_bin_counts, overall_bin_counts, n_bins = check_length_stratification(
        clusters, seq_lengths, bin_edges
    )
    total = sum(overall_bin_counts)
    targets = {
        "train": total * (1 - val_frac - test_frac),
        "val": total * val_frac,
        "test": total * test_frac,
    }
    target_bins = {
        split: [targets[split] * (overall_bin_counts[b] / total) for b in range(n_bins)]
        for split in targets
    }

    running_count = {s: 0 for s in targets}
    running_bins = {s: [0] * n_bins for s in targets}
    assignment = {}

    # Largest clusters first: forces the hardest-to-place clusters into the
    # split with the most remaining room, before smaller clusters fill gaps.
    ordered = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
    for cid, members in ordered:
        counts = cluster_bin_counts[cid]

        def deficit(split):
            size_gap = targets[split] - running_count[split]
            bin_gap = sum(target_bins[split][b] - running_bins[split][b] for b in range(n_bins))
            return size_gap + bin_gap

        best_split = max(targets, key=deficit)
        for seq_id in members:
            assignment[seq_id] = best_split
        running_count[best_split] += len(members)
        for b in range(n_bins):
            running_bins[best_split][b] += counts[b]

    summary = {"counts": running_count, "bin_counts": running_bins, "bin_edges": bin_edges}
    return assignment, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clstr", required=True, help="Path to CD-HIT .clstr file")
    parser.add_argument("--val-frac", type=float, default=0.05, required=False)
    parser.add_argument("--test-frac", type=float, default=0.05, required=False)
    parser.add_argument("--out", required=True, help="Output CSV: id,split")
    parser.add_argument("--reference", type=str, help="Path to CSV with reference dataframe(with sequences)")
    args = parser.parse_args()

    clusters, seq_lengths = extract_cluster_sizes(args.clstr)
    assignment, summary = greedy_bin_pack_split(
        clusters, seq_lengths, val_frac=args.val_frac, test_frac=args.test_frac
    )

    split_df = pd.DataFrame(
        sorted(assignment.items()), columns=["id", "split"]
    )

    reference_df = pd.read_csv(args.reference)

    merged_split_df = pd.merge(split_df, reference_df[['id', 'sequence']], on="id")
    merged_split_df.to_csv(args.out, index=False)

    print(f"Split sizes: {summary['counts']}")
    print(f"Per-bin counts (edges {summary['bin_edges']}):")
    for split, counts in summary["bin_counts"].items():
        print(f"  {split}: {counts}")