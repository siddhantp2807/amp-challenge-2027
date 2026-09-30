"""Peptide dataset with length-bucketed batching.

Median sequence length in both corpora is well under the 50 AA max, so
fixed-padding every batch to 50 would waste most of each batch on PAD tokens
and dilute the per-residue gradient signal (and risks the encoder using
padding structure as a length side-channel in z). Bucketing groups similar
lengths together to avoid that.

Each batch is padded to its *bucket's* fixed upper edge, not to that specific
batch's own max length. This matters beyond just simplicity: padding to each
batch's own max means nearly every batch has a distinct shape (sequences
within a bucket still span a range), and on the MPS backend that means a
distinct compiled kernel graph gets cached per shape with no eviction --
observed in practice to accumulate unboundedly over an epoch and blow past
physical RAM once the decoder got large enough for each cached graph to be
non-trivial, causing severe swap-thrashing partway through an epoch. Padding
to a fixed per-bucket length collapses shape diversity down to `len(edges)`
fixed shapes for the whole epoch, which keeps that cache stable.
"""
import random

import torch
from torch.utils.data import Dataset

from src.data.tokenizer import PAD_ID, VOCAB


class PeptideDataset(Dataset):
    def __init__(self, sequences: list[str]):
        self.sequences = sequences
        self.encoded = [VOCAB.encode(s) for s in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.encoded[idx]


def collate_fn(batch: list[list[int]], pad_to: int | None = None) -> dict[str, torch.Tensor]:
    max_len = pad_to if pad_to is not None else max(len(x) for x in batch)
    tokens = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    for i, ids in enumerate(batch):
        tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    pad_mask = tokens == PAD_ID  # True where padded
    return {"tokens": tokens, "pad_mask": pad_mask}


class BucketBatchSampler:
    """Buckets by raw AA length (dataset stores BOS+seq+EOS, so bucket on
    len-2), shuffles bucket order and within-bucket order each epoch, then
    chunks each bucket into batches of batch_size. Each yielded item is
    `(bucket_idx, indices)` -- callers use `pad_len_for_bucket(bucket_idx)` to
    collate with a fixed, bucket-determined pad length (see module docstring
    for why this matters, not just for padding waste).
    """

    def __init__(
        self,
        sequences: list[str],
        batch_size: int,
        bucket_edges: list[int],
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed

        edges = sorted(bucket_edges)

        def bucket_of(length: int) -> int:
            for i, e in enumerate(edges):
                if length <= e:
                    return i
            return len(edges)

        self.buckets: dict[int, list[int]] = {}
        for idx, seq in enumerate(sequences):
            b = bucket_of(len(seq))
            self.buckets.setdefault(b, []).append(idx)

        # fixed pad-to length per bucket = raw AA edge + BOS/EOS. Sequences
        # past the last edge (shouldn't occur post-cleaning, but handled
        # safely) fall into bucket len(edges), padded to the last edge too.
        self.bucket_pad_len = {i: e + 2 for i, e in enumerate(edges)}
        if edges:
            self.bucket_pad_len[len(edges)] = edges[-1] + 2

    def pad_len_for_bucket(self, bucket_idx: int) -> int:
        return self.bucket_pad_len[bucket_idx]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for bucket_idx, indices in self.buckets.items():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batches.append((bucket_idx, indices[i : i + self.batch_size]))
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return sum(
            (len(idxs) + self.batch_size - 1) // self.batch_size
            for idxs in self.buckets.values()
        )


def bucketed_batches(dataset: PeptideDataset, sampler: BucketBatchSampler):
    """Iterate a dataset via a BucketBatchSampler, collating each batch to its
    bucket's fixed pad length. Use this instead of a plain torch DataLoader
    for bucketed training data -- DataLoader's batch_sampler protocol only
    passes index lists to collate_fn, with no way to tell it which bucket
    (and therefore which fixed pad length) a batch came from.
    """
    for bucket_idx, indices in sampler:
        encoded = [dataset[i] for i in indices]
        yield collate_fn(encoded, pad_to=sampler.pad_len_for_bucket(bucket_idx))
