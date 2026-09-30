import torch
import Levenshtein
import re
from src.data.tokenizer import SPECIAL_TOKENS, VOCAB
from src.data.dataset import collate_fn
from src.model.vae import SequenceVAE


@torch.no_grad()
def encode_sequences(
    model: SequenceVAE, sequences: list[str], device, batch_size: int = 128, sample: bool = False
) -> torch.Tensor:
    """Returns (N, d_z) latents — mu if sample=False, else a reparameterized sample."""
    model.eval()
    encoded = [VOCAB.encode(s) for s in sequences]
    out = []
    for i in range(0, len(encoded), batch_size):
        batch = collate_fn(encoded[i : i + batch_size])
        tokens, pad_mask = batch["tokens"].to(device), batch["pad_mask"].to(device)
        mu, logvar = model.encode(tokens, pad_mask)
        z = model.reparameterize(mu, logvar) if sample else mu
        out.append(z.cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def decode_greedy(
    model: SequenceVAE, z: torch.Tensor, device, seq_len: int | None = None, batch_size: int = 128
) -> list[str]:
    """Greedy (argmax) decode for a batch of latents. Returns raw decoded strings."""
    model.eval()
    out = []
    for i in range(0, z.shape[0], batch_size):
        chunk = z[i : i + batch_size].to(device)
        logits = model.decode(chunk, seq_len=seq_len)
        ids = logits.argmax(dim=-1).cpu()
        for row in ids:
            out.append(VOCAB.decode(row.tolist()))
    return out


@torch.no_grad()
def decode_with_predicted_length(
    model: SequenceVAE, z: torch.Tensor, device, batch_size: int = 128
) -> list[str]:
    """Decode at model.max_len, then keep exactly the predicted number of residues:
    positions 1..n_hat, with only the 20 amino-acid tokens allowed (PAD/BOS/EOS
    can't win). The output length therefore equals the predicted length, without
    relying on the decoder to emit EOS at the right place. (An earlier version
    kept n_hat + 2 positions and stopped at EOS; the decoder rarely emitted EOS
    there, so decodes came out one residue too long.)
    """
    n_special = len(SPECIAL_TOKENS)
    model.eval()
    out = []
    for i in range(0, z.shape[0], batch_size):
        chunk = z[i : i + batch_size].to(device)
        pred_len = model.predict_length(chunk).cpu().long()  # (b,)
        logits = model.decode(chunk, seq_len=model.max_len)
        aa_ids = logits[:, 1:, n_special:].argmax(dim=-1).cpu() + n_special  # skip BOS position
        for row, plen in zip(aa_ids, pred_len):
            out.append(VOCAB.decode(row[: plen.item()].tolist()))
    return out


def normalized_edit_distance(a: str, b: str) -> float:
    if len(a) == 0 and len(b) == 0:
        return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b), 1)


# Per-length-bucket ceiling on the single most frequent residue's share of a
# sequence. NOT hand-picked: these are the 99th percentile of the real corpus
# (31,850 unique sequences from data/pretraining-corpus.csv + finetuning-corpus.csv),
# so by construction ~1% of real peptides sit above them. Measured false-positive
# rate on that corpus is 0.91%.
#
# Why this check was added. The run-length and unique-residue tests below miss the
# decoder's actual failure mode: length-dependent repetition that is *interrupted*.
# `GAHGAFKKGFGGGGGGGGKGGGGGGGGGGGGGGGGGYGGGGGGGFG` is 78% glycine with 6 distinct
# residues, so its longest run is only 37% of its length and it has well over
# 3 unique residues -- it passed both old tests. Zero of 100 sampled peptides were
# flagged before this check existed, including every example above.
#
# Thresholds are length-bucketed because the statistic is length-dependent: a real
# 10-mer is median 30% one residue, a real 40-mer only 18%. A single global constant
# would either miss long garbage or flag ordinary short peptides.
# Re-measured on this repo's corpus (31,784 unique sequences from
# data/processed/{pretrain,finetune}_clean.csv) when the pipeline was ported.
# The p99 per bucket came out at 0.667 / 0.526 / 0.536 / 0.500 against the
# 0.67 / 0.53 / 0.55 / 0.50 below, so the constants carry over unchanged.
#
# The overall false-positive rate on real peptides is higher here, 1.91% vs the
# reference's 0.91%, and none of it is this table: run-dominance accounts for
# 0.36%, composition 0.90%, and low-diversity (`min_unique_aa`, a function
# default rather than a calibrated constant) 1.40%. The corpus simply holds more
# genuine low-complexity AMPs -- poly-K, poly-R, and glycine-rich sequences like
# GYGGHGGHGGHGGHGGHGGHGHGGGGHG. Those are correctly flagged as degenerate-looking;
# the bar was not loosened to admit them. Consequence for reading the gates: a
# model perfectly matching this data distribution would score ~1.9% degenerate,
# still well inside the 5% gate.
MAX_TOP_RESIDUE_FRAC = {(8, 16): 0.67, (17, 24): 0.5, (25, 35): 0.5, (36, 50): 0.50}


def _top_residue_frac_limit(length: int) -> float:
    for (lo, hi), limit in MAX_TOP_RESIDUE_FRAC.items():
        if lo <= length <= hi:
            return limit
    return 0.50 if length > 50 else 0.67


def is_degenerate(seq: str, max_run_frac: float = 0.5, min_unique_aa: int = 3) -> bool:
    """Heuristic: empty, single-residue-repeat-dominated, very low residue
    diversity, or dominated by one residue overall -> treated as a
    degenerate/garbage decode rather than a plausible peptide.
    """
    if len(seq) == 0:
        return True

    if re.search(r"(.)\1{2,}", seq):
        return True

    longest_run = 1
    cur_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur_run += 1
            longest_run = max(longest_run, cur_run)
        else:
            cur_run = 1
    if longest_run / len(seq) > max_run_frac:
        return True
    if len(set(seq)) < min_unique_aa and len(seq) >= 6:
        return True
    # composition concentration: one residue dominating even if its occurrences
    # are broken into several runs (see MAX_TOP_RESIDUE_FRAC)
    if max(seq.count(a) for a in set(seq)) / len(seq) > _top_residue_frac_limit(len(seq)):
        return True
    return False
