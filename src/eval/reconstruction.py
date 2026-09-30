"""§5.1 Reconstruction fidelity at the residue level.

Aggregate loss can hide charge/helicity-disrupting substitutions concentrated
in specific length ranges, so this reports exact-match rate and mean edit
distance both in aggregate and broken out by length bucket.
"""
import numpy as np
import torch

from src.eval.common import decode_with_predicted_length, encode_sequences, normalized_edit_distance

LENGTH_BUCKETS = [(8, 16), (17, 24), (25, 35), (36, 50)]


def bucket_of(length: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= length <= hi:
            return f"{lo}-{hi}"
    return "out_of_range"


def reconstruction_fidelity(model, sequences: list[str], device) -> dict:
    mu = encode_sequences(model, sequences, device, sample=False)
    decoded = decode_with_predicted_length(model, mu, device)

    with torch.no_grad():
        pred_len = model.predict_length(mu.to(device)).cpu().numpy()
    true_len = np.array([len(s) for s in sequences])

    exact = [d == s for s, d in zip(sequences, decoded)]
    edit_dist = [normalized_edit_distance(s, d) for s, d in zip(sequences, decoded)]

    by_bucket: dict[str, dict] = {}
    for s, e, ed in zip(sequences, exact, edit_dist):
        b = bucket_of(len(s))
        by_bucket.setdefault(b, {"exact": [], "edit": []})
        by_bucket[b]["exact"].append(e)
        by_bucket[b]["edit"].append(ed)

    result = {
        "n": len(sequences),
        "exact_match_rate": float(np.mean(exact)),
        "mean_edit_distance": float(np.mean(edit_dist)),
        # exact match can never exceed length_exact_rate, so report it next to it
        "length_exact_rate": float(np.mean(pred_len == true_len)),
        "length_mae": float(np.mean(np.abs(pred_len - true_len))),
        "by_length_bucket": {
            b: {
                "n": len(v["exact"]),
                "exact_match_rate": float(np.mean(v["exact"])),
                "mean_edit_distance": float(np.mean(v["edit"])),
            }
            for b, v in sorted(by_bucket.items())
        },
    }
    return result
