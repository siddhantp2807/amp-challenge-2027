"""AMP Challenge 2027 submission entry point.

    uv run generate

Writes generate/library.fasta (50,000 sequences), generate/top.fasta (the
ranked top 100), and generate/ranking.csv (the selection/ranking record: every
library sequence with each scorer family's score and rank beside the
rank-averaged ensemble). Every argument is defaulted and every stochastic step
is seeded, so repeated runs produce byte-identical files -- the validator checks
this by generating twice and diffing the two FASTAs.

METHOD
-------------------------------------------------------------
Two-stage generative model. A transformer VAE maps peptides to a 64-d latent;
a conditional diffusion model over that latent generates new points conditioned
on (length, charge, GRAVY). Conditions are drawn from the training distribution,
widened so the library reaches sparser regions than the data alone covers.

Candidates are oversampled, then put through the competition's own gates:
canonical alphabet, length 8-50, uniqueness, no exact match to
data/antibacterial.fasta, no degenerate decodes, and a per-cluster cap at 90%
identity so the library is not a few thousand scaffolds wearing 50,000 names.

The top-100 is ranked by the MIC scorer: a rank-averaged
ensemble of per-species gradient boosting and a multi-head Tobit MLP, fitted to
6509 medium-controlled DBAASP MIC measurements across six organisms, aggregated
across organisms with a dispersion penalty so breadth means uniformly active. It
is additionally held to the stricter novelty gate: no candidate may exceed 80%
Levenshtein identity with any sequence in the antibacterial reference set.
"""

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Installed as a console script, so sys.path[0] is .venv/bin -- put the repo
# root on the path so the `src.` modules resolve. Everything else in this file
# is cwd-relative, matching how the validator invokes it.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diffusion.conditioning import PROPERTIES  # noqa: E402
from src.diffusion.latents import load_vae  # noqa: E402
from src.diffusion.metrics import decode_latents  # noqa: E402
from src.diffusion.sample import load_diffusion  # noqa: E402
from src.eval.common import is_degenerate  # noqa: E402
from src.eval.properties import charge_bjellqvist  # noqa: E402
from src.train.utils import get_device  # noqa: E402

DIFFUSION_CKPT = "checkpoints/release/diffusion_target_v1_best.pt"
VAE_CKPT = "checkpoints/release/finetune_fb3p0_lip0p1_v1_best.pt"
SCORER_CKPT = "checkpoints/release/scorer_v1.pt"
ANTIBACTERIAL_FASTA = "data/antibacterial.fasta"

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 8, 50

# Guidance weight. DIFFUSION.md section 6 pinned 2.0 on validation, but that was
# measured with ALL THREE properties active. Under charge-only conditioning (see
# ACTIVE_MASK below) guidance has nothing pulling back against the charge
# direction and over-extrapolates badly: a request for +10 lands at +13.4 at
# cfg 2.0, outside the (+1, +12] band. Measured across cfg in {0, 1, 1.5, 2, 3},
# 1.0 is the value at which requested charge is actually delivered -- +10 -> +10.63,
# with length 28.9 and GRAVY -1.005 against real peptides at charge 9-11 sitting
# at 28.6 / -1.153. This is a different operating point from the validated one
# and is chosen on this measurement, not inherited.
CFG_WEIGHT = 1.0
DDIM_STEPS = 50

# Which properties are requested. The model was trained with per-example random
# condition masking, so a masked property is replaced by a learned mask token and
# the model supplies it from p(property | the active ones). Masking length and
# GRAVY is strictly better than requesting them: asking for the marginal values
# alongside a high charge requests combinations the training data does not
# contain (charge and GRAVY correlate -0.42), which is what drove the +9/+10
# buckets to bleed yield. Letting the model fill them in reproduces the real
# joint distribution -- see the cfg table above.
ACTIVE_MASK = np.array([False, True, False])   # length, charge, gravy

# Charge is swept rather than sampled: eight equally-spaced targets from +3 to
# +10, an equal budget at each. The band is where activity is measurably
# enriched (notebooks/002: the rate rises from ~33% below charge 0 to ~86% at
# 8-10, then turns over), and stopping at +10 avoids the region where guidance
# over-extrapolates -- DIFFUSION.md section 11 records charge bias reaching
# +3.98 at a requested +10 in the reference project.
CHARGE_VALUES = np.arange(3, 11)      # +3, +4, ..., +10

CLUSTER_IDENTITY = 0.90      # library-level redundancy cap
TOP_MAX_IDENTITY = 0.80      # competition gate for the top-100


def read_fasta(path):
    seqs, cur = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
        elif line.strip():
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def write_fasta(sequences, path):
    with open(path, "w") as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f">seq{i}\n{seq}\n")


# Ranking documentation is rounded to the same precision the ranking sort uses,
# so a reader can reproduce the shipped order from the CSV alone.
RANK_DECIMALS = 6


def rank_column(scores):
    """1-based ranks under the ranking sort's exact convention, 1 = best.

    Same round-then-stable-argsort used for the shipped top-100: rounding turns
    last-bit float noise into exact ties, which the stable sort then resolves by
    library order. scipy's rankdata is deliberately NOT used here -- it averages
    ties into .5 ranks, a different convention from the sort this documents.
    """
    order = np.argsort(-np.round(scores, RANK_DECIMALS), kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int64)
    return ranks


def fixed(values, places=RANK_DECIMALS):
    """Fixed-width decimal strings, so pandas never picks a float repr.

    Pre-rendering is what makes the CSV byte-identical across runs and
    platforms. The zero fixup matters: np.round(-1e-9, 6) is -0.0, which would
    otherwise print as "-0.000000" depending on the sign of noise.
    """
    x = np.round(np.asarray(values, dtype=np.float64), places)
    x[x == 0] = 0.0
    return [f"{v:.{places}f}" for v in x]


def ranking_frame(library, order, detail, bucket_of, top):
    """One row per selected library sequence, best ensemble rank first.

    `order` is the same array used to write top.fasta, reused rather than
    recomputed so the two artifacts cannot disagree.
    """
    fam = detail["families"]
    ens = np.asarray(detail["breadth"], dtype=np.float64)
    top_rank = {s: i + 1 for i, s in enumerate(top)}
    n = len(library)
    frame = pd.DataFrame({
        "rank": rank_column(ens),
        "library_index": np.arange(1, n + 1, dtype=np.int64),
        "sequence": list(library),
        "length": np.fromiter((len(s) for s in library), np.int64, n),
        "charge_requested": np.fromiter((bucket_of.get(s, -1) for s in library),
                                        np.int64, n),
        "charge_realized": fixed([charge_bjellqvist(s) for s in library], 4),
        "ensemble_score": fixed(ens),
        "trees_score": fixed(fam["trees"]["breadth"]),
        "trees_rank": rank_column(fam["trees"]["breadth"]),
        "tobit_score": fixed(fam["tobit"]["breadth"]),
        "tobit_rank": rank_column(fam["tobit"]["breadth"]),
        # 1..100 for the sequences that reached top.fasta, empty otherwise. NOT
        # the same as rank <= 100: the 80% identity gate skips candidates, and
        # that gap between rank order and the shipped list is exactly what this
        # file exists to document.
        "top100_rank": [top_rank.get(s, "") for s in library],
    })
    return frame.iloc[order].reset_index(drop=True)


def write_ranking_csv(frame, path):
    frame.to_csv(path, index=False, lineterminator="\n")


def draw_conditions(prop_std, n, charge):
    """Condition rows at a fixed target charge.

    Only the charge column carries meaning; length and GRAVY are masked by
    ACTIVE_MASK and the model supplies them. They are still filled with the
    training mean so the standardised value is exactly 0 -- the embedder
    substitutes a mask token regardless, but a NaN there would be a silent
    hazard.
    """
    c = np.tile(prop_std.mean, (n, 1))
    c[:, 1] = float(charge)
    return c


@torch.no_grad()
def generate_batch(diff, vae, prop_std, z_stats, cond, device, seed, batch=4096):
    """Condition rows -> decoded sequences."""
    out = []
    for i in range(0, len(cond), batch):
        block = cond[i : i + batch]
        c = torch.tensor(prop_std.transform(block), dtype=torch.float32, device=device)
        mask = torch.tensor(np.tile(ACTIVE_MASK, (len(block), 1)),
                            dtype=torch.bool, device=device)
        is_target = torch.ones(len(block), dtype=torch.long, device=device)
        g = torch.Generator().manual_seed(seed + i)
        z = diff.ddim_sample(c, mask, is_target, steps=DDIM_STEPS,
                             cfg_weight=CFG_WEIGHT, eta=0.0, generator=g)
        out.extend(decode_latents(vae, z_stats.denormalize(z.cpu()), device))
    return out


def valid(seq):
    return bool(seq) and MIN_LEN <= len(seq) <= MAX_LEN and set(seq) <= CANONICAL


def cluster_band(length, identity=CLUSTER_IDENTITY):
    """Widest length gap at which a peptide of `length` can still hit `identity`.

    `normalized_distance` is `distance / max(len_a, len_b)`, so a length gap of
    D costs at least D/max_len -- and the divisor is the LONGER sequence, which
    is what sets the bound. Banding from the candidate's own length L, the two
    directions differ:

        against a shorter rep:  max_len = L,      so D <= (1-identity)*L
        against a longer rep:   max_len = L + D,  so D <= L*(1-identity)/identity

    The second is looser, so it governs. At identity=0.90 that is L/9, i.e. a
    band of 0 up to length 8, rising to 5 at length 45-50 -- so this reproduces
    a 2/3/4/5 table over the 8-50 range while staying correct if
    CLUSTER_IDENTITY changes. A literal table would not: at 0.80 the bound is
    L/4, needing a band of 11 by length 45.

    Returning the bound rather than a fixed constant keeps the banding EXACT --
    it only skips pairs that provably cannot reach `identity`.

    Evaluated as an exact rational, not in floating point: at identity=0.90 and
    length=45 the bound is exactly 5, but 45*0.1/0.9 is 4.999... in float64 and
    truncates to 4, silently narrowing the band by one at the lengths where it
    matters most.
    """
    f = Fraction(identity).limit_denominator(10_000)
    return int(length * (1 - f) / f)


def cluster_cap(sequences, reps_by_len, identity=CLUSTER_IDENTITY):
    """Greedy redundancy cap within length bands, order-stable.

    Each sequence is compared against the representatives already accepted and
    dropped if it is `identity` or more similar to any of them; otherwise it
    becomes a representative itself. Banding by `cluster_band` turns an O(n^2)
    screen over the whole library into a handful of small ones without changing
    the result -- see that function for why the band has to grow with length.

    `reps_by_len` is carried across rounds and mutated in place. Re-clustering
    the whole library each round instead would make the top-up loop quadratic
    in the number of rounds for no benefit -- the accepted representatives do
    not change once chosen.

    Compared bucket by bucket on INTEGER distance rather than in one call on
    normalized_distance. `normalized_distance` forces a float comparison against
    `1 - identity`, and that is not decidable at the cutoff: cdist returns
    float32 while `1 - identity` is float64, so NumPy's weak scalar promotion
    demotes the threshold (1 - 0.9 = 0.09999999999999998 becomes 0.1f) and pairs
    sitting exactly on the cutoff are decided by a rounding artifact. Since all
    references in one bucket share a length, `max_len` is fixed per bucket and
    the cutoff becomes an exact integer -- which doubles as a `score_cutoff`,
    letting rapidfuzz abandon hopeless pairs early.
    """
    from rapidfuzz.distance import Levenshtein as RFLev
    from rapidfuzz.process import cdist

    slack = 1 - Fraction(identity).limit_denominator(10_000)
    kept = []
    for s in sequences:
        L = len(s)
        band = cluster_band(L, identity)
        drop = False
        for dl in range(-band, band + 1):
            pool = reps_by_len.get(L + dl)
            if not pool:
                continue
            cutoff = int(slack * max(L, L + dl))   # floor; >= identity iff dist <= cutoff
            d = cdist([s], pool, scorer=RFLev.distance,
                      score_cutoff=cutoff, workers=-1)[0]
            if d.min() <= cutoff:
                drop = True
                break
        if drop:
            continue
        reps_by_len.setdefault(L, []).append(s)
        kept.append(s)
    return kept


def index_by_length(sequences):
    """Group sequences by length, so an edit-distance screen can band by it."""
    idx = {}
    for s in sequences:
        idx.setdefault(len(s), []).append(s)
    return idx


def far_from_references(seq, refs_by_len, d_max):
    """True if `seq` needs MORE than `d_max` edits to reach any reference.

    `d_max` is the rejection radius, not the acceptance floor: with d_max=3 a
    sequence 3 or fewer edits from a reference is rejected, so survivors are at
    least 4 edits away. That matches "filter out edit distance <= 3".

    Two exact prunes make this cheap enough to run on every candidate:

    - Length banding. Edit distance is at least the length difference, so a
      reference whose length differs by more than `d_max` cannot be within
      `d_max`. Only the bands L-d_max .. L+d_max are compared.
    - `score_cutoff=d_max`. rapidfuzz then runs the banded bit-parallel
      Levenshtein and abandons a pair as soon as the distance provably exceeds
      the cutoff, returning d_max+1 instead of the true distance.

    Neither prune changes the answer -- both only skip work that could not have
    produced a hit.
    """
    from rapidfuzz.distance import Levenshtein as RFLev
    from rapidfuzz.process import cdist

    L = len(seq)
    pool = [r for dl in range(-d_max, d_max + 1) for r in refs_by_len.get(L + dl, [])]
    if not pool:
        return True
    d = cdist([seq], pool, scorer=RFLev.distance,
              score_cutoff=d_max, workers=-1)[0]
    return bool(d.min() > d_max)


def screen_edit_distance(sequences, refs_by_len, d_max):
    """`far_from_references` over a list, order-stable.

    Blocked rather than one call per sequence: cdist releases the GIL and
    threads over the pool, and the per-call overhead dominates otherwise. Pools
    are shared within a length band, so candidates are grouped by length.
    """
    from rapidfuzz.distance import Levenshtein as RFLev
    from rapidfuzz.process import cdist

    pools = {}
    for L in {len(s) for s in sequences}:
        pools[L] = [r for dl in range(-d_max, d_max + 1)
                    for r in refs_by_len.get(L + dl, [])]

    survivors = set()
    by_len = index_by_length(sequences)
    for L, block in by_len.items():
        pool = pools[L]
        if not pool:
            survivors.update(block)
            continue
        d = cdist(block, pool, scorer=RFLev.distance,
                  score_cutoff=d_max, workers=-1)
        survivors.update(s for s, row in zip(block, d) if row.min() > d_max)
    return [s for s in sequences if s in survivors]


def screen_identity(candidates, references, max_identity, want):
    """First `want` candidates below `max_identity` against every reference."""
    from rapidfuzz.distance import Levenshtein as RFLev
    from rapidfuzz.process import cdist

    refs = list(references)
    out, step = [], 256
    for i in range(0, len(candidates), step):
        block = candidates[i : i + step]
        sim = cdist(block, refs, scorer=RFLev.normalized_similarity, workers=-1)
        for s, row in zip(block, sim):
            if row.max() <= max_identity:
                out.append(s)
                if len(out) >= want:
                    return out
    return out


def score_detail(sequences):
    """Rank candidates by predicted broad-spectrum potency. Higher is better.

    Backed by the MIC scorer in src/scorer (see SCORER.md): an ensemble of
    per-species gradient boosting and a multi-head Tobit MLP, both on
    physicochemical descriptors, trained on 5,882 MHB MIC measurements from
    DBAASP over six organisms and combined by rank averaging. The Tobit head is
    the only member that uses the 1,749 censored ("no activity up to X")
    measurements, which are the inactive tail a ranker needs for contrast.

    Candidates are ranked on a lower confidence bound across ensemble members,
    then aggregated across organisms with a dispersion penalty -- breadth means
    uniformly active, not high on average.

    This runs on CPU in about 20 seconds for the full library and loads a
    committed checkpoint, so nothing is fitted at generate time and repeated
    runs are byte-identical.

    Returns the full payload, including the per-family fused scores used for the
    ranking CSV. Call it exactly once per run: `descriptor_frame` is an uncached
    per-sequence Python loop and is the dominant cost at 50,000 candidates, and
    the ranks are pool-relative (normalized over whatever pool is passed), so a
    second call on a different pool would produce different numbers.
    """
    from src.scorer.predict import load_scorer

    return load_scorer(SCORER_CKPT).score(sequences, detail=True)


def score(sequences):
    """Breadth only, for callers that want just the ranking signal."""
    return score_detail(sequences)["breadth"]


def main():
    entry_point = Path(sys.argv[0]).stem

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sequences", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-charge", type=int, default=20000,
                        help="raw candidates generated per charge value")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--min-charge", type=int, default=3,
                        help="drop realized charge <= this (exclusive lower bound)")
    parser.add_argument("--max-charge", type=int, default=12,
                        help="drop realized charge > this (inclusive upper bound)")
    parser.add_argument("--min-edits", type=int, default=3,
                        help="drop candidates within this many edits of any "
                             "reference peptide (0 disables; exact matches are "
                             "always dropped)")
    parser.add_argument("--degenerate-filter", action="store_false",
                        help="also drop run-dominated / low-complexity decodes")
    parser.add_argument("--cluster-cap", action="store_false",
                        help="also apply the 90%% identity redundancy cap")
    args = parser.parse_args()

    out_dir = Path(entry_point)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"device: {device}")
    diff, prop_std, z_stats, _ = load_diffusion(DIFFUSION_CKPT, device, use_ema=True)
    vae = load_vae(VAE_CKPT, device)

    antibacterial = set(read_fasta(ANTIBACTERIAL_FASTA))
    refs_by_len = index_by_length(antibacterial)
    print(f"charge targets: {list(CHARGE_VALUES)} | cfg {CFG_WEIGHT} | "
          f"active mask {list(ACTIVE_MASK)} | reference set {len(antibacterial)} sequences"
          f" | min edits {args.min_edits}")

    library, seen, reps_by_len, stats, by_charge = [], set(), {}, [], []
    # Requested charge per sequence, recorded where the resume and fresh paths
    # converge. Not derived from `chosen` later: the shortfall branch never
    # builds it. setdefault so a dedup regression cannot silently relabel.
    bucket_of = {}

    # Sweep charge: an equal raw budget at each target, filtered as we go.
    # Duplicates are tracked globally, so a sequence produced in an earlier
    # bucket is not counted again in a later one.
    # Per-bucket checkpointing. This box has ~8 GB shared between CPU and GPU and
    # the memory manager kills a long MPS job outright -- twice, with no traceback
    # and buffered stdout lost. Writing each bucket as it finishes means a kill
    # costs one bucket instead of the whole run, and a rerun resumes.
    work = out_dir / ".buckets"
    work.mkdir(parents=True, exist_ok=True)

    for k, charge in enumerate(CHARGE_VALUES):
        part = work / f"charge_{int(charge):+03d}.txt"
        if part.exists():
            cached = [s for s in part.read_text().split() if s]
            seen.update(cached)
            # Re-screened on resume rather than trusted: a bucket may have been
            # written by a run with a different (or no) --min-edits, and the
            # screen costs ~1s per bucket. Filtered in memory only -- the file
            # keeps the full bucket, so a later run at a different --min-edits
            # still has everything to work from.
            kept = (screen_edit_distance(cached, refs_by_len, args.min_edits)
                    if args.min_edits > 0 else cached)
            by_charge.append(kept)
            for s in kept:
                bucket_of.setdefault(s, int(charge))
            stats.append({"charge": int(charge), "raw": 0, "valid": 0, "unique": 0,
                          "novel": 0, f"chg_{args.min_charge:g}-{args.max_charge:g}": 0,
                          f"edits>{args.min_edits}": 0, "kept": len(kept)})
            library.extend(kept)
            print(f"  charge +{int(charge):<2}  resumed from {part.name}: {len(kept):,} kept"
                  f"   (total {len(library):,})")
            continue

        cond = draw_conditions(prop_std, args.per_charge, charge=charge)
        raw = generate_batch(diff, vae, prop_std, z_stats, cond, device,
                             seed=args.seed + 1000 * (k + 1), batch=args.batch_size)

        n_valid = 0
        n_uniq = 0
        n_novel = 0
        n_charge = 0
        n_clean = 0
        fresh = []
        for s in raw:
            if not valid(s):                       # canonical alphabet, 8-50
                continue
            n_valid += 1
            if s in seen:                          # duplicate, across all buckets
                continue
            n_uniq += 1
            if s in antibacterial:                 # exact match to the reference set
                continue
            n_novel += 1
            q = charge_bjellqvist(s)               # REALIZED charge, not requested
            if not (args.min_charge < q <= args.max_charge):
                continue
            n_charge += 1
            if args.degenerate_filter and is_degenerate(s):
                continue
            n_clean += 1
            seen.add(s)
            fresh.append(s)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        far = (screen_edit_distance(fresh, refs_by_len, args.min_edits)
               if args.min_edits > 0 else fresh)
        n_far = len(far)

        kept = cluster_cap(far, reps_by_len) if args.cluster_cap else far
        part.write_text("\n".join(kept) + "\n")
        by_charge.append(kept)
        for s in kept:
            bucket_of.setdefault(s, int(charge))
        library.extend(kept)
        stats.append({"charge": int(charge), "raw": len(raw), "valid": n_valid,
                      "unique": n_uniq, "novel": n_novel,
                      f"chg_{args.min_charge:g}-{args.max_charge:g}": n_charge,
                      f"edits>{args.min_edits}": n_far,
                      "kept": len(kept)})
        print(f"  charge +{int(charge):<2}  raw {len(raw):>6,} -> valid {n_valid:>6,}"
              f" -> unique {n_uniq:>6,} -> novel {n_novel:>6,}"
              f" -> in charge band {n_charge:>6,}"
              f" -> >{args.min_edits} edits {n_far:>6,} -> kept {len(kept):>6,}"
              f"   (total {len(library):,})")

    print("\nper-charge funnel:")
    print(pd.DataFrame(stats).to_string(index=False))

    print(f"\n{len(library):,} sequences survived the filters")
    if len(library) < args.n_sequences:
        print(f"WARNING: short of {args.n_sequences:,}. Raise --per-charge and rerun.")
        library = library[: args.n_sequences]
    else:
        # Take an equal share from each charge bucket rather than the first N in
        # bucket order. With generous --per-charge the surplus is large, and a
        # plain head-of-list truncation would drop the last buckets entirely --
        # deleting the high-charge end the sweep exists to reach. Shortfalls in
        # one bucket are redistributed over the buckets that still have surplus.
        target = args.n_sequences // len(by_charge)
        chosen = [b[:target] for b in by_charge]
        deficit = args.n_sequences - sum(len(c) for c in chosen)
        # Round-robin the remainder. The loop terminates on a full pass that
        # added nothing (every bucket exhausted); it must NOT stop on a fixed
        # iteration count -- with several buckets short of `target` the deficit
        # is thousands, and a capped loop silently returns fewer than
        # --n-sequences.
        while deficit > 0:
            progress = False
            for b, c in zip(by_charge, chosen):
                if deficit == 0:
                    break
                if len(c) < len(b):
                    c.append(b[len(c)])
                    deficit -= 1
                    progress = True
            if not progress:                    # nothing left anywhere
                print(f"WARNING: only {args.n_sequences - deficit:,} of "
                      f"{args.n_sequences:,} available after balancing.")
                break
        library = [s for c in chosen for s in c]
        print("  balanced selection per charge bucket: "
              + ", ".join(f"+{s['charge']}:{len(c):,}" for s, c in zip(stats, chosen)))

    library_path = out_dir / "library.fasta"
    write_fasta(library, library_path)
    print(f"\nwrote {len(library):,} sequences -> {library_path}")

    # Rank, then apply the stricter top-100 novelty gate BEFORE selecting, so
    # removals cannot punch holes in the final list.
    # Rounded before sorting: the scorer runs in float32 on CPU, and two
    # candidates whose breadth differs only in the last bits would otherwise be
    # ordered by float noise. Rounding makes those exact ties, which the stable
    # sort then resolves by library order -- itself deterministic.
    detail = score_detail(library)
    order = np.argsort(-np.round(detail["breadth"], RANK_DECIMALS), kind="stable")
    ranked = [library[i] for i in order]
    top = screen_identity(ranked, antibacterial, TOP_MAX_IDENTITY, args.top_k)
    if len(top) < args.top_k:
        raise SystemExit(f"only {len(top)} candidates cleared the "
                         f"{TOP_MAX_IDENTITY:.0%} identity gate")

    top_path = out_dir / "top.fasta"
    write_fasta(top, top_path)
    print(f"wrote top {len(top)} -> {top_path}")

    # Ranking documentation: every selected sequence with each family's score and
    # rank beside the rank-averaged ensemble, in the same order as top.fasta.
    # Written after the two graded artifacts so a failure here cannot cost them.
    unmapped = sum(1 for s in library if s not in bucket_of)
    if unmapped:
        print(f"WARNING: {unmapped:,} sequences have no charge bucket (written as -1)")
    ranking_path = out_dir / "ranking.csv"
    write_ranking_csv(
        ranking_frame(library, order, detail, bucket_of, top), ranking_path)
    print(f"wrote ranking documentation ({len(library):,} rows) -> {ranking_path}")


if __name__ == "__main__":
    main()
