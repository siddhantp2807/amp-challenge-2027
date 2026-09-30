#!/usr/bin/env python3
"""Execute every command documented in docs/USAGE.md and report pass/fail.

The point of this script is that "every documented command is tested" stays true after the code
changes, instead of being true only on the day the docs were written. `docs/USAGE.md` §9 is
generated from `--emit-table`.

Tiers
-----
A   executed for real, at full settings
B   executed through a tiny config in config/smoke/ -- real code path, seconds instead of hours
C   documented but deliberately NOT executed (held-out test split, live API scrapers, ...)

Tier C is not representable here: those commands are absent from CASES by construction, so no
flag can cause this script to spend the held-out test split or re-scrape DBAASP.

Clean room (--clean-room, the default for tier A)
------------------------------------------------
Copies the *intended public fileset* -- tracked files plus untracked-but-not-ignored ones -- into
a temporary directory and runs there. Two reasons:

1. It proves the documented commands work for someone who cloned the repo, rather than for someone
   sitting in this working tree. A command that silently depends on a gitignored file (a top-level
   `checkpoints/*.pt`, `reports/`, a `scripts/*.sh`) fails here, which is the whole point.
2. Nothing it does can touch the real `generate/`, `reports/` or `checkpoints/`. The shipped
   submission artifacts are safe without needing a backup-and-restore dance.

Usage
-----
    uv run python scripts/test_usage.py --list
    uv run python scripts/test_usage.py --tier a --clean-room
    uv run python scripts/test_usage.py --tier all --clean-room --include-slow
    uv run python scripts/test_usage.py --tier a --clean-room --emit-table
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Checkpoints live under checkpoints/release/ in the public repo; the bare checkpoints/*.pt paths
# that several CLIs default to are gitignored, so every case names the release copy explicitly.
VAE_CKPT = "checkpoints/release/finetune_fb3p0_lip0p1_v1_best.pt"
DIFF_CKPT = "checkpoints/release/diffusion_target_v1_best.pt"
SCORER_CKPT = "checkpoints/release/scorer_v1.pt"


def case(cid, tier, argv, *, produces=(), expect=(), forbid=(), counts=(), rm=(),
         slow=False, timeout=3600, note=""):
    """One documented command plus what must be true after it runs.

    produces  paths that must exist afterwards
    expect    substrings that must appear in stdout+stderr
    forbid    substrings that must NOT appear -- catches commands that warn and exit 0
    counts    (path, n) pairs: the FASTA must hold exactly n records
    rm        paths deleted before the command runs, e.g. a stale resume cache
    """
    return dict(id=cid, tier=tier, argv=argv, produces=list(produces), expect=list(expect),
                forbid=list(forbid), counts=list(counts), rm=list(rm),
                slow=slow, timeout=timeout, note=note)


# Ordered: later cases may consume what earlier ones produce.
CASES = [
    # ---- setup ---------------------------------------------------------------
    case("sync", "A", ["uv", "sync"],
         produces=[".venv/bin/generate"],
         note="installs the project; the console script must appear or [project.scripts] is broken"),

    # ---- data pipeline stage 1: DBAASP, from the committed archives ----------
    # 01 and 03 are the live-API steps and are tier C. The zips are the reproducibility
    # guarantee, so everything downstream starts by unzipping them.
    case("unzip-dbaasp-list", "A",
         ["unzip", "-o", "-q", "data/raw-data/dbaasp/list_pages.zip",
          "-d", "data/raw-data/dbaasp/"],
         produces=["data/raw-data/dbaasp/list_pages"]),
    case("unzip-dbaasp-details", "A",
         ["unzip", "-o", "-q", "data/raw-data/dbaasp/details.zip",
          "-d", "data/raw-data/dbaasp/"],
         produces=["data/raw-data/dbaasp/details"], slow=True, timeout=1800,
         note="50.9 MB archive"),
    # Explicit paths, not defaults: dbaasp_common.py sets DEFAULT_RAW_DIR to
    # "../../data/raw-data/dbaasp", which is relative to scripts/dbaasp/, so the defaults
    # resolve outside the repo when run from the root as everything else is.
    case("dbaasp-02", "A",
         ["uv", "run", "python", "scripts/dbaasp/02-build-peptides-csv.py",
          "--list-pages-dir", "data/raw-data/dbaasp/list_pages",
          "--out", "data/raw-data/dbaasp/dbaasp-peptides-compiled.csv"],
         produces=["data/raw-data/dbaasp/dbaasp-peptides-compiled.csv"], timeout=1800),
    case("dbaasp-04", "A",
         ["uv", "run", "python", "scripts/dbaasp/04-build-peptide-activity-hc-df.py",
          "--details_dir", "data/raw-data/dbaasp/details",
          "--out_activity_csv", "data/raw-data/dbaasp/dbaasp-activity.csv",
          "--out_hc_csv", "data/raw-data/dbaasp/dbaasp-hc.csv"],
         produces=["data/raw-data/dbaasp/dbaasp-activity.csv"], slow=True, timeout=3600,
         note="note the underscore flags here, unlike the hyphens elsewhere"),
    case("dbaasp-05", "A",
         ["uv", "run", "python", "scripts/dbaasp/05-standardize-mic.py"],
         produces=["data/processed-data/dbaasp-std/filtered-dbaasp.csv"], timeout=1800,
         note="this script's defaults ARE repo-root-relative, unlike 02's"),

    # ---- data pipeline stage 3: cluster and segregate ------------------------
    # Stage 2 (prepare-finetune / prepare-pretrain) is tier C: GRAMPA's CSV and MarlysAMP's
    # JSON are gitignored, so a clone cannot run it without re-downloading those sources.
    # Its outputs under data/processed-data/clean_and_combine/ ARE committed, so stage 3 runs.
    case("cdhit-split", "A",
         ["uv", "run", "python", "scripts/cluster_and_segregate/cdhit-split.py",
          "--clstr", "data/processed-data/cluster_and_segregate/cd-hit-output/finetune50_output.clstr",
          "--reference", "data/processed-data/clean_and_combine/finetune.csv",
          "--out", "finetune_segregated_check.csv"],
         produces=["finetune_segregated_check.csv"], timeout=1800),
    case("filter-pretrain", "A",
         ["uv", "run", "python", "scripts/cluster_and_segregate/filter-pretrain.py",
          "--cluster", "data/processed-data/cluster_and_segregate/cd-hit-output/pretrain50_output.clstr",
          "--pretrain", "data/processed-data/clean_and_combine/pretrain.csv",
          "--finetune", "data/processed-data/cluster_and_segregate/segregate/finetune_segregated.csv",
          "--out", "pretrain_segregated_check.csv"],
         produces=["pretrain_segregated_check.csv"], timeout=1800),
    case("mic-labels", "A",
         ["uv", "run", "python", "scripts/mic/01-build-mic-labels.py",
          "--out", "mic_labels_check.csv"],
         produces=["mic_labels_check.csv"], timeout=1800,
         note="writes to a check path so the committed mic_labels.csv is left alone"),

    # ---- data preparation ----------------------------------------------------
    # Must precede tests-ld: data/processed/ and data/ld-processed/ are gitignored, so in a fresh
    # clone they do not exist and test_ld_properties.py asserts "no label files under
    # data/ld-processed". Found by running this harness in the clean room.
    case("clean-corpus", "A", ["uv", "run", "python", "-m", "src.data.clean_corpus"],
         produces=["data/processed"],
         note="rebuilds the gitignored data/processed/ from the committed data/processed-data/"),
    case("build-ld-labels", "A", ["uv", "run", "python", "-m", "src.data.build_ld_labels"],
         produces=["data/ld-processed"],
         note="prerequisite for tests/test_ld_properties.py"),

    # ---- tests ---------------------------------------------------------------
    case("tests-censoring", "A", ["uv", "run", "python", "tests/test_censoring_sign.py"],
         note="self-contained; runs on a fresh clone"),
    case("tests-ld", "A", ["uv", "run", "python", "tests/test_ld_properties.py"],
         note="requires src.data.build_ld_labels to have run first"),

    # ---- scorer (stage 3) ----------------------------------------------------
    case("scorer-report", "A",
         ["uv", "run", "python", "-m", "src.scorer.report", "--bundle", SCORER_CKPT],
         produces=["reports/scorer_report.json"], expect=["weighted"], timeout=5400, slow=True,
         note="exits non-zero on drift, so this is both a documented command and a regression test"),
    case("scorer-baseline", "A", ["uv", "run", "python", "-m", "src.scorer.baseline"],
         produces=["reports/scorer_f0.json"], slow=True, timeout=5400),

    case("scorer-export", "A",
         ["uv", "run", "python", "-m", "src.scorer.export", "--out", "scorer_export_check.pt"],
         produces=["scorer_export_check.pt"], slow=True, timeout=5400,
         note="--out redirected so the released bundle is never overwritten"),
    case("scorer-train-smoke", "B",
         ["uv", "run", "python", "-m", "src.scorer.train", "--features", "f1",
          "--folds", "2", "--seeds", "0", "--tag", "smoke"],
         timeout=3600, note="the real trainer at 2 folds / 1 seed instead of 5 / 3"),

    # ---- VAE (stage 1) -------------------------------------------------------
    case("freeze-report-val", "A",
         ["uv", "run", "python", "-m", "src.eval.freeze_report",
          "--checkpoint", VAE_CKPT, "--split", "val"],
         slow=True, timeout=5400,
         note="--split val only. test is a one-shot decision and is tier C"),
    case("compare-checkpoints", "A",
         ["uv", "run", "python", "-m", "src.eval.compare_checkpoints", VAE_CKPT],
         timeout=3600, note="cheaper than a full freeze report"),

    # ---- diffusion (stage 2) -------------------------------------------------
    case("diffusion-preflight", "A",
         ["uv", "run", "python", "-m", "src.diffusion.preflight",
          "--vae-checkpoint", VAE_CKPT],
         produces=["reports/diffusion_preflight.json"], slow=True, timeout=5400,
         note="the --vae-checkpoint default points at a gitignored checkpoints/*.pt; must override"),
    case("diffusion-sample", "A",
         ["uv", "run", "python", "-m", "src.diffusion.sample",
          "--checkpoint", DIFF_CKPT, "--vae-checkpoint", VAE_CKPT,
          "--n", "64", "--cfg-weight", "2.0", "--out-fasta", "smoke_sample.fasta"],
         produces=["smoke_sample.fasta"], timeout=1800),
    # diffusion_report resolves the VAE from the diffusion checkpoint payload
    # (src/eval/diffusion_report.py:111) and has NO --vae-checkpoint override, unlike
    # src/diffusion/sample.py:89. The baked path is the bare checkpoints/*.pt, which is
    # gitignored, so from a clone this dies with FileNotFoundError. The copy below is the
    # workaround, and it is what the documentation tells the reader to do.
    case("diffusion-report-vae-shim", "A",
         ["cp", "checkpoints/release/finetune_fb3p0_lip0p1_v1_best.pt",
          "checkpoints/finetune_fb3p0_lip0p1_v1_best.pt"],
         produces=["checkpoints/finetune_fb3p0_lip0p1_v1_best.pt"],
         note="workaround for the missing --vae-checkpoint override in diffusion_report"),
    case("diffusion-report-val", "A",
         ["uv", "run", "python", "-m", "src.eval.diffusion_report",
          "--checkpoint", DIFF_CKPT, "--split", "val", "--cfg-weights", "2.0"],
         slow=True, timeout=5400,
         note="needs the shim above; module default is --split val (safe), while the "
              "gitignored wrapper defaulted to test with a full cfg sweep"),

    # ---- the submission ------------------------------------------------------
    # generate/.buckets is a resume cache keyed on nothing: a bucket file on disk is reused even
    # when --per-charge, the seed or the guidance weight changed. Without the rm below, the full
    # run silently reuses the smoke run's 500-per-charge buckets, delivers ~1.4k sequences instead
    # of 50,000, prints "WARNING: short of" and still exits 0. Caught by this harness.
    # --per-charge 1200, not 500. Only ~36% of raw decodes survive the seven gates, so 500 per
    # charge (4,000 raw) yields ~1,400 -- short of 2,400. generate then prints
    # "WARNING: short of 2,400", writes the short library and exits 0.
    case("generate-smoke", "A",
         ["uv", "run", "generate", "--n-sequences", "2400", "--per-charge", "1200"],
         rm=["generate/.buckets"],
         produces=["generate/library.fasta", "generate/top.fasta"],
         counts=[("generate/library.fasta", 2400), ("generate/top.fasta", 100)],
         forbid=["WARNING: short of"], timeout=2400,
         note="fast path; writes into the clean room's own generate/, never the real one"),
    case("generate-full", "A", ["uv", "run", "generate"],
         rm=["generate/.buckets"],
         produces=["generate/library.fasta", "generate/top.fasta"],
         counts=[("generate/library.fasta", 50000), ("generate/top.fasta", 100)],
         forbid=["WARNING: short of"],
         slow=True, timeout=7200,
         note="the shipped defaults (seed 72); must deliver exactly 50,000"),

    # ---- tier B: training, via smoke configs --------------------------------
    case("vae-pretrain-smoke", "B",
         ["uv", "run", "python", "-m", "src.train.pretrain",
          "--train-config", "config/smoke/train_pretrain.yaml"],
         timeout=1800, note="real trainer, 1 epoch on a subsample"),
    case("vae-finetune-smoke", "B",
         ["uv", "run", "python", "-m", "src.train.finetune",
          "--train-config", "config/smoke/train_finetune.yaml"],
         timeout=1800),
    case("diffusion-train-smoke", "B",
         ["uv", "run", "python", "-m", "src.diffusion.train",
          "--train-config", "config/smoke/train_diffusion.yaml"],
         timeout=1800),
]

# Commands that are documented but must never run from here. Listed so --list can show the whole
# documented surface, and so a reader can see nothing was quietly skipped.
TIER_C = [
    ("scripts/clean_and_combine/prepare-finetune.py",
     "GRAMPA csv and the DRAMP directory are gitignored; re-download them first (see USAGE.md §3)"),
    ("scripts/clean_and_combine/prepare-pretrain.py",
     "the MarlysAMP JSON is gitignored; re-download it first (see USAGE.md §3)"),
    ("src.eval.freeze_report --split test",
     "held-out test split is unspent; one-shot freeze decision"),
    ("src.eval.diffusion_report --split test",
     "same split; if ever spent, pass --cfg-weights 2.0 once, not the default sweep"),
    ("src.freeze.export_frozen",
     "must never run: a frozen VAE double-normalizes silently"),
    ("scripts/dbaasp/01-fetch-list-pages.py",
     "live API; re-scraping yields a different snapshot. unzip list_pages.zip instead"),
    ("scripts/dbaasp/03-fetch-peptide-details.py",
     "live API; unzip details.zip and resume from 02"),
    ("scripts/verify_submission.py <url>",
     "needs a pushed public repo; clones and generates twice (~16+ min)"),
]


def check_doc(repo: Path) -> int:
    """Every fenced command in docs/USAGE.md must be a case here or listed in TIER_C.

    This is what stops the documentation and the harness drifting apart: add a command to the doc
    without adding it here (or declaring it tier C) and this fails.
    """
    import re

    doc = (repo / "docs" / "USAGE.md").read_text()
    body = doc[: doc.index("## 9. Verification evidence")]
    skip = ("ls ", "git ", "rm ", "cp ", "unzip", "uv add", ">", "#")
    cmds = []
    for block in re.findall(r"```bash\n(.*?)```", body, re.S):
        for line in re.sub(r"\\\n\s*", " ", block).splitlines():
            line = re.sub(r"\s+#.*$", "", line).strip()
            if line and not line.startswith(skip) and "scripts/test_usage.py" not in line:
                cmds.append(line)

    harness = " ".join(" ".join(c["argv"]) for c in CASES)
    tierc = " ".join(c for c, _ in TIER_C)

    def key(c):
        m = re.search(r"-m (src\.[\w.]+)", c) or re.search(r"(scripts/[\w/.\-]+\.py)", c)
        if m:
            return m.group(1)
        if c.startswith("uv sync"):
            return "uv sync"
        return "generate" if "uv run generate" in c else c

    missing = [c for c in cmds
               if key(c) not in harness and key(c) not in tierc
               and key(c).replace("src.", "") not in tierc]
    print(f"docs/USAGE.md: {len(cmds)} fenced commands")
    for c in missing:
        print(f"  UNCOVERED: {c[:100]}")
    print("  all covered by a case or TIER_C" if not missing else f"  {len(missing)} uncovered")
    return 1 if missing else 0


def fileset(repo: Path) -> list[str]:
    """Tracked + untracked-but-not-ignored = what the public repo will contain."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True,
                              check=True).stdout.splitlines()

    tracked = git("ls-files")
    untracked = [l[3:] for l in git("status", "--porcelain", "--untracked-files=all")
                 if l.startswith("??")]
    out = []
    for p in sorted(set(tracked) | set(untracked)):
        if "__pycache__" in p or p.endswith(".bkp") or p.startswith(".$"):
            continue
        if (repo / p).is_file():
            out.append(p)
    return out


def build_clean_room(repo: Path, dest: Path) -> int:
    files = fileset(repo)
    for rel in files:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, target)
    return len(files)


def run_case(c: dict, cwd: Path) -> dict:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    for rel in c["rm"]:
        target = cwd / rel
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()
    t0 = time.time()
    try:
        p = subprocess.run(c["argv"], cwd=cwd, env=env, capture_output=True, text=True,
                           timeout=c["timeout"])
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return dict(c, ok=False, secs=time.time() - t0, why=f"timeout after {c['timeout']}s")

    secs = time.time() - t0
    if rc != 0:
        tail = (err or out).strip().splitlines()[-3:]
        return dict(c, ok=False, secs=secs, why=f"exit {rc}: " + " | ".join(tail))

    for rel in c["produces"]:
        if not (cwd / rel).exists():
            return dict(c, ok=False, secs=secs, why=f"did not produce {rel}")
    blob = out + err
    for needle in c["expect"]:
        if needle not in blob:
            return dict(c, ok=False, secs=secs, why=f"output missing {needle!r}")
    for needle in c["forbid"]:
        if needle in blob:
            return dict(c, ok=False, secs=secs, why=f"output contains {needle!r}")
    for rel, want in c["counts"]:
        got = sum(1 for line in (cwd / rel).read_text().splitlines() if line.startswith(">"))
        if got != want:
            return dict(c, ok=False, secs=secs, why=f"{rel}: {got} records, expected {want}")
    return dict(c, ok=True, secs=secs, why="")


def emit_table(results: list[dict], clean_room: bool) -> str:
    host = f"{platform.system()} {platform.machine()}, Python {platform.python_version()}"
    lines = [
        f"Verified {date.today().isoformat()} on {host}"
        f"{' in a clean room (public fileset only)' if clean_room else ' in the working tree'}.",
        "",
        "| command | tier | result | runtime |",
        "|---|---|---|---|",
    ]
    for r in results:
        cmd = " ".join(r["argv"])
        mark = "pass" if r["ok"] else f"**FAIL** — {r['why']}"
        lines.append(f"| `{cmd}` | {r['tier']} | {mark} | {r['secs']:.0f}s |")
    lines += ["", "| documented, deliberately not executed | reason |", "|---|---|"]
    for cmd, why in TIER_C:
        lines.append(f"| `{cmd}` | {why} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=("a", "b", "all"), default="a")
    ap.add_argument("--only", nargs="*", help="case ids to run")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean-room", action="store_true",
                    help="run against a copy of the intended public fileset (recommended)")
    ap.add_argument("--keep", action="store_true", help="do not delete the clean room")
    ap.add_argument("--include-slow", action="store_true",
                    help="also run the multi-minute cases (full generate, reports, preflight)")
    ap.add_argument("--emit-table", action="store_true", help="print the docs/USAGE.md table")
    ap.add_argument("--check-doc", action="store_true",
                    help="only verify docs/USAGE.md and this file agree, then exit")
    a = ap.parse_args()

    if a.check_doc:
        return check_doc(REPO)

    want = {"a": {"A"}, "b": {"B"}, "all": {"A", "B"}}[a.tier]
    cases = [c for c in CASES if c["tier"] in want]
    if a.only:
        cases = [c for c in cases if c["id"] in set(a.only)]
    if not a.include_slow:
        cases = [c for c in cases if not c["slow"]]

    if a.list:
        for c in CASES:
            flag = " (slow)" if c["slow"] else ""
            print(f"  [{c['tier']}] {c['id']:22s}{flag}\n      {' '.join(c['argv'])}")
            if c["note"]:
                print(f"      note: {c['note']}")
        print("\n  [C] documented, never executed here:")
        for cmd, why in TIER_C:
            print(f"      {cmd}\n          {why}")
        return 0

    if a.dry_run:
        for c in cases:
            print(f"  [{c['tier']}] {' '.join(c['argv'])}")
        return 0

    cwd, room = REPO, None
    if a.clean_room:
        room = Path(subprocess.run(["mktemp", "-d"], capture_output=True, text=True,
                                   check=True).stdout.strip())
        n = build_clean_room(REPO, room)
        print(f"clean room: {room} ({n} files from the intended public fileset)\n", flush=True)
        cwd = room

    results = []
    for c in cases:
        print(f"[{c['tier']}] {c['id']} ...", end=" ", flush=True)
        r = run_case(c, cwd)
        results.append(r)
        print(f"{'ok' if r['ok'] else 'FAIL'} ({r['secs']:.0f}s)"
              + ("" if r["ok"] else f" -- {r['why']}"), flush=True)

    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if a.emit_table:
        print("\n" + emit_table(results, a.clean_room))
    (REPO / "reports").mkdir(exist_ok=True)
    (REPO / "reports" / "usage_test.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "argv"} | {"cmd": " ".join(r["argv"])}
         for r in results], indent=2))

    if room and not a.keep:
        shutil.rmtree(room, ignore_errors=True)
    elif room:
        print(f"clean room kept at {room}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
