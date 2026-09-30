#!/usr/bin/env python3
"""
03_fetch_peptide_details.py
============================
STEP 3 of the DBAASP dump pipeline.

Reads dbaasp_peptides_list.csv (produced by 02_build_peptides_csv.py),
takes every `dbaaspId`, and fetches the FULL detail record for each from
GET /peptides/{dbaaspId} -- this is where physicoChemicalProperties,
targetActivities, antibiofilmActivities, hemoliticCytotoxicActivities
[sic], synergies, sourceGenes, articles, smiles, bonds, etc. actually live
(the list endpoint from step 1 only has general/structural fields).

Each peptide's full JSON is saved, UNMODIFIED, one file per peptide, to
dbaasp_raw/details/<dbaaspId>.json. Resumable: already-cached files are
skipped unless --no-resume is given -- safe to Ctrl-C and rerun.

This script does NOT flatten/CSV-ify the detail data. That's a natural
step 4 (not yet written -- TBD once the pipeline's execution order/
orchestration is decided, per your note).

USAGE
-----
    python 03_fetch_peptide_details.py                  # full pull
    python 03_fetch_peptide_details.py --limit 20         # sanity run, first 20 only
    python 03_fetch_peptide_details.py --no-resume         # refetch everything
    python 03_fetch_peptide_details.py --details-dir /data/dbaasp/details \\
                                        --checkpoint-file /data/dbaasp/details_checkpoint.json \\
                                        --peptides-csv /data/dbaasp/peptides.csv
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from dbaasp_common import (
    DETAIL_URL_TMPL, DEFAULT_DETAILS_DIR, DEFAULT_DETAIL_CHECKPOINT_FILE, DEFAULT_PEPTIDES_CSV, api_get,
)


def get_peptide_detail(dbaasp_id):
    return api_get(DETAIL_URL_TMPL.format(dbaasp_id=dbaasp_id))


def load_checkpoint(checkpoint_file):
    if checkpoint_file.exists():
        return set(json.loads(checkpoint_file.read_text()))
    return set()


def save_checkpoint(checkpoint_file, done):
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file.write_text(json.dumps(sorted(done)))


def main():
    ap = argparse.ArgumentParser(description="Step 3: fetch full detail JSON for every peptide")
    ap.add_argument("--peptides-csv", default=str(DEFAULT_PEPTIDES_CSV),
                     help=f"CSV from step 2 to read dbaaspIds from (default: {DEFAULT_PEPTIDES_CSV})")
    ap.add_argument("--details-dir", default=str(DEFAULT_DETAILS_DIR),
                     help=f"directory to save per-peptide detail JSON files (default: {DEFAULT_DETAILS_DIR})")
    ap.add_argument("--checkpoint-file", default=str(DEFAULT_DETAIL_CHECKPOINT_FILE),
                     help=f"resume checkpoint file path (default: {DEFAULT_DETAIL_CHECKPOINT_FILE})")
    ap.add_argument("--sleep", type=float, default=0.2, help="delay between requests (seconds)")
    ap.add_argument("--limit", type=int, default=None, help="only fetch first N peptides (debug/sanity run)")
    ap.add_argument("--no-resume", action="store_true", help="ignore cache/checkpoint, refetch everything")
    args = ap.parse_args()

    details_dir = Path(args.details_dir)
    checkpoint_file = Path(args.checkpoint_file)

    df = pd.read_csv(args.peptides_csv)
    if "dbaaspId" not in df.columns:
        raise SystemExit(f"'dbaaspId' column not found in {args.peptides_csv} -- did step 2 run correctly?")

    dbaasp_ids = df["dbaaspId"].dropna().astype(str).unique().tolist()
    n_missing = df["dbaaspId"].isna().sum()
    if n_missing:
        print(f"  [warn] {n_missing} peptides in {args.peptides_csv} have no dbaaspId and will be skipped.",
              file=sys.stderr)

    if args.limit:
        dbaasp_ids = dbaasp_ids[: args.limit]

    details_dir.mkdir(parents=True, exist_ok=True)
    done = load_checkpoint(checkpoint_file) if not args.no_resume else set()
    total = len(dbaasp_ids)

    for i, did in enumerate(dbaasp_ids, 1):
        out_path = details_dir / f"{did}.json"
        if not args.no_resume and did in done and out_path.exists():
            continue
        try:
            detail = get_peptide_detail(did)
            out_path.write_text(json.dumps(detail))
            done.add(did)
        except Exception as e:
            print(f"  [error] {did}: {e}", file=sys.stderr)

        if i % 50 == 0 or i == total:
            print(f"  progress: {i}/{total} ({100*i/total:.1f}%)")
            save_checkpoint(checkpoint_file, done)
        time.sleep(args.sleep)

    save_checkpoint(checkpoint_file, done)
    print(f"\nDone. {len(done)}/{total} peptide detail records cached under: {details_dir}/")


if __name__ == "__main__":
    main()