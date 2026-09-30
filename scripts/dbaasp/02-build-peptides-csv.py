#!/usr/bin/env python3
"""
02_build_peptides_csv.py
=========================
STEP 2 of the DBAASP dump pipeline.

Reads every raw list-page JSON file produced by 01_fetch_list_pages.py
(under dbaasp_raw/list_pages/), combines and de-duplicates all peptide
records (by numeric `id`, in case pages ever overlapped), flattens each
into one CSV row, and writes the result to dbaasp_peptides_list.csv.

This CSV is the GENERAL/SUMMARY peptide list from GET /peptides -- id,
dbaaspId, name, sequence, termini, complexity, synthesis type, plus a
joined-string summary of `monomers` for multimer/multi-peptide entries.
It does NOT contain activity or physicochemical-property data -- that
only lives in the per-peptide detail endpoint (see step 3).

Critically, the `dbaaspId` column this script writes is exactly what step
3 needs: GET /peptides/{dbaaspId} takes the STRING id, not the numeric one.

USAGE
-----
    python 02_build_peptides_csv.py
    python 02_build_peptides_csv.py --out custom_name.csv
    python 02_build_peptides_csv.py --list-pages-dir /data/dbaasp/raw --out /data/dbaasp/peptides.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from dbaasp_common import DEFAULT_LIST_PAGES_DIR, DEFAULT_PEPTIDES_CSV, flatten_value


def load_all_page_items(list_pages_dir):
    paths = sorted(list_pages_dir.glob("page_offset_*.json"))
    if not paths:
        raise SystemExit(f"No pages found under {list_pages_dir}/ -- run 01_fetch_list_pages.py first "
                          f"(or check --list-pages-dir points at the right place).")
    print(f"Found {len(paths)} cached list pages.")
    for p in paths:
        data = json.loads(p.read_text())
        for item in data.get("data", []):
            yield item


def flatten_peptide_summary(item):
    """One list-endpoint peptide record -> one flat dict (CSV row)."""
    item = dict(item)  # shallow copy, we're about to pop from it
    monomers = item.pop("monomers", None)

    row = {}
    for k, v in item.items():
        row[k] = flatten_value(v) if isinstance(v, (dict, list)) else v

    if monomers:
        row["monomer_count"] = len(monomers)
        row["monomers_summary"] = "; ".join(
            f"{m.get('name', '')} [{m.get('sequence', '')}]" for m in monomers
        )
    else:
        row["monomer_count"] = 0
        row["monomers_summary"] = None

    return row


def main():
    ap = argparse.ArgumentParser(description="Step 2: combine raw list pages into one CSV")
    ap.add_argument("--list-pages-dir", default=str(DEFAULT_LIST_PAGES_DIR),
                     help=f"directory to read raw list-page JSON files from (default: {DEFAULT_LIST_PAGES_DIR})")
    ap.add_argument("--out", default=str(DEFAULT_PEPTIDES_CSV),
                     help=f"output CSV path (default: {DEFAULT_PEPTIDES_CSV})")
    args = ap.parse_args()

    list_pages_dir = Path(args.list_pages_dir)

    seen_ids = set()
    rows = []
    for item in load_all_page_items(list_pages_dir):
        pid = item.get("id")
        if pid in seen_ids:
            continue  # de-dupe in case pages overlapped
        seen_ids.add(pid)
        rows.append(flatten_peptide_summary(item))

    if not rows:
        raise SystemExit("No peptide records found in cached pages -- something's wrong upstream.")

    df = pd.DataFrame(rows)
    if "id" in df.columns:
        df = df.sort_values("id").reset_index(drop=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df)} unique peptides x {df.shape[1]} columns -> {args.out}")
    if "dbaaspId" in df.columns:
        n_missing = df["dbaaspId"].isna().sum()
        print(f"  ({n_missing} rows missing dbaaspId -- these can't be fetched in step 3)")
    else:
        print("  [warn] no 'dbaaspId' column found -- check the raw page schema, step 3 needs this.")


if __name__ == "__main__":
    main()