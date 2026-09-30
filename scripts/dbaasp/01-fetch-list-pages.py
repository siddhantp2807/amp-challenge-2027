#!/usr/bin/env python3
"""
01_fetch_list_pages.py
=======================
STEP 1 of the DBAASP dump pipeline.

Fetches every page of GET /peptides (limit/offset pagination -- confirmed
working against the live API) and saves each page's raw JSON response,
UNMODIFIED, to its own file in a specified directory.

This step deliberately does NO processing/flattening -- that's step 2
(02_build_peptides_csv.py). Keeping raw pages on disk means you never have
to re-hit the network if step 2's logic needs adjusting later.

USAGE
-----
    python 01_fetch_list_pages.py                                  # full pull, limit=100/page
    python 01_fetch_list_pages.py --limit 200                      # bigger page size, fewer requests
    python 01_fetch_list_pages.py --max-pages 3                     # sanity run: first 3 pages only
    python 01_fetch_list_pages.py --no-resume                       # ignore cached pages, refetch all
    python 01_fetch_list_pages.py --list-pages-dir /data/dbaasp/raw  # custom save location

Resumable: a page already saved on disk is skipped unless --no-resume is given.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dbaasp_common import LIST_URL, DEFAULT_LIST_PAGES_DIR, api_get


def fetch_page(limit, offset):
    return api_get(LIST_URL, params={"limit": limit, "offset": offset})


def page_path(list_pages_dir, offset):
    return list_pages_dir / f"page_offset_{offset:07d}.json"


def main():
    ap = argparse.ArgumentParser(description="Step 1: fetch all /peptides list pages to raw JSON files")
    ap.add_argument("--limit", type=int, default=100, help="page size (items per request)")
    ap.add_argument("--sleep", type=float, default=0.2, help="delay between requests (seconds)")
    ap.add_argument("--max-pages", type=int, default=None, help="stop after N pages (debug/sanity run)")
    ap.add_argument("--no-resume", action="store_true", help="refetch pages even if already cached on disk")
    ap.add_argument("--list-pages-dir", default=str(DEFAULT_LIST_PAGES_DIR),
                     help=f"directory to save raw list-page JSON files (default: {DEFAULT_LIST_PAGES_DIR})")
    args = ap.parse_args()

    list_pages_dir = Path(args.list_pages_dir)
    list_pages_dir.mkdir(parents=True, exist_ok=True)

    offset = 0
    page_count = 0
    total_count = None
    seen_ids = set()

    while True:
        if args.max_pages is not None and page_count >= args.max_pages:
            print(f"Reached --max-pages={args.max_pages}, stopping.")
            break

        path = page_path(list_pages_dir, offset)
        if not args.no_resume and path.exists():
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                data = fetch_page(args.limit, offset)
                path.write_text(json.dumps(data))
                time.sleep(args.sleep)
        else:
            data = fetch_page(args.limit, offset)
            path.write_text(json.dumps(data))
            time.sleep(args.sleep)

        total_count = data.get("totalCount", total_count)
        items = data.get("data", [])
        new_ids = 0
        for item in items:
            pid = item.get("id")
            if pid is not None and pid not in seen_ids:
                seen_ids.add(pid)
                new_ids += 1

        page_count += 1
        print(f"  page offset={offset}: {len(items)} items ({new_ids} new) -> {path.name}   "
              f"[total so far: {len(seen_ids)}" + (f"/{total_count}" if total_count else "") + "]")

        if not items:
            print("Empty page -- stopping.")
            break
        if new_ids == 0:
            print("  [warn] this page contributed zero new ids -- pagination may have wrapped, "
                  "or we've reached the end / offset params stopped working. Stopping here.",
                  file=sys.stderr)
            break
        if total_count is not None and len(seen_ids) >= total_count:
            print("Reached totalCount -- all pages fetched.")
            break

        offset += args.limit

    print(f"\nDone. {page_count} pages fetched, {len(seen_ids)} unique peptide ids seen"
          + (f" out of totalCount={total_count}" if total_count else "") + ".")
    print(f"Raw pages saved under: {list_pages_dir}/")


if __name__ == "__main__":
    main()