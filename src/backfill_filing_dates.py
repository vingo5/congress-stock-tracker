"""
backfill_filing_dates.py — Phase 1b.

The aggregate all_transactions.json (used in ingest.py) has no filing date,
only the trade's transaction_date. But the same source repo also publishes
one JSON file PER FILING, named transaction_report_for_MM_DD_YYYY.json,
where the date IS the filing date (field: date_recieved) and each filing
contains a nested list of the individual trades it discloses.

This script joins those per-filing records back to our existing
`transactions` rows (matched via the same source_hash logic used in
ingest.py) and fills in filing_date.

Known limitation, found by testing this join against the full dataset:
only ~77-78% of rows get a filing_date. The rest fall into two buckets,
confirmed by inspection rather than assumed:
  1. Filing report files exist for dates outside the aggregate file's
     actual coverage window (e.g. late-2020 filings with no matching
     transaction_date in all_transactions.json) - a genuine inconsistency
     between two files in the same source repo, not a bug in this script.
  2. A minority of asset_description fields in the per-filing files contain
     embedded HTML (e.g. nested <div> elements describing private-placement
     assets) that isn't present in the cleaned aggregate file, so the two
     representations of the same trade don't string-match.
This ceiling is documented rather than chased with fuzzy string matching,
which would trade a known, explainable gap for an unverifiable one.
"""

import io
import json
import re
import sqlite3
import tarfile
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_TARBALL_URL = (
    "https://codeload.github.com/timothycarambat/"
    "senate-stock-watcher-data/tar.gz/refs/heads/master"
)
AGGREGATE_URL = (
    "https://raw.githubusercontent.com/timothycarambat/"
    "senate-stock-watcher-data/master/aggregate/all_transactions.json"
)
DB_PATH = Path(__file__).parent.parent / "data" / "congress_trades.db"

JUNK_TICKERS = {"--", "N/A", ""}
TICKER_HTML_RE = re.compile(r">([^<]*)<")


def strip_ticker_html(raw: str | None) -> str | None:
    """Per-filing files wrap tickers in an <a> tag; the aggregate file
    doesn't. Both need to resolve to the same plain string to join."""
    if raw is None:
        return None
    match = TICKER_HTML_RE.search(raw)
    text = (match.group(1) if match else raw).strip()
    return None if text in JUNK_TICKERS else text


def make_source_hash(senator: str, tx_date: str, ticker: str, amount: str,
                      ttype: str, asset_description: str) -> str:
    """Must exactly match ingest.py's make_source_hash so results match
    the source_hash values already stored in the transactions table."""
    import hashlib
    key = "|".join([senator, tx_date, ticker or "", amount, ttype, asset_description])
    return hashlib.sha256(key.encode()).hexdigest()


def parse_filing_date(raw_date: str) -> str | None:
    if not raw_date:
        return None
    try:
        return datetime.strptime(raw_date, "%m/%d/%Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def build_aggregate_index() -> dict:
    """Indexes aggregate records by every field EXCEPT senator name, so we
    can join on the rest and match senator case-insensitively (the two
    files disagree on name capitalization for some senators, e.g.
    'Iii' vs 'III')."""
    req = urllib.request.Request(AGGREGATE_URL, headers={"User-Agent": "congress-stock-tracker"})
    with urllib.request.urlopen(req) as resp:
        agg = json.loads(resp.read())

    index = {}
    for r in agg:
        key = (
            r.get("transaction_date", ""), r.get("amount", ""),
            r.get("type", ""), r.get("asset_description", ""),
            r.get("ticker", ""),
        )
        index.setdefault(key, []).append(r)
    return index


def download_and_extract_filings() -> list[Path]:
    """Downloads the whole repo tarball once (1 request) rather than
    hitting 1000+ individual raw.githubusercontent.com URLs."""
    req = urllib.request.Request(REPO_TARBALL_URL, headers={"User-Agent": "congress-stock-tracker"})
    with urllib.request.urlopen(req) as resp:
        tar_bytes = resp.read()

    extract_dir = Path(__file__).parent.parent / "data" / "_filing_reports_tmp"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        members = [
            m for m in tar.getmembers()
            if "/data/" in m.name and m.name.endswith(".json") and not m.name.endswith(".yaml")
        ]
        tar.extractall(path=extract_dir, members=members)

    return sorted(extract_dir.rglob("transaction_report_for_*.json"))


def run_backfill():
    print("Building aggregate index for matching...")
    agg_index = build_aggregate_index()

    print("Downloading and extracting per-filing report files...")
    filing_files = download_and_extract_filings()
    print(f"Found {len(filing_files)} filing report files.")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    matched, unmatched = 0, 0
    # Two-pass: gather all candidate filing_dates per source_hash first,
    # rather than taking whichever filing file we process first. Filenames
    # like transaction_report_for_01_02_2013.json sort alphabetically by
    # month-then-year, NOT chronologically across years, so a naive
    # "first one wins" approach can attach a wrong, earlier-sorted-but-
    # later-actual-date filing to a transaction (confirmed by testing:
    # this produced a handful of impossible filing_date < transaction_date
    # rows). Collecting all candidates and picking the earliest valid one
    # fixes this deterministically regardless of file processing order.
    candidates_by_hash: dict[str, list[str]] = {}

    for fp in filing_files:
        try:
            filings = json.loads(fp.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        for filing in filings:
            full_name = f"{filing.get('first_name', '').strip()} {filing.get('last_name', '').strip()}"
            filing_date = parse_filing_date(filing.get("date_recieved"))
            if not filing_date:
                continue

            for tx in filing.get("transactions", []):
                ticker_clean = strip_ticker_html(tx.get("ticker")) or ""
                key = (
                    tx.get("transaction_date", ""), tx.get("amount", ""),
                    tx.get("type", ""), tx.get("asset_description", ""),
                    ticker_clean,
                )
                candidates = agg_index.get(key, [])
                match = next(
                    (c for c in candidates if c.get("senator", "").lower() == full_name.lower()),
                    None,
                )
                if match is None:
                    unmatched += 1
                    continue

                matched += 1
                source_hash = make_source_hash(
                    match.get("senator", ""), match.get("transaction_date", ""),
                    match.get("ticker", ""), match.get("amount", ""),
                    match.get("type", ""), match.get("asset_description", ""),
                )
                candidates_by_hash.setdefault(source_hash, []).append(filing_date)

    updated, anomalies = 0, 0
    for source_hash, dates in candidates_by_hash.items():
        cur.execute(
            "SELECT transaction_date FROM transactions WHERE source_hash = ?", (source_hash,)
        )
        row = cur.fetchone()
        if row is None:
            continue
        tx_date = row[0]

        valid_dates = sorted(d for d in dates if d >= tx_date)
        if valid_dates:
            chosen_date = valid_dates[0]
        else:
            # Every candidate filing_date is before the trade date - a
            # genuine data anomaly (likely a duplicate/amended disclosure
            # mismatch upstream). Recorded, not silently dropped: we skip
            # setting filing_date here rather than writing an impossible
            # value, and count it so it's visible in the run summary.
            anomalies += 1
            continue

        cur.execute(
            "UPDATE transactions SET filing_date = ? WHERE source_hash = ?",
            (chosen_date, source_hash),
        )
        if cur.rowcount:
            updated += 1

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM transactions")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM transactions WHERE filing_date IS NOT NULL")
    with_filing_date = cur.fetchone()[0]

    print(f"\nFiling-record matches attempted: {matched + unmatched} "
          f"(matched: {matched}, unmatched: {unmatched})")
    print(f"Rows updated in DB: {updated}")
    print(f"Anomalies skipped (all candidate filing_dates precede transaction_date): {anomalies}")
    print(f"Coverage: {with_filing_date}/{total} transactions now have filing_date "
          f"({with_filing_date/total:.1%})")

    conn.close()


if __name__ == "__main__":
    run_backfill()
