"""Run the full pipeline: extract -> embed -> cluster -> label -> report.

Usage:
    python run.py                     # auto: MongoDB if configured in .env, else sample data
    python run.py --source sample     # force the bundled sample dataset
    python run.py --source mongo      # force MongoDB (requires MONGO_URI + MONGO_DB in .env)
    python run.py --limit 500         # only read the first 500 tickets (quick test)
    python run.py --incremental       # phase 5: only pull/embed tickets newer than last run
                                       # (requires MongoDB; point Windows Task Scheduler / cron
                                       # at this to keep the report current without a full re-run)
"""
from __future__ import annotations

import argparse

from src.generate_report import generate_report
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["auto", "mongo", "sample"], default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N tickets")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only fetch/embed tickets newer than the last run (MongoDB only), then re-cluster and refresh the report.",
    )
    args = parser.parse_args()

    result = run_pipeline(source=args.source, limit=args.limit, incremental=args.incremental)
    if result is None:
        print("\nNo new tickets since the last run - report left as-is.")
        return

    path = generate_report()
    print(f"\nReport written to: {path}")
    print("Open it in your browser to view the dashboard.")


if __name__ == "__main__":
    main()
