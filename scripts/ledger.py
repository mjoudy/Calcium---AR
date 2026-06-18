"""
Experiment-ledger CLI — backfill and query the run index.

The ledger (results/ledger.csv) is normally kept up to date automatically:
run_single appends one row per run.  This CLI is for the two things you do
by hand.

Usage
-----
    # Rebuild from scratch by scanning every run dir (captures past runs too)
    python scripts/ledger.py rebuild
    python scripts/ledger.py rebuild --results results --out results/ledger.csv

    # Quick "what does this parameter do?" — averaged over everything else
    python scripts/ledger.py effect lag_ms
    python scripts/ledger.py effect smooth_window_ms --metric connectivity/auc_roc

    # Dump the whole ledger to the terminal
    python scripts/ledger.py show
    python scripts/ledger.py show --cols lag_ms,smooth_window_ms,connectivity/pearson
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calcium_ar.experiments.ledger import rebuild, effect_of


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("rebuild", help="scan run dirs and rewrite the ledger")
    pr.add_argument("--results", default="results")
    pr.add_argument("--out", default="results/ledger.csv")

    pe = sub.add_parser("effect", help="mean metric grouped by a parameter")
    pe.add_argument("param")
    pe.add_argument("--metric", default="connectivity/pearson")
    pe.add_argument("--ledger", default="results/ledger.csv")

    ps = sub.add_parser("show", help="print the ledger")
    ps.add_argument("--ledger", default="results/ledger.csv")
    ps.add_argument("--cols", default=None, help="comma-separated subset of columns")

    args = p.parse_args()

    if args.cmd == "rebuild":
        rebuild(args.results, args.out)

    elif args.cmd == "effect":
        table = effect_of(args.param, args.metric, args.ledger)
        print(f"\nEffect of '{args.param}' on '{args.metric}':\n")
        print(table.to_string(index=False))

    elif args.cmd == "show":
        import pandas as pd
        df = pd.read_csv(args.ledger)
        if args.cols:
            cols = [c.strip() for c in args.cols.split(",")]
            df = df[[c for c in cols if c in df.columns]]
        with pd.option_context("display.max_rows", None, "display.width", None):
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
