"""
Experiment ledger — one CSV row per run, the queryable index of every experiment.

This is Layer 1 of the experiment record (the *numbers*).  Layer 2 — the
plain-language conclusions — lives in docs/experiments/notebook.md.

Each row = one run: every ExperimentConfig field + every metric + run_dir +
timestamp, flattened into a single CSV.  With it, any quantitative question
("pearson vs lag", "which noise level broke recovery") is a one-line groupby.

Two entry points
----------------
    append_row(result, config, ledger_path)   — called automatically by
                                                 run_single after each run
    rebuild(results_root, ledger_path)         — backfill: scan every run dir
                                                 under results_root and rewrite
                                                 the ledger from scratch

Schema evolution
----------------
New config fields or newly registered metrics simply become new columns; older
rows get NaN for them.  Nothing breaks when the pipeline grows — that is why
the ledger is rebuilt with pandas (column-aligned concat) rather than appended
with the stdlib csv writer.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from .config import ExperimentConfig
from .result import ExperimentResult
from .persistence import load_result


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #

def _row(result: ExperimentResult, config: ExperimentConfig) -> dict:
    """One flat dict: config fields + metrics + duration_s + timestamp + run_dir."""
    row = dataclasses.asdict(config)
    row.update(result.summary())   # metrics + duration_s + timestamp + run_dir
    return row


# --------------------------------------------------------------------------- #
# Append one run (called from run_single)
# --------------------------------------------------------------------------- #

def append_row(
    result: ExperimentResult,
    config: ExperimentConfig,
    ledger_path: str | Path = "results/ledger.csv",
) -> Path:
    """
    Append a single run to the ledger CSV, creating it (and parents) if needed.

    Column-aligned: if this run introduced a new config field or metric, the
    column is added and prior rows get NaN.  Idempotent-ish — a re-logged
    run_dir produces a duplicate row; use rebuild() to deduplicate.
    """
    import pandas as pd

    ledger_path = Path(ledger_path)
    new = pd.DataFrame([_row(result, config)])

    if ledger_path.exists():
        existing = pd.read_csv(ledger_path)
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        combined = new

    combined.to_csv(ledger_path, index=False)
    return ledger_path


# --------------------------------------------------------------------------- #
# Rebuild from scratch (backfill / deduplicate)
# --------------------------------------------------------------------------- #

def rebuild(
    results_root: str | Path = "results",
    ledger_path: str | Path = "results/ledger.csv",
):
    """
    Scan every run directory under results_root and rewrite the ledger.

    A run directory is any folder containing both config.json and metrics.json.
    Runs that fail to load (missing/corrupt files, e.g. legacy archive runs)
    are skipped with a warning rather than aborting the whole rebuild.

    Returns the resulting pandas DataFrame.
    """
    import pandas as pd

    results_root = Path(results_root)
    rows = []
    for cfg_path in sorted(results_root.rglob("config.json")):
        run_dir = cfg_path.parent
        if not (run_dir / "metrics.json").exists():
            continue
        try:
            result, config = load_result(run_dir)
        except Exception as e:           # noqa: BLE001 — never abort a backfill
            print(f"[ledger] skipping {run_dir}: {e}")
            continue
        rows.append(_row(result, config))

    df = pd.DataFrame(rows)
    if "run_dir" in df.columns:
        df = df.drop_duplicates(subset="run_dir", keep="last")

    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ledger_path, index=False)
    print(f"[ledger] wrote {len(df)} runs → {ledger_path}")
    return df


# --------------------------------------------------------------------------- #
# Quick query helper
# --------------------------------------------------------------------------- #

def effect_of(
    param: str,
    metric: str = "connectivity/pearson",
    ledger_path: str | Path = "results/ledger.csv",
):
    """
    Quick look at how one parameter moves one metric, averaged over everything
    else in the ledger.

        effect_of("lag_ms", "connectivity/auc_roc")

    Returns a pandas DataFrame grouped by `param` with mean/std/count of `metric`.
    """
    import pandas as pd

    df = pd.read_csv(ledger_path)
    if param not in df.columns:
        raise KeyError(f"'{param}' not in ledger. Columns: {list(df.columns)}")
    if metric not in df.columns:
        raise KeyError(f"'{metric}' not in ledger. Columns: {list(df.columns)}")
    return (
        df.groupby(param)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(param)
    )
