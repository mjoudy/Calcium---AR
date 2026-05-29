"""
Solver comparison script.

Simulates one ground-truth dataset (or loads an existing one), then runs
every registered solver on identical data and produces a side-by-side report.

Usage
-----
    python scripts/solver_comparison.py                         # N=100 defaults
    python scripts/solver_comparison.py --ne 1000 --ni 250     # N=1250
    python scripts/solver_comparison.py --data-path my_ds/ --solvers ols ridge

Outputs are written to  <out_dir>/solver_comparison_N<N>/
    summary_table.txt     — human-readable metric table
    summary_table.csv     — machine-readable
    metrics_bar.png       — pearson / AUC-ROC / MSE bar chart
    adj_heatmaps.png      — adj_true + adj_inferred for each solver
    adj_scatter.png       — inferred vs true weight scatter per solver
    loss_curves.png       — convergence curves for gradient-based solvers
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── make sure package is importable from scripts/ ──────────────────────────
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from calcium_ar.experiments.config import ExperimentConfig
from calcium_ar.experiments.runner import run_single


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare all solvers on one dataset")
    p.add_argument("--ne",         type=int,   default=80,       help="# excitatory neurons")
    p.add_argument("--ni",         type=int,   default=20,       help="# inhibitory neurons")
    p.add_argument("--sim-time",   type=float, default=5_000.0,  help="simulation length (ms)")
    p.add_argument("--dt",         type=float, default=0.1,      help="time step (ms)")
    p.add_argument("--g",          type=float, default=5.0)
    p.add_argument("--eta",        type=float, default=2.0)
    p.add_argument("--tau",        type=float, default=100.0,    help="calcium decay (ms)")
    p.add_argument("--lag",        type=int,   default=10,       help="AR lag (time steps)")
    p.add_argument("--lam",        type=float, default=1.0,      help="regularisation λ")
    p.add_argument("--n-epochs",   type=int,   default=10,       help="epochs for torch solvers")
    p.add_argument("--chunk-size", type=int,   default=10_000)
    p.add_argument("--threads",    type=int,   default=1,        help="NEST CPU threads")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument(
        "--solvers",
        nargs="+",
        default=["ols", "ridge", "torch_normal_eq", "torch_minibatch", "torch_linear_layer"],
        help="subset of solvers to run",
    )
    p.add_argument(
        "--data-path",
        default=None,
        help="path to existing/new SimulatedDataset (shared across all solvers). "
             "Defaults to <out_dir>/dataset/",
    )
    p.add_argument(
        "--out-dir",
        default="results",
        help="parent output directory",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_REF_JEX_CE = 80.0   # Brunel 2000 conservation law: J_ex × C_E = 80

def _auto_j_ex(ne: int, epsilon: float) -> float:
    c_e = max(1, int(ne * epsilon))
    return _REF_JEX_CE / c_e


def _print_table(rows: list[dict], solvers: list[str]) -> str:
    metrics = [
        ("connectivity/pearson",         "Pearson r",    ".4f"),
        ("connectivity/auc_roc",         "AUC-ROC",      ".4f"),
        ("connectivity/mse",             "MSE",          ".6f"),
        ("connectivity/sparsity_ratio",  "Sparsity ratio",".3f"),
        ("tau/tau_relative_error",       "τ rel. error", ".4f"),
        ("solver/final_loss",            "Final loss",   ".4e"),
        ("solver/convergence_ratio",     "Conv. ratio",  ".4f"),
        ("_duration",                    "Time (s)",     ".2f"),
    ]

    col_w = 18
    header = f"{'Metric':<28}" + "".join(f"{s:>{col_w}}" for s in solvers)
    sep    = "─" * (28 + col_w * len(solvers))
    lines  = [sep, header, sep]

    for key, label, fmt in metrics:
        row_str = f"{label:<28}"
        for r in rows:
            val = r.get(key, float("nan"))
            try:
                row_str += f"{format(val, fmt):>{col_w}}"
            except (ValueError, TypeError):
                row_str += f"{'—':>{col_w}}"
        lines.append(row_str)

    lines.append(sep)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _plot_metrics_bar(rows: list[dict], solvers: list[str], out: Path) -> None:
    metric_keys = [
        ("connectivity/pearson",  "Pearson r",   None),
        ("connectivity/auc_roc",  "AUC-ROC",     None),
        ("connectivity/mse",      "MSE",         None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(solvers)))

    for ax, (key, label, _) in zip(axes, metric_keys):
        vals = [r.get(key, float("nan")) for r in rows]
        bars = ax.bar(solvers, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(label, fontsize=11)
        ax.set_xticks(range(len(solvers)))
        ax.set_xticklabels(solvers, rotation=30, ha="right", fontsize=8)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Solver comparison — connectivity metrics", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out.name}")


def _plot_adj_heatmaps(adj_true: np.ndarray, adj_dict: dict[str, np.ndarray],
                       out: Path) -> None:
    # Convention alignment
    # --------------------
    # adj_true[source, target]: row = source neuron (E rows first, then I rows)
    # adj_inferred[target, source]: A[i,j] = effect of j on i  →  A.T[source, target]
    # We plot A.T so both matrices share the same row=source convention.
    n_panels = 1 + len(adj_dict)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    # Ground truth — natural scale, own colorbar  (option 4)
    vmax_gt = np.abs(adj_true).max() or 1.0
    im0 = axes[0].imshow(adj_true, aspect="auto", cmap="RdBu_r",
                         vmin=-vmax_gt, vmax=vmax_gt)
    axes[0].set_title("Ground truth\n[row = source, mV]", fontsize=9)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Estimated — 98th-percentile clipping (option 1), own colorbar (option 4)
    for ax, (name, A) in zip(axes[1:], adj_dict.items()):
        A_plot = A.T                                           # row = source
        vmax_i = float(np.percentile(np.abs(A_plot), 98)) or 1.0
        im_i = ax.imshow(A_plot, aspect="auto", cmap="RdBu_r",
                         vmin=-vmax_i, vmax=vmax_i)
        ax.set_title(f"{name}  (A.T, 98-pct scale)\n[row = source, a.u.]", fontsize=9)
        ax.axis("off")
        plt.colorbar(im_i, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Adjacency matrices", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out.name}")


def _plot_adj_histograms(adj_true: np.ndarray, adj_dict: dict[str, np.ndarray],
                         out: Path) -> None:
    """
    Two histogram panels per solver:
      Left  — distribution of all off-diagonal estimated entries.
      Right — same entries split by ground-truth category
               (excitatory / inhibitory / unconnected).
    """
    N    = adj_true.shape[0]
    mask = ~np.eye(N, dtype=bool)
    g    = adj_true.T[mask]     # ground-truth in A's convention

    exc_m = g > 0
    inh_m = g < 0
    unc_m = g == 0

    n_solvers = len(adj_dict)
    fig, axes = plt.subplots(n_solvers, 2,
                             figsize=(10, 3.8 * n_solvers),
                             squeeze=False)

    for row, (name, A) in enumerate(adj_dict.items()):
        a = A[mask]

        # Left: all off-diagonal entries
        ax_l = axes[row, 0]
        ax_l.hist(a, bins=80, color="steelblue", edgecolor="none", alpha=0.85)
        ax_l.axvline(0, color="k", lw=0.9, ls="--")
        ax_l.set_xlabel("Estimated weight  (a.u.)", fontsize=9)
        ax_l.set_ylabel("Count", fontsize=9)
        ax_l.set_title(f"{name} — all off-diagonal entries  (n={len(a):,})", fontsize=9)

        # Right: split by ground-truth category
        ax_r = axes[row, 1]
        bins = np.linspace(a.min(), a.max(), 80)
        ax_r.hist(a[unc_m], bins=bins, alpha=0.55, color="gray",
                  label=f"Unconnected  (n={unc_m.sum():,})", edgecolor="none")
        ax_r.hist(a[exc_m], bins=bins, alpha=0.70, color="tomato",
                  label=f"Excitatory   (n={exc_m.sum():,})", edgecolor="none")
        ax_r.hist(a[inh_m], bins=bins, alpha=0.70, color="steelblue",
                  label=f"Inhibitory   (n={inh_m.sum():,})", edgecolor="none")
        ax_r.axvline(0, color="k", lw=0.9, ls="--")
        ax_r.set_xlabel("Estimated weight  (a.u.)", fontsize=9)
        ax_r.set_ylabel("Count", fontsize=9)
        ax_r.set_title(f"{name} — by ground-truth type", fontsize=9)
        ax_r.legend(fontsize=8)

    fig.suptitle("Distribution of inferred connectivity weights", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out.name}")


def _plot_adj_scatter(adj_true: np.ndarray, adj_dict: dict[str, np.ndarray],
                      out: Path) -> None:
    n = len(adj_dict)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).ravel()

    N = adj_true.shape[0]
    mask = ~np.eye(N, dtype=bool)
    true_flat = adj_true.T[mask]   # adj_true.T: same convention as A (row=target, col=source)

    for ax, (name, A) in zip(axes, adj_dict.items()):
        inf_flat = A[mask]
        ax.scatter(true_flat, inf_flat, s=1, alpha=0.3, rasterized=True)
        lim = max(np.abs(true_flat).max(), np.abs(inf_flat).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=0.8, alpha=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("True weight  (adj_true.T)", fontsize=8)
        ax.set_ylabel("Inferred weight", fontsize=8)
        ax.set_title(name, fontsize=9)
        corr = np.corrcoef(true_flat, inf_flat)[0, 1]
        ax.text(0.05, 0.92, f"r={corr:.3f}", transform=ax.transAxes, fontsize=8)

    for ax in axes[len(adj_dict):]:
        ax.axis("off")

    fig.suptitle("Inferred vs true weights (off-diagonal)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out.name}")


def _plot_loss_curves(loss_dict: dict[str, list[float]], out: Path) -> None:
    curves = {k: v for k, v in loss_dict.items() if v}
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, curve in curves.items():
        ax.plot(range(1, len(curve) + 1), curve, marker="o", markersize=4, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Gradient solver convergence")
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out.name}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = _parse()

    N  = args.ne + args.ni
    out_dir = Path(args.out_dir) / f"solver_comparison_N{N}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data_path or str(out_dir / "dataset")

    epsilon = 0.1
    j_ex    = _auto_j_ex(args.ne, epsilon)

    print("=" * 60)
    print(f"  Solver comparison  N={N} (NE={args.ne}, NI={args.ni})")
    print(f"  J_ex={j_ex:.4f} mV  (auto-scaled: {_REF_JEX_CE} / C_E({int(args.ne*epsilon)}))")
    print(f"  g={args.g}  eta={args.eta}  sim_time={args.sim_time} ms")
    print(f"  solvers: {args.solvers}")
    print(f"  dataset: {data_path}")
    print(f"  output : {out_dir}")
    print("=" * 60)

    base_cfg = ExperimentConfig(
        n_excitatory = args.ne,
        n_inhibitory = args.ni,
        epsilon      = epsilon,
        g            = args.g,
        eta          = args.eta,
        J_ex         = j_ex,
        sim_time     = args.sim_time,
        dt           = args.dt,
        n_threads    = args.threads,
        tau          = args.tau,
        lag          = args.lag,
        lam          = args.lam,
        chunk_size   = args.chunk_size,
        n_epochs     = args.n_epochs,
        seed         = args.seed,
        data_path    = data_path,
        output_dir   = str(out_dir),
    )

    # ------------------------------------------------------------------ #
    # Run each solver
    # ------------------------------------------------------------------ #
    rows:      list[dict]             = []
    adj_dict:  dict[str, np.ndarray] = {}
    loss_dict: dict[str, list[float]] = {}
    adj_true:  np.ndarray | None      = None

    for solver in args.solvers:
        print(f"\n{'─'*40}")
        print(f"  Solver: {solver}")
        print(f"{'─'*40}")

        cfg = base_cfg.replace(solver=solver, name=solver)
        t0  = time.perf_counter()
        try:
            result = run_single(cfg)
        except Exception as exc:
            print(f"  [FAILED] {exc}")
            rows.append({"_solver": solver, "_duration": float("nan"), "_failed": True})
            continue

        duration = time.perf_counter() - t0
        row = dict(result.metrics)
        row["_solver"]   = solver
        row["_duration"] = duration
        row["_failed"]   = False
        rows.append(row)

        # collect arrays for plots
        A_inf = np.load(Path(result.adj_inferred_path))
        adj_dict[solver] = A_inf
        loss_dict[solver] = result.loss_curve

        if adj_true is None:
            adj_true = np.load(Path(result.adj_true_path))

        print(f"  pearson={row.get('connectivity/pearson', float('nan')):.4f}  "
              f"auc={row.get('connectivity/auc_roc', float('nan')):.4f}  "
              f"time={duration:.1f}s")

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    table_str = _print_table(rows, args.solvers)
    print(table_str)

    (out_dir / "summary_table.txt").write_text(table_str + "\n")

    # CSV
    import csv
    all_keys = sorted({k for r in rows for k in r if not k.startswith("_")})
    with open(out_dir / "summary_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["solver"] + all_keys + ["duration_s"])
        writer.writeheader()
        for r in rows:
            csv_row = {"solver": r["_solver"], "duration_s": r["_duration"]}
            csv_row.update({k: r.get(k, "") for k in all_keys})
            writer.writerow(csv_row)

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    print("\n  Generating plots …")
    successful = [r for r in rows if not r["_failed"]]
    if successful:
        _plot_metrics_bar(successful, [r["_solver"] for r in successful],
                          out_dir / "metrics_bar.png")

    if adj_true is not None and adj_dict:
        _plot_adj_heatmaps(adj_true, adj_dict,    out_dir / "adj_heatmaps.png")
        _plot_adj_scatter(adj_true, adj_dict,     out_dir / "adj_scatter.png")
        _plot_adj_histograms(adj_true, adj_dict,  out_dir / "adj_histograms.png")

    if loss_dict:
        _plot_loss_curves(loss_dict, out_dir / "loss_curves.png")

    print(f"\n  All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
