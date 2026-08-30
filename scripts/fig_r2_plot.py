"""
R.2 (calcium-observation section) — PLOT stage, runs LOCALLY.

Reads r2_data.npz and renders the PRIMARY figure as a 2x2 panel
(rows = ROC-AUC, correlation):

  column A  vs camera frame interval (ms), dye tau FIXED
      deconv_rate + raw_rate
  column B  vs spike bin size (ms) : the ceiling curve (DAAD-style ~5 ms optimum)

Camera and tau are swept ONE AT A TIME in the compute stage (see
fig_r2_compute.py) so the camera panel answers one physical question on its
own. The dye-tau panel (camera fixed) is NOT in the primary figure — in this
regime it's much flatter than the camera panel and was crowding the main read;
it's saved separately (--out-tau) so the analysis isn't lost.

A SECONDARY figure (--out-ratio) reproduces the old "does only the ratio
matter?" collapse view, for anyone who wants to check whether the two curves
line up when replotted against dt/tau after the fact.

A METRIC-SET figure (--out-metrics) plots the density-quantile precision /
recall / F1 and the excitatory- vs inhibitory-source recall split (see
scripts/r2_metrics.py) over the same camera-rate and spike-bin columns. It is
skipped automatically for an r2_data.npz produced before those columns existed.

All cosmetics live here. Saves PDF + PNG.

Usage:
  python scripts/fig_r2_plot.py --data results/fig_data/r2_data.npz --out figures/fig_R2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

STYLE_RATE = {
    "deconv_rate": dict(color="#2a78d6", marker="o", label="deconvolved"),
    "raw_rate":    dict(color="#e34948", marker="o", label="raw calcium"),
}
STYLE_TAU = {
    "deconv_tau": dict(color="#2a78d6", marker="o", label="deconvolved"),
    "raw_tau":    dict(color="#e34948", marker="o", label="raw calcium"),
}
STYLE_RATIO = {
    "deconv_tau":  dict(color="#2a78d6", marker="o", label="deconvolved (dye τ)"),
    "deconv_rate": dict(color="#2a78d6", marker="s", ls="--", label="deconvolved (camera dt)"),
    "raw_tau":     dict(color="#e34948", marker="o", label="raw calcium (dye τ)"),
    "raw_rate":    dict(color="#e34948", marker="s", ls="--", label="raw calcium (camera dt)"),
}

# Nominal default smoothing window left after deconvolution (fig_r2_compute.py's
# SMOOTH_MS) — marked on the bin-size axis alongside the raw-calcium (~tau) and
# tuned-best-window (measured) reference lines.
SMOOTH_MS = 3.1

# y-axis label per metric column in r2_data.npz (see scripts/r2_metrics.py).
MEASURE_LABEL = {
    "auc": "ROC-AUC",
    "corr": "correlation",
    "precision": "precision",
    "recall": "recall",
    "f1": "F1",
    "recall_exc": "recall (exc. sources)",
    "recall_inh": "recall (inh. sources)",
}
# metric-set figure (--out-metrics): threshold-based numbers, one row each
METRIC_ROWS = ["precision", "recall", "f1", "recall_exc", "recall_inh"]


def _ylabel(measure):
    return MEASURE_LABEL.get(measure, measure)


def _ceiling(ax, z, measure):
    ceiling = np.nanmax(z[f"spikes_{measure}"])       # best binned-spikes score
    ax.axhline(ceiling, color=fs.MUTED, ls=":", lw=1.2)
    ax.text(0.02, ceiling, "best binned spikes", color=fs.MUTED, va="bottom",
            ha="left", fontsize=8, transform=ax.get_yaxis_transform())


def rate_panel(ax, z, measure):
    """vs camera frame interval (ms), dye tau fixed."""
    for key, st in STYLE_RATE.items():
        x, y = z[f"{key}_x"], z[f"{key}_{measure}"]
        ax.plot(x, y, lw=1.8, ms=6, **st)
    _ceiling(ax, z, measure)
    ax.axvline(float(z["fixed_tau_ms"]), color=fs.MUTED, lw=1.0, ls="-.")
    ax.set_xscale("log")
    ax.set(xlabel="camera frame interval (ms)",
           ylabel=_ylabel(measure), ylim=(0, 1.02))
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)


def tau_panel(ax, z, measure):
    """vs dye tau (ms), camera rate fixed."""
    for key, st in STYLE_TAU.items():
        x, y = z[f"{key}_x"], z[f"{key}_{measure}"]
        ax.plot(x, y, lw=1.8, ms=6, **st)
    _ceiling(ax, z, measure)
    cam = float(z["fixed_cam_ms"])
    ax.axvline(cam, color=fs.MUTED, lw=1.0, ls="-.")
    ax.text(cam, 0.02, " camera dt", color=fs.MUTED, fontsize=7.5, rotation=90,
            va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set(xlabel="dye τ (ms)",
           ylabel=_ylabel(measure), ylim=(0, 1.02))
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)


def spike_panel(ax, z, measure, tau=None, smooth=None):
    ax.plot(z["spikes_x"], z[f"spikes_{measure}"], "o-", color=fs.INK, lw=1.8, ms=6)
    _ceiling(ax, z, measure)
    ax.set_xscale("log")
    ax.set(xlabel="spike bin size (ms)",
           ylabel=_ylabel(measure), ylim=(0, 1.02))
    if tau is not None:
        xmax = max(float(z["spikes_x"].max()), tau) * 1.6
        ax.set_xlim(float(z["spikes_x"].min()) * 0.8, xmax)
        ax.axvline(tau, color="#e34948", lw=1.3, ls="--", zorder=1)
        ax.text(tau, 0.04, f"raw calcium ~{tau:.0f} ms", color="#e34948", fontsize=7.5,
               rotation=90, va="bottom", ha="right")
        ax.axvline(SMOOTH_MS, color="#2a78d6", lw=1.3, ls="--", zorder=1)
        ax.text(SMOOTH_MS, 0.04, f"deconvolved (default) ~{SMOOTH_MS:.0f} ms",
               color="#2a78d6", fontsize=7.5, rotation=90, va="bottom", ha="right")
    if smooth is not None:
        rx, ry = z["spikes_x"], z[f"spikes_{measure}"]
        xmax = float(rx[int(np.argmax(ry))])          # x of the reference curve's OWN peak
        ax.axvline(xmax, color="#2a78d6", lw=1.8, ls="-", zorder=1)
        ax.text(xmax, 0.04, f"max ~{xmax:.0f} ms", color="#2a78d6", fontsize=7.5,
               fontweight="bold", rotation=90, va="bottom", ha="left")
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)


def ratio_panel(ax, z, measure):
    for key, st in STYLE_RATIO.items():
        x, y = z[f"{key}_ratio"], z[f"{key}_{measure}"]
        order = np.argsort(x)
        ax.plot(x[order], y[order], lw=1.8, ms=6, **st)
    _ceiling(ax, z, measure)
    ax.set_xscale("log")
    ax.set(xlabel="dt / τ   (calcium blur)",
           ylabel=_ylabel(measure), ylim=(0, 1.02))
    ax.grid(True, color=fs.GRID, lw=0.6); ax.set_axisbelow(True); fs.despine(ax)


def net_title(z):
    return f"{str(z['net'])}, N={int(z['N'])}, T={int(z['T_ms'])//1000}k ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/fig_data/r2_data.npz")
    ap.add_argument("--smooth-data", default=None,
                    help="optional r2_smooth_data.npz (deconv. window sweep) to overlay "
                         "on the binned-spikes reference panel")
    ap.add_argument("--out", default="figures/fig_R2")
    ap.add_argument("--out-ratio", default="figures/fig_R2_ratio",
                    help="secondary collapse-view figure (dt/tau shared axis)")
    ap.add_argument("--out-tau", default="figures/fig_R2_tau",
                    help="appendix figure: dye tau sweep, camera fixed")
    ap.add_argument("--out-metrics", default="figures/fig_R2_metrics",
                    help="metric-set figure: density-quantile precision / "
                         "recall / F1 / exc.- vs inh.-source recall, camera "
                         "rate + spike-bin columns (skipped if the npz predates "
                         "these columns)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    fs.apply_style()
    z = np.load(args.data, allow_pickle=False)
    smooth = np.load(args.smooth_data, allow_pickle=False) if args.smooth_data else None

    # ---- primary: camera rate + spike-bin reference ------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8),
                             gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.88, bottom=0.09,
                        hspace=0.32, wspace=0.28)
    rate_panel(axes[0, 0], z, "auc");  axes[0, 0].set_title(
        f"camera rate  (dye τ fixed at {float(z['fixed_tau_ms']):.0f} ms)")
    rate_panel(axes[1, 0], z, "corr")
    tau = float(z["fixed_tau_ms"]) if smooth is not None else None
    spike_panel(axes[0, 1], z, "auc", tau=tau, smooth=smooth)
    axes[0, 1].set_title("binned spikes (reference)")
    spike_panel(axes[1, 1], z, "corr", tau=tau, smooth=smooth)
    axes[0, 0].legend(loc="lower left", fontsize=8.5)

    title = args.title or f"Calcium observation  —  {net_title(z)}"
    fig.suptitle(title, fontsize=13, color=fs.INK, x=0.06, ha="left", y=0.975)
    fs.save(fig, args.out)

    # ---- appendix: dye tau sweep, camera fixed ------------------------------ #
    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.6))
    fig3.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.14, wspace=0.28)
    tau_panel(axes3[0], z, "auc")
    tau_panel(axes3[1], z, "corr")
    axes3[0].legend(loc="lower left", fontsize=8)
    fig3.suptitle(
        f"Dye τ sweep, camera fixed at {float(z['fixed_cam_ms']):.0f} ms  —  {net_title(z)}",
        fontsize=12, color=fs.INK, x=0.08, ha="left", y=0.97)
    fs.save(fig3, args.out_tau)

    # ---- secondary: does the ratio alone explain it? ----------------------- #
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.6))
    fig2.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.14, wspace=0.28)
    ratio_panel(axes2[0], z, "auc");  axes2[0].set_title("blur (dt/τ): does only the ratio matter?")
    ratio_panel(axes2[1], z, "corr")
    axes2[0].legend(loc="lower left", fontsize=8)
    fig2.suptitle(f"Ratio collapse check  —  {net_title(z)}",
                 fontsize=12, color=fs.INK, x=0.08, ha="left", y=0.97)
    fs.save(fig2, args.out_ratio)

    # ---- metric set: density-quantile precision / recall / exc-inh split --- #
    rows = [m for m in METRIC_ROWS if f"spikes_{m}" in z.files]
    if rows:
        dens = float(z["density"]) if "density" in z.files else None
        op = f"top {100 * dens:.0f}% of |A|" if dens is not None else "density operating point"
        figm, axm = plt.subplots(len(rows), 2, figsize=(11.5, 3.3 * len(rows)),
                                 gridspec_kw=dict(width_ratios=[1.15, 1.0]),
                                 squeeze=False)
        figm.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.06,
                             hspace=0.42, wspace=0.28)
        for r, meas in enumerate(rows):
            rate_panel(axm[r, 0], z, meas)
            spike_panel(axm[r, 1], z, meas)
        axm[0, 0].set_title(
            f"camera rate  (dye τ fixed at {float(z['fixed_tau_ms']):.0f} ms)")
        axm[0, 1].set_title("binned spikes (reference)")
        axm[0, 0].legend(loc="lower left", fontsize=8.5)
        figm.suptitle(f"Calcium observation — metric set ({op})  —  {net_title(z)}",
                      fontsize=12, color=fs.INK, x=0.08, ha="left", y=0.985)
        fs.save(figm, args.out_metrics)


if __name__ == "__main__":
    main()
