"""
R.2 (calcium-observation section) — RESCORE stage. Pure numpy, runs LOCALLY in
seconds.

fig_r2_compute.py freezes the expensive part of every sweep point (streaming
moments -> OLS solve) as <est-dir>/<kind>_<param>.npy plus a _meta.npz
(adj, n_exc). This script re-derives the connectivity metric set
(scripts/r2_metrics.py) from those frozen estimates and rewrites r2_data.npz —
so adding a metric, or retuning the operating-point density, costs a few
seconds instead of the ~1h cluster sweep.

The x / ratio axes are copied from the input --data npz (keyed by the same
<kind>_<param> the .npy files are named after); only the metric columns and
density are replaced.

Usage:
  python scripts/fig_r2_rescore.py \
      --data results/fig_data/r2_data_r4.npz \
      --est-dir results/fig_data/r2_data_r4_estimates \
      --out results/fig_data/r2_data_r4.npz \
      --density 0.10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2_metrics import METRICS, metrics_from_A

KINDS = ("spikes", "deconv_tau", "deconv_rate", "raw_tau", "raw_rate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="r2_data.npz from fig_r2_compute.py (x/ratio axes are "
                         "reused; metric columns are recomputed)")
    ap.add_argument("--est-dir", default=None,
                    help="per-point estimate dir (default: "
                         "<data-without-.npz>_estimates/)")
    ap.add_argument("--density", type=float, default=None,
                    help="operating-point density for precision/recall/F1/"
                         "recall_exc/recall_inh; default: the value stored in "
                         "_meta.npz")
    ap.add_argument("--out", required=True,
                    help="output npz (may be the same path as --data)")
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=False)
    est_dir = Path(args.est_dir) if args.est_dir else (
        Path(args.data).with_suffix("").parent
        / f"{Path(args.data).with_suffix('').name}_estimates")
    meta = np.load(est_dir / "_meta.npz", allow_pickle=False)
    adj = np.asarray(meta["adj"], dtype=np.float64)
    n_exc = int(meta["n_exc"])
    density = float(args.density) if args.density is not None else float(meta["density"])
    print(f"rescoring from {est_dir}  (n_exc={n_exc}, density={density})", flush=True)

    save = {k: data[k] for k in data.files}
    save["density"] = density
    save["metrics"] = np.array(METRICS)

    n_pts = 0
    for kind in KINDS:
        xkey = f"{kind}_x"
        if xkey not in data.files:
            continue
        xs = data[xkey]
        cols = {mn: np.full(len(xs), np.nan) for mn in METRICS}
        for i, x in enumerate(xs):
            npy = est_dir / f"{kind}_{float(x):g}.npy"
            if not npy.exists():
                print(f"  WARN missing {npy.name} -- leaving NaN", flush=True)
                continue
            A = np.load(npy).astype(np.float64)
            mets = metrics_from_A(A, adj, n_exc, density)
            for mn in METRICS:
                cols[mn][i] = mets[mn]
            n_pts += 1
        for mn in METRICS:
            save[f"{kind}_{mn}"] = cols[mn]

    tmp = f"{args.out}.tmp.npz"
    np.savez(tmp, **save)
    Path(tmp).replace(args.out)
    print(f"\nrescored {n_pts} points -> {args.out}")


if __name__ == "__main__":
    main()
