"""
Pool K independent chunks (same fixed connectivity, different noise seed --
see BrunelNetwork's adjacency_override) into one long-recording estimate.

Each chunk directory (results/<name>_T<T>k/seed<i>/) holds Cxx.npy, Cyx.npy
(from --save-moments) and adj_true.npy (identical across all chunks, verified
below). Since every chunk is the SAME length, an unweighted average of the
already-normalized Cxx_i is exactly equal to pooling the raw sums -- no need
to touch the accumulator internals.

Usage:
  python scripts/pool_chunks_and_solve.py --dir $WS/results/n12500_kpool_T12000000k \
      --out $WS/results/n12500_kpool_pooled
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from calcium_ar.solvers.from_moments import ols_from_moments
from analyze_run import edge_index, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir containing seed*/{Cxx,Cyx,adj_true}.npy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--sample", type=int, default=20_000_000)
    args = ap.parse_args()

    data = Path(args.dir)
    seed_dirs = sorted(p for p in data.glob("seed*") if (p / "Cxx.npy").exists())
    if not seed_dirs:
        raise SystemExit(f"no seed*/Cxx.npy under {data}")
    print(f"pooling {len(seed_dirs)} chunks: {[d.name for d in seed_dirs]}")

    adj_true = np.load(seed_dirs[0] / "adj_true.npy")
    Cxx_sum = np.zeros_like(np.load(seed_dirs[0] / "Cxx.npy"))
    Cyx_sum = np.zeros_like(Cxx_sum)
    for d in seed_dirs:
        a = np.load(d / "adj_true.npy")
        if not np.allclose(a, adj_true):
            raise SystemExit(f"{d} has a DIFFERENT adj_true -- chunks are not the same "
                              f"network, refusing to pool (check adjacency_file was used "
                              f"consistently)")
        Cxx_sum += np.load(d / "Cxx.npy")
        Cyx_sum += np.load(d / "Cyx.npy")

    Cxx = Cxx_sum / len(seed_dirs)
    Cyx = Cyx_sum / len(seed_dirs)
    A = ols_from_moments(Cxx, Cyx)

    N = adj_true.shape[0]
    rng = np.random.default_rng(0)
    i, j, sampled = edge_index(N, args.sample, rng)
    g = np.asarray(adj_true[j, i], dtype=np.float64)          # axon=row convention
    a = np.asarray(A[i, j], dtype=np.float64)
    r = score(a, g, args.density)
    r.update(n_chunks=len(seed_dirs), n_scored=len(g), sampled=bool(sampled))
    print(json.dumps(r, indent=2))

    out = Path(args.out) if args.out else data.parent / (data.name + "_pooled")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "Cxx_pooled.npy", Cxx)
    np.save(out / "Cyx_pooled.npy", Cyx)
    np.save(out / "A_ols.npy", A.astype(np.float32))
    np.save(out / "adj_true.npy", adj_true.astype(np.float32))
    with open(out / "metrics.json", "w") as fh:
        json.dump(r, fh, indent=2)
    print(f"\nwrote {out}/metrics.json + A_ols.npy")


if __name__ == "__main__":
    main()
