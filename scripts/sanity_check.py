"""
Brunel network sanity-check script.

Simulates a Brunel balanced random network at any size, generates calcium
fluorescence, and saves a set of neuroscientific diagnostic plots.

J_ex is auto-scaled so that J_ex × C_E stays constant relative to the
Brunel 2000 reference (NE=8000, epsilon=0.1, J_ex=0.1 → C_E=800, J×C_E=80).
This preserves the AI-state dynamical regime across network sizes.

Usage
-----
    # N=100 (local test)
    python scripts/sanity_check.py --ne 80 --ni 20

    # N=1000 (full run)
    python scripts/sanity_check.py --ne 800 --ni 200 --sim-time 30000

    # Override J_ex manually (skip auto-scaling)
    python scripts/sanity_check.py --ne 80 --ni 20 --j-ex 1.0

    # Custom output folder
    python scripts/sanity_check.py --ne 80 --ni 20 --out results/sanity_n100

Options
-------
    --ne          Number of excitatory neurons  [default: 80]
    --ni          Number of inhibitory neurons  [default: 20]
    --epsilon     Connection probability        [default: 0.1]
    --g           |J_in|/J_ex ratio             [default: 5.0]
    --eta         External drive / threshold    [default: 2.0]
    --j-ex        Excitatory weight (mV).
                  If omitted, auto-scaled from reference so J_ex × C_E
                  equals the reference value (80 × 0.1 = 8.0).
    --sim-time    Simulation duration (ms)      [default: 10000]
    --dt          Time resolution (ms)          [default: 0.1]
    --threads     NEST CPU threads              [default: 4]
    --tau-ca      Calcium decay τ (ms)          [default: 400]
    --sigma-extra Recording noise std           [default: 0.05]
    --out         Output directory              [default: auto]
    --seed        RNG seed                      [default: 42]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ── package path ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calcium_ar.simulation import BrunelNetwork, simulate_calcium


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

# Reference from Brunel 2000: N=10,000, NE=8000, epsilon=0.1 → C_E=800, J=0.1
# J × C_E = 0.1 × 800 = 80  ← this is the conserved quantity
# (our old default NE=800 with J=0.1 gave J×C_E=8, which is 10× too small)
_REF_NE      = 8000
_REF_EPSILON = 0.1
_REF_JEX     = 0.1                         # mV  (Brunel 2000 original)
_REF_CE      = int(_REF_NE * _REF_EPSILON)  # 800
_REF_JEX_CE  = _REF_JEX * _REF_CE          # 80.0  ← conserved quantity


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ne",          type=int,   default=80)
    p.add_argument("--ni",          type=int,   default=20)
    p.add_argument("--epsilon",     type=float, default=0.1)
    p.add_argument("--g",           type=float, default=5.0)
    p.add_argument("--eta",         type=float, default=2.0)
    p.add_argument("--j-ex",        type=float, default=None,
                   help="Override auto-scaled J_ex (mV)")
    p.add_argument("--sim-time",    type=float, default=10_000.0)
    p.add_argument("--dt",          type=float, default=0.1)
    p.add_argument("--threads",     type=int,   default=4)
    p.add_argument("--tau-ca",      type=float, default=400.0)
    p.add_argument("--sigma-extra", type=float, default=0.05)
    p.add_argument("--out",         type=str,   default=None)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def spike_times_per_neuron(spikes_matrix, dt_ms):
    return {
        i: np.where(spikes_matrix[i] > 0)[0] * dt_ms
        for i in range(spikes_matrix.shape[0])
    }


def compute_isi_stats(spike_dict):
    all_isis, cvs = [], []
    for st in spike_dict.values():
        if len(st) > 1:
            isi = np.diff(st)
            all_isis.append(isi)
            cvs.append(isi.std() / isi.mean())
        else:
            cvs.append(np.nan)
    flat = np.concatenate(all_isis) if all_isis else np.array([])
    return flat, np.array(cvs)


def pop_rate_hz(spikes_matrix, dt_ms, smooth_ms=10.0):
    n = spikes_matrix.shape[0]
    rate = spikes_matrix.sum(axis=0) / (n * dt_ms / 1000.0)
    win = max(1, int(smooth_ms / dt_ms))
    return uniform_filter1d(rate, size=win)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── J_ex auto-scaling ────────────────────────────────────────────────────
    CE = max(1, int(args.epsilon * args.ne))
    if args.j_ex is not None:
        J_EX = args.j_ex
        scaling_note = f"manual override"
    else:
        # keep J_ex × C_E = reference value  →  J_ex = ref_val / C_E
        J_EX = _REF_JEX_CE / CE
        scaling_note = (
            f"auto-scaled: {_REF_JEX_CE} / C_E({CE}) = {J_EX:.4f} mV"
        )

    N  = args.ne + args.ni
    CI = max(1, int(args.epsilon * args.ni))

    # ── output directory ─────────────────────────────────────────────────────
    out_dir = (
        Path(args.out) if args.out
        else Path(__file__).parent / f"sanity_n{N}_output"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── parameter summary ────────────────────────────────────────────────────
    nu_th = 20.0 / (J_EX * CE * 20.0)   # V_th=20, tau_m=20
    header = (
        f"{'='*62}\n"
        f"  Brunel network sanity check\n"
        f"  NE={args.ne}  NI={args.ni}  N={N}\n"
        f"  epsilon={args.epsilon}  C_E={CE}  C_I={CI}\n"
        f"  J_ex={J_EX:.4f} mV  ({scaling_note})\n"
        f"  J_ex × C_E = {J_EX * CE:.2f}  (ref = {_REF_JEX_CE})\n"
        f"  g={args.g}  eta={args.eta}\n"
        f"  ν_th = {nu_th*1000:.1f} Hz  →  η×ν_th = {args.eta*nu_th*1000:.1f} Hz\n"
        f"  sim_time={args.sim_time} ms   tau_ca={args.tau_ca} ms\n"
        f"  output → {out_dir}\n"
        f"{'='*62}"
    )
    print(header)

    # ── simulate ─────────────────────────────────────────────────────────────
    net = BrunelNetwork(
        n_excitatory=args.ne,
        n_inhibitory=args.ni,
        epsilon=args.epsilon,
        g=args.g,
        eta=args.eta,
        J_ex=J_EX,
        sim_time=args.sim_time,
        dt=args.dt,
        n_threads=args.threads,
    )
    net.build()
    net.run()
    spikes, adj = net.get_results()   # (N, T), (N, N)

    time_ms = np.arange(spikes.shape[1]) * args.dt

    print("\nSimulating calcium fluorescence …")
    fluorescence = simulate_calcium(
        spikes,
        tau=args.tau_ca,
        dt=args.dt,
        amplitude=1.0,
        sigma_intra=0.01,
        sigma_extra=args.sigma_extra,
        seed=args.seed,
    )

    # ── statistics ───────────────────────────────────────────────────────────
    spike_dict  = spike_times_per_neuron(spikes, args.dt)
    mean_rates  = np.array([len(st) / (args.sim_time / 1000) for st in spike_dict.values()])
    isi_flat, cv_arr = compute_isi_stats(spike_dict)
    pop_r        = pop_rate_hz(spikes, args.dt)
    n_active     = int((mean_rates > 0).sum())

    summary = (
        f"\n=== SANITY SUMMARY  N={N} (NE={args.ne}, NI={args.ni}) ===\n"
        f"  Total spikes        : {int(spikes.sum())}\n"
        f"  Active neurons      : {n_active}/{N}  ({100*n_active/N:.0f}%)\n"
        f"  Mean firing rate    : {mean_rates.mean():.1f} ± {mean_rates.std():.1f} Hz\n"
        f"  Median firing rate  : {np.median(mean_rates):.1f} Hz\n"
        f"  Mean pop. rate      : {pop_r.mean():.1f} Hz\n"
        f"  Mean CV(ISI)        : {np.nanmean(cv_arr):.3f}  (target ≈ 1.0 for AI state)\n"
        f"  Median CV(ISI)      : {np.nanmedian(cv_arr):.3f}\n"
    )
    print(summary)
    (out_dir / "summary_stats.txt").write_text(header + summary)

    # ─────────────────────────────────────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────────────────────────────────────
    RASTER_MS = min(2_000.0, args.sim_time)
    CA_MS     = min(3_000.0, args.sim_time)
    NE        = args.ne

    # ── 1: Raster ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    for nid, st in spike_dict.items():
        st_w = st[st <= RASTER_MS]
        if len(st_w):
            ax.scatter(st_w, np.full_like(st_w, nid), s=1.5,
                       c="steelblue" if nid < NE else "tomato",
                       lw=0, rasterized=True)
    ax.axhline(NE - 0.5, color="k", lw=0.8, ls="--", label="E/I boundary")
    ax.set(xlabel="Time (ms)", ylabel="Neuron ID",
           title=f"Raster  (first {RASTER_MS:.0f} ms)  [blue=E  red=I]",
           xlim=(0, RASTER_MS), ylim=(-1, N))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "raster.png", dpi=150); plt.close(fig)

    # ── 2: Population firing rate ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(time_ms, pop_r, lw=0.7, color="k")
    ax.axhline(pop_r.mean(), color="red", ls="--",
               label=f"mean = {pop_r.mean():.1f} Hz")
    ax.set(xlabel="Time (ms)", ylabel="Population rate (Hz)",
           title="Population firing rate  (10 ms boxcar smooth)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pop_rate.png", dpi=150); plt.close(fig)

    # ── 3: Per-neuron mean rate histogram ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(mean_rates, bins=20, color="steelblue", edgecolor="white", lw=0.5)
    ax.axvline(mean_rates.mean(), color="red", ls="--",
               label=f"mean = {mean_rates.mean():.1f} Hz")
    ax.set(xlabel="Mean firing rate (Hz)", ylabel="# neurons",
           title="Per-neuron mean firing rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "firing_rate_hist.png", dpi=150); plt.close(fig)

    # ── 4: ISI distribution ───────────────────────────────────────────────────
    if len(isi_flat) > 10:
        fig, ax = plt.subplots(figsize=(6, 4))
        p99 = np.percentile(isi_flat, 99)
        bins = np.linspace(0, p99, 60)
        ax.hist(isi_flat, bins=bins, density=True,
                color="mediumpurple", edgecolor="white", lw=0.4)
        mean_isi = isi_flat.mean()
        x = np.linspace(0, p99, 300)
        ax.plot(x, np.exp(-x / mean_isi) / mean_isi, "r--", lw=1.5,
                label=f"Exponential fit  (mean={mean_isi:.0f} ms)")
        ax.set(xlabel="ISI (ms)", ylabel="Probability density",
               title="ISI distribution  (all neurons pooled)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "isi_distribution.png", dpi=150); plt.close(fig)

    # ── 5: CV of ISI ──────────────────────────────────────────────────────────
    cv_valid = cv_arr[~np.isnan(cv_arr)]
    if len(cv_valid):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(cv_valid, bins=20, color="darkorange", edgecolor="white", lw=0.5)
        ax.axvline(1.0, color="k", ls="--", lw=1.2, label="CV=1  (Poisson ideal)")
        ax.axvline(np.nanmean(cv_arr), color="red", lw=1.5,
                   label=f"mean = {np.nanmean(cv_arr):.2f}")
        ax.set(xlabel="CV(ISI)", ylabel="# neurons",
               title="CV of ISI per neuron  (AI state target: ~1.0)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "cv_isi.png", dpi=150); plt.close(fig)

    # ── 6: Calcium traces + spike overlay ─────────────────────────────────────
    active_ids = np.where(mean_rates > 0.5)[0]
    n_show = min(6, len(active_ids))
    if n_show:
        rng_p = np.random.default_rng(args.seed)
        show_ids = rng_p.choice(active_ids, size=n_show, replace=False)
        t_win = time_ms[time_ms <= CA_MS]
        win_T = len(t_win)

        fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.2 * n_show), sharex=True)
        if n_show == 1:
            axes = [axes]
        for ax, nid in zip(axes, show_ids):
            ax.plot(t_win, fluorescence[nid, :win_T], lw=0.8, color="green")
            for s in spike_dict[nid][spike_dict[nid] <= CA_MS]:
                ax.axvline(s, color="k", lw=0.5, alpha=0.4)
            kind = "E" if nid < NE else "I"
            ax.set_ylabel(f"n{nid} ({kind})", fontsize=8)
            ax.set_yticks([])
        axes[-1].set_xlabel("Time (ms)")
        axes[0].set_title(
            f"Calcium traces  (τ={args.tau_ca} ms)  |  black ticks = true spikes"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "calcium_traces.png", dpi=150); plt.close(fig)

    # ── 7: Adjacency matrix ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.abs(adj).max() or 1.0
    im = ax.imshow(adj, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.axhline(NE - 0.5, color="k", lw=0.8, ls="--")
    ax.axvline(NE - 0.5, color="k", lw=0.8, ls="--")
    ax.set(xlabel="Target neuron", ylabel="Source neuron",
           title="Ground-truth adjacency  [blue=E, red=I]")
    plt.colorbar(im, ax=ax, label="Synaptic weight (mV)")
    for (r, c, lbl) in [(NE/2, NE/2, "E→E"), (NE+args.ni/2, NE/2, "E→I"),
                        (NE/2, NE+args.ni/2, "I→E"), (NE+args.ni/2, NE+args.ni/2, "I→I")]:
        ax.text(r, c, lbl, ha="center", va="center", fontsize=9, color="gray")
    fig.tight_layout()
    fig.savefig(out_dir / "adjacency_matrix.png", dpi=150); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # CHECKLIST
    # ─────────────────────────────────────────────────────────────────────────
    print("""
╔══════════════════════════════════════════════════════════════╗
║  WHAT TO LOOK FOR                                            ║
╠══════════════════════════════════════════════════════════════╣
║  raster.png                                                  ║
║    ✅ salt-and-pepper, no vertical stripes or dead rows      ║
║    ❌ synchronous bursts, propagating waves, silent neurons   ║
║                                                              ║
║  pop_rate.png                                                ║
║    ✅ roughly stationary fluctuations                        ║
║    ❌ systematic drift, burst+silence oscillations           ║
║                                                              ║
║  firing_rate_hist.png                                        ║
║    ✅ unimodal, peak 5–20 Hz, most neurons active            ║
║    ❌ bimodal (some silent, some bursting)                   ║
║                                                              ║
║  isi_distribution.png                                        ║
║    ✅ close to exponential                                   ║
║    ❌ sharp single peak (regular firing)                     ║
║                                                              ║
║  cv_isi.png                                                  ║
║    ✅ mean CV ≈ 1.0  (irregular, Poisson-like)               ║
║    ❌ CV << 1 (too regular)  or  CV >> 1.5 (bursting)       ║
║                                                              ║
║  calcium_traces.png                                          ║
║    ✅ visible bumps at spike ticks, exponential decay        ║
║    ❌ flat line or always saturated                          ║
║                                                              ║
║  adjacency_matrix.png                                        ║
║    ✅ sparse, visible E/I block structure                    ║
╚══════════════════════════════════════════════════════════════╝
""")

    print(f"All outputs → {out_dir}\nDone.")


if __name__ == "__main__":
    main()
