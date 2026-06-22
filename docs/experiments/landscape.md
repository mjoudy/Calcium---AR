# Parameter Landscape — overview & what to tune

One-page map of **every knob in the pipeline**: what it does, how much it
matters for connectivity recovery, the best setting found so far, and whether
it still needs work — especially before scaling N up and moving to HPC.

This is the consolidating view. Drill-down lives in the other docs:
- `notebook.md` — dated log + "findings at a glance" (the evidence).
- `methods_overview.md` — the 15-method post-processing scoreboard.
- `open_directions.md` — what is still untested.
- `README.md` — the recommended end-to-end pipeline.

> **Scope caveat (read this first).** Everything below was measured on **one**
> network: N=100 (80 E / 20 I), Brunel balanced, g=5, τ=100 ms, T=50 k samples.
> "Best setting" = best *on that network*. The numbers are settled; their
> **generalization to other N / regimes is the open question** the HPC phase exists
> to answer. See the last section.

---

## 1. What "the landscape" is

The pipeline has four stages, each with its own parameters:

```
SIMULATION ─► CALCIUM ─► PREPROCESSING ─► SOLVER (+ POST-PROCESS)
 (the regime)  (the signal)  (recover feed)   (infer connectivity Â)
```

Parameters split into two fundamentally different roles:

- **Estimator knobs** — how we *recover* connectivity from a fixed dataset
  (lag, solver, regularization, post-processing). These are what you *tune* to
  do better. **Mostly solved on N=100.**
- **Regime knobs** — what network/signal you're *trying* to recover
  (N, density, g, τ, T, noise). These *define the problem*. Held fixed so far;
  they become the **axes you vary on HPC** to test generalization.

---

## 2. Master parameter table

`config` = field in [`ExperimentConfig`](../../calcium_ar/experiments/config.py).
Importance = how much it moves connectivity recovery.

| # | config field | stage | role | importance | best setting found | what it does / finding | status |
|---|---|---|---|---|---|---|---|
| **Tier 1 — dominant estimator knobs (set these right or nothing works)** ||||||||
| 1 | `lag_ms` | solver | estimator | ★★★★★ | **1.5 ms** (= synaptic delay = 15 samples) | Pearson peaks sharply at the synaptic delay; off by 0.5 ms → ≈ 0. **Default 10 ms in config is wrong** — earlier "regularization failed" was this. | settled (1 net) |
| 2 | `solver` + `lam` | solver | estimator | ★★★★☆ | **fista** (Elastic Net), `lam`(L1) **≈ 3e-3** | Regularization: F1 0.30→0.62, precision 0.18→0.84. Useful λ range tiny (1e-3…1e-2); ≥3e-2 zeros 92 % and kills inhibition first. **Default `ridge`/`lam=1.0` is not the recommended pipeline.** | settled |
| 3 | post-processing | solver | estimator | ★★★★☆ | **Dale-reg → balance_nz rescale** | Not a config field yet — chosen in scripts. Dale's law = strongest unsupervised lever (type 0.83→0.92, perfect Dale); balance rescale fixes magnitude (Pearson →0.585, > oracle). | settled |
| **Tier 2 — secondary estimator knobs (characterized, lower leverage)** ||||||||
| 4 | `lam_l2` | solver | estimator | ★★☆☆☆ | small (≈1e-3) | L2 barely matters at this lag (neurons not very collinear); L1 does the work. | settled |
| 5 | preprocessing (on/off) | preproc | estimator | ★★★☆☆ | **always preprocess** | Preprocessed feed recovers ~95 % of spike-level Pearson. **Raw calcium gives a fake-high Pearson** with worse AUC/type/precision — preprocessing removes that confound. | settled |
| 6 | `smooth_window_ms` | preproc | estimator | ★★☆☆☆ (?) | 0.5 ms (provisional) | Held fixed; worked fine. Interaction with `lag_ms` **never swept**. | untested |
| 7 | `tau_method` | preproc | estimator | ★☆☆☆☆ | ransac | Robust τ-estimation method. Not shown to matter much; not swept. | untested |
| 8 | `spike_cut_window` | preproc | estimator | ★☆☆☆☆ | 5 | Half-window removed around spikes. Not swept. | untested |
| **Tier 3 — regime knobs (define the problem; the HPC generalization axes)** ||||||||
| 9 | `n_excitatory`/`n_inhibitory` (N) | network | regime | ★★★★★ (cost+validity) | N=100 tested | The scaling axis. All conclusions are on N=100. Bigger N = more collinearity, harder regression, far higher cost. **Primary HPC variable.** | open |
| 10 | `sim_time` (T) | network | regime | ★★★★☆ | more = better precision | Precision & Pearson still *rising* at 50 k (variance-limited → reg helps). But E/I ratio *worsens* toward ~0.3 with more data (bias-limited → needs post-proc). Likely must grow with N. | partial |
| 11 | `g` | network | regime | ★★★☆☆ | 5.0 tested | E/I weight ratio. Currently *estimated back* from balance for the magnitude rescale (g≈4–6 vs true 5). Untested at other g. | open |
| 12 | `epsilon` | network | regime | ★★★☆☆ | 0.1 tested | Connection density. More density = more collinearity. Untested elsewhere. | open |
| 13 | `tau` | calcium | regime | ★★★☆☆ | 100 ms tested | Calcium decay. The slower **400 ms** regime (harder, more smearing) untested. | open |
| 14 | `eta` | network | regime | ★★☆☆☆ | 2.0 tested | External drive → firing-rate regime. Affects how much data you need. Untested. | open |
| 15 | `J_ex` | network | regime | ★★☆☆☆ | rule-bound | **Must rescale with N**: J_ex × C_E = 80 (C_E = ε·N_E). NE=800→1.0, NE=1000→0.8, NE=80→10.0. Get this wrong and the network is in the wrong dynamical regime. | rule known |
| 16 | `sigma_extra` | calcium | regime | ★☆☆☆☆ | minor | Recording noise costs only ~0.02 Pearson at the optimum. | settled |
| 17 | `sigma_intra` | calcium | regime | ★☆☆☆☆ | 0.01 | Intracellular noise. Not separately swept. | untested |
| 18 | `amplitude` | calcium | regime | ★☆☆☆☆ | 1.0 | ΔF per spike; sets SNR scale. Not swept. | untested |
| **Compute knobs (no effect on accuracy — they decide feasibility at scale)** ||||||||
| 19 | `device` | solver | compute | — | `cuda` on HPC | CPU/GPU. Irrelevant for accuracy; decisive for whether big N runs at all. | scale-up |
| 20 | `chunk_size` | solver | compute | — | fit to memory | Time steps per chunk/mini-batch. The whole point of the chunked rewrite: keeps O(N²) not O(N·T) in memory. | scale-up |
| 21 | `n_epochs`, `lr` | solver | compute | — | torch only | Only for gradient solvers. Tune for convergence, not accuracy ceiling. | scale-up |
| 22 | `n_threads` | network | compute | — | many on HPC | NEST CPU threads for the simulation. | scale-up |
| 23 | `dt` | network | both | — | 0.1 ms | Time resolution. Sets the sample↔ms conversion (1.5 ms = 15 samples). Changing it reindexes lag/smoothing. | fixed |
| 24 | `seed`, `data_path`, `name`, `output_dir` | meta | meta | — | — | Reproducibility / IO. `data_path` → build dataset once, reuse across a sweep. | — |

---

## 3. The short answer — what to actually consider

If someone asks "which parameters matter?", the honest ranking:

1. **`lag_ms` = 1.5 ms.** Single most important knob. It is the synaptic delay,
   not a free hyperparameter — and the config default (10 ms) is wrong.
2. **Regularization: Elastic Net, L1 ≈ 3e-3.** Buys detection. Narrow useful range.
3. **Post-processing: Dale-reg → balance rescale.** Buys type + magnitude,
   fully unsupervised, near/above the ground-truth ceiling.
4. **Always preprocess** (don't feed raw calcium — fool's-gold Pearson).
5. Everything else is either second-order (L2, smoothing, noise) or a **regime
   knob** you hold fixed until the HPC generalization phase.

Recommended deployable pipeline (from README):
`preprocess → lag 1.5 ms → EN(L1≈3e-3) → Dale-reg → balance_nz rescale`.

⚠️ **The `ExperimentConfig` defaults are stale** relative to these findings
(`lag_ms=10`, `solver=ridge`, `lam=1.0`). Sweeps pass explicit configs, so results
are fine — but updating the defaults (or adding a `recommended()` constructor)
would prevent a wrong-lag run on HPC. *(Suggested follow-up, not done.)*

---

## 4. Before scaling N up / moving to HPC

The method is validated on **one** N=100 network. Moving to larger networks is
not just "more compute" — it changes the problem. Three separate concerns:

### (a) Conclusions that must be re-validated at larger N
- **Lag optimum = synaptic delay.** Should hold (it's physical, not fitted), but
  confirm the 1.5 ms peak survives at N=1000 and across seeds.
- **The ~0.37 Pearson / ~0.90 AUC ceiling.** Larger N = more neurons = more
  collinearity / shared input → the regression ceiling may *drop*. This is the
  biggest scientific risk in scaling.
- **The λ sweet spot (1e-3…1e-2).** Optimal regularization strength typically
  shifts with N and T — re-sweep, don't assume.
- **Balance-based g estimate.** Derived for this E/I split; re-check the formula
  at other N_E/N_I ratios.

### (b) Things that change mechanically with N (get them right up front)
- **`J_ex` scaling rule:** J_ex × C_E = 80 (C_E = ε·N_E). Wrong J_ex → wrong
  dynamical regime → meaningless inference.
- **`sim_time` (T) likely scales with N.** Precision was still climbing at 50 k for
  N=100; bigger N (more parameters per row) needs more samples. Budget for it.
- **Solver/compute path:** move from numpy/sklearn to **torch on GPU**
  (`solver=torch_*`, `device=cuda`), tune `chunk_size` to fit memory. The chunked
  rewrite exists precisely so memory stays O(N²) not O(N·T).

### (c) Generalization sweep this enables (the actual HPC experiment)
Vary the **regime knobs** one axis at a time and check the pipeline holds:
`N`, `epsilon` (density), `g`, `eta` (rate), `tau` (incl. 400 ms). This is
`open_directions.md §E` — the deferred "second-level" validation. The local work
was building/validating the estimator; HPC is for proving it generalizes.

**Suggested first HPC step:** re-run the exact recommended pipeline at N=1000 (one
network, correct J_ex, T scaled up, GPU solver) and check whether the four Tier-1
conclusions survive. If they do, fan out the regime sweep. If the ceiling collapses
at N=1000, that result reshapes the whole project — so it's the right first probe.
