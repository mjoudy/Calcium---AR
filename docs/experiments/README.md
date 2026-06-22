# Connectivity-Inference Experiments — START HERE

Entry point for the whole investigation (June 2026). If you're returning after a break,
read this first; it summarizes what was done, what was concluded, the recommended
pipeline, and where everything lives — so you don't re-run anything.

**Scope:** all results below are on ONE simulated network: N=100 (80 exc / 20 inh),
Brunel balanced random net, g=5 (true weights +10 / −50), calcium τ=100 ms, T=50 k
samples, dt=0.1 ms. Dataset: `results/solver_comparison_N100/dataset/`.
**Generalization to other networks is NOT yet done** (see open_directions.md §E).

---

## TL;DR — the recommended pipeline

```
preprocess (feed = dC/dt + C/τ, Savitzky-Golay)
  → AR regression at lag = 1.5 ms (= synaptic delay = 15 samples)
  → Elastic Net  (L1 ≈ 3e-3, L2 ≈ 1e-3)              [detection]
  → Dale-regularization (sign-constrained refit)      [type/sign]
  → balance_nz rescale (g from network balance)        [magnitude]
```

Fully unsupervised (no ground truth used to produce the estimate). Scores vs plain OLS:

| direction | metric | OLS | full pipeline | GT oracle |
|---|---|---|---|---|
| detection | F1 / AUC | 0.30 / 0.87 | **0.49 / 0.86** | 0.51 / 0.81 |
| type | accuracy / Dale | 0.83 / 0.54 | **0.92 / 1.00** | 0.84 |
| magnitude | Pearson | 0.35 | **0.585** (> oracle) | 0.553 |

Best alternative: swap Dale-reg for `dale+mixture` if you want max Spearman (0.50)
instead of max Pearson.

---

## Key findings (each settled by an experiment)

1. **Lag is the dominant knob; optimum = the synaptic delay (1.5 ms).** Off by 0.5 ms →
   correlation ≈ 0. The old default (1.0 ms) was wrong. *(oracle_ladder)*
2. **Calcium preprocessing is NOT the bottleneck.** Noisy-calcium feed recovers ~95 % of
   spike-level performance — but you must preprocess (raw calcium gives a fake-high
   Pearson). *(oracle_ladder)*
3. **The estimator ceiling is ~Pearson 0.37 / AUC 0.90 even on perfect spikes** — the
   limit is the regression, not the calcium. We operate near this ceiling. *(oracle_ladder)*
4. **Why weights are small:** the regression removes shared/common input and keeps only
   each neuron's unique contribution — which is intrinsically tiny. Not a bug. *(analysis)*
5. **Good at regions, weak per-cell:** type/sign is reliable (averaging cancels noise);
   single connections are noisy (precision low). *(analysis + numerical demo)*
6. **Detection is variance-limited** (still improves with data) → regularization helps it.
   **Magnitude is bias-limited** → regularization can't, but balance-rescale can. *(data_size_test)*
7. **Regularization (Elastic Net):** F1 0.30 → 0.62; useful λ range is tiny (1e-3…1e-2);
   bigger λ zeros everything (why earlier Lasso "failed"). *(regularization_test)*
8. **Magnitude is recoverable unsupervised** via g estimated from network balance
   (g ≈ N_E·νE / N_I·νI ≈ 4–6 vs true 5): Pearson → 0.553 = oracle. *(balance)*
9. **Dale's law is the strongest unsupervised lever**, and **in-solver (regularization) >
   post-hoc cleanup**. *(dale_reg_test)*
10. **One pipeline does all three** once the rescale uses non-zero entries. *(composition fix)*
11. **Remaining gap:** E/I *ratio* still 0.9 (not 5) — capped by 70 % inhibitory type-ID.

---

## Where everything lives

**Docs (`docs/experiments/`):**
- `README.md` — this file (entry point).
- `landscape.md` — **parameter landscape overview**: every knob in one table (importance,
  best setting, status) + what to tune + what to re-validate before scaling N up / HPC.
- `notebook.md` — dated lab log, every experiment with conclusions + a "findings at a
  glance" table + open questions.
- `methods_overview.md` — scoreboard of all 16 methods (detection/type/magnitude +
  unsupervised quality) with each method's mechanism & assumptions (for theory vetting).
- `open_directions.md` — untested / unfinished directions.

**Scripts (`scripts/`) — each is one experiment, re-runnable:**
- `sanity_check.py` — network/calcium diagnostics.
- `oracle_ladder.py` — lag sweep + spikes-vs-calcium ladder (findings 1–3).
- `data_size_test.py` — variance vs bias (finding 6).
- `regularization_test.py` — Elastic Net λ sweep (finding 7).
- `postprocess_test.py` — rescaling methods (magnitude).
- `combine_test.py` — regularize-then-rescale.
- `unsup_rescale_test.py` — unsupervised type-inference rules.
- `dale_reg_test.py` — Dale as regularization vs post-hoc (finding 9).
- `methods_overview.py` — regenerates the full scoreboard + the recommended pipeline.
- `ledger.py` — query/rebuild the run ledger (`effect`, `rebuild`, `show`).

**Results (`results/*/`):** each experiment dir has a `ledger.csv` (one row per run, full
config + metrics). Master query: `python scripts/ledger.py rebuild` then
`python scripts/ledger.py effect lag_ms`.

---

## Status

- ✅ Method development on N=100 — complete, near the achievable ceiling.
- ⬜ E/I absolute ratio (inhibitory-ID lever) — see open_directions §A.
- ⬜ Generalization across networks/sizes/τ — open_directions §E (deferred by choice).
- ⬜ Theory pass on method admissibility — open_directions §F (user-led).

## How to reproduce the headline result
```
python scripts/methods_overview.py     # regenerates methods_overview.md + the pipeline row
```
