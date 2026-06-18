# Open Directions (untested / unfinished)

Running list of things we have NOT yet tested or finished, so nothing is lost. Tested
findings live in `notebook.md`; the method scoreboard is `methods_overview.md`.
Updated 2026-06-19.

## A. Finishing the current pipeline (highest value)
- ~~**Fix the detection+magnitude composition.**~~ **DONE (2026-06-19):**
  `rescale_balance_nz` (medians over non-zero entries) composes. Full pipeline
  `EN_daleReg → balance_nz`: F1 0.49, type 0.92, Pearson 0.585 (> oracle), perfect Dale.
- **Better inhibitory type-inference.** Strongest-entry reaches 0.70 on inhibitory
  neurons; this caps both the E/I ratio (stuck ~1.6) and the dale/balance combos. Ideas:
  iterative type refinement (re-estimate types from the constrained fit, repeat);
  cluster neurons by connectivity fingerprint; use temporal/lag features.
- **Sharpen the g estimate.** Balance formula drops the external-drive term (g≈4.4 vs
  true 5 even with true types). Add it, or estimate from rate statistics.

## B. Dale-regularization variants
- **Soft vs hard Dale** (λ_Dale finite sweep) instead of the hard sign projection.
- **Iterative types**: alternate {estimate types ↔ sign-constrained solve} to self-correct.
- **Dale + mixture + balance** as a single in-solver-then-postproc pipeline (once the
  median bug is fixed).

## C. Parameters held fixed (never swept)
- `smooth_window_ms` (held at 0.5 ms) — interaction with lag untested.
- Larger `sim_time` / T — precision was still climbing at 50 k; does it plateau?
- Lag robustness — only mapped on ONE network; is the 1.5 ms optimum stable across seeds?
- Calcium `tau` — tested at 100 ms; the slower 400 ms regime (harder) untested.

## D. Method / model variants
- **Multi-lag VAR** (use all lags 1…L, not a single lag) — archive has a stub
  (`torch_linear_layer`).
- Alternative feed reconstructions / preprocessing variants (`compare_preprocessing.py`).
- Spectral analysis angle (`scripts/spectral_analysis.py`, currently unused).

## E. Generalization (the "second-level" checks the user set aside)
- Vary network config: N, density (epsilon), g, firing rates — does the pipeline hold?
- Larger networks (N=1000+) — scaling of the conclusions.

## F. Theory pass (user-led)
- Vet each method in `methods_overview.md` for theoretical eligibility (some may be
  inadmissible regardless of empirical score). The "mechanism · assumptions" section of
  that file is written for exactly this.
