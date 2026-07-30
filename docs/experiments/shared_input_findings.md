# Shared input & the excitatory ceiling — what works, what doesn't

Consolidated record of the R.7 / R.8 / Way-2 / Way-3 / timing investigation into
*why* excitatory connectivity recovery hits a ceiling, and which tools help.
Everything here is measured (single seed, N=1250 and N=12500, r4/low-rate
regimes) unless noted. Dates 2026-07-28 … 2026-07-30.

## The short version

- The excitatory ceiling is **shared-input confounding**: neurons driven by a
  common (often *unobserved*) source look connected. Inhibition is unaffected
  (strong, sign-separated → essentially solved everywhere).
- **Partial observation is the cause, in aggregate** (R.7, robust).
- **Directional symmetry** flags fakes and gives a *modest* real precision gain
  (R.8 phase 1) — the one filter that survives realistic (calcium) observation.
- **Timing** (multi-lag) separates real from fake *on spikes* but is **destroyed
  by the calcium blur** (R.8b + spikes-vs-calcium figure). Clear, demonstrated
  reason (see resolution numbers below).
- The confound is **diffuse and multi-lag** — it is *not* cleanly attributable to,
  or removable via, individual hidden drivers once you observe through calcium
  with a single-lag estimator (Way 2 and Way 3 both null/negative on real data).

## What WORKS

### R.7 — partial observation causes the fakes (aggregate, robust)
Observing a fraction of the network and inferring among the observed subset (a
re-solve of cached sub-block moments, samples-per-observed-neuron held constant):
as the observed fraction falls 100→10%, correlation 0.70→0.36, excitatory recall
0.27→0.11, **excitatory precision 0.38→0.18**, and even inhibition finally cracks
(I-recall 1.0→0.87 at 10%). Because data-per-neuron is matched, this degradation
is purely hidden-input confounding. This is the population-level proof.

### R.8 phase 1 — directional symmetry (the usable filter)
A true edge j→i is one-directional (A[i,j] large, A[j,i]≈0, asymmetry≈1); a
shared-input fake is symmetric (A[i,j]≈A[j,i], asymmetry≈0). False positives ARE
more symmetric than true edges, and filtering symmetric edges lifts excitatory
precision (0.56→0.83 at N=12500, 0.67→0.90 at N=1250) at a recall cost.
**Caveat:** symmetry conflates shared-input fakes with *reciprocal real edges*
(i→j and j→i both real, ~1% of pairs in Brunel). So it is a statistical operating
point, not a per-edge certainty. Without ground truth, report edges with an
asymmetry-based *confidence*, don't hard-delete; calibrate the expected error on a
matched simulation.

## What DOESN'T (and why — these are honest negative results)

### Timing / multi-lag diagnostic — works on spikes, killed by calcium
Compute the connectivity at many lags; a real edge should peak at the synaptic
delay, a shared-input fake instantaneously.
- **On raw spikes:** true edges show a sharp peak exactly at the 1.5 ms delay and
  ~0 at short lags; fakes are broad with a large instantaneous (<1 ms) shoulder.
  **Separable.**
- **Through the deconvolved calcium feed:** both collapse onto the *same* broad
  ~3 ms bump peaking at 1.5 ms. **Not separable.** Timing-only ≈ baseline;
  symmetry×timing ≈ symmetry alone (N=12500).

**Why — the resolution numbers:**

| quantity | value | vs 1.5 ms delay |
|---|---|---|
| synaptic delay | 1.5 ms | — |
| calcium decay τ | 100 ms | 67× |
| deconvolution smoothing window | 3.1 ms (half 1.55 ms) | ~2× |

After preprocessing the temporal resolution (~1.5–3 ms) is at/coarser than the
delay, so instantaneous (0 ms) and delayed (1.5 ms) are indistinguishable. The
calcium low-pass (τ=100 ms, cutoff ~1–2 Hz) *destroys* the ~700 Hz information of
a 1.5 ms delay before deconvolution; sharpening the deconvolution amplifies noise,
not signal, and hurts recovery. Real indicators (GCaMP τ ~ hundreds of ms to ~1 s)
are worse. **The synaptic timescale is below what calcium imaging can see** — a
property of the measurement, not of the method.

### Way 2 — fakes are NOT cleanly "hidden common drivers"
Split each false positive's common presynaptic drivers into observed vs hidden and
regress the fake magnitude on both (joint, so correlation can't hide the effect).
Result: fakes track **observed** common drivers *as much or more* than hidden
(b_observed 3.7e-4 vs b_hidden 7e-5 on the synthetic). Reason: single-lag OLS
conditions at one lag, but shared drive acts across the driver's *autocorrelation
timescale* (multiple lags), so **even observed shared input leaks into fakes.**
→ single-lag OLS is an *imperfect* deconfounder even for recorded neurons. So
"partial correlation removes observed shared input" is only approximately true.

### Way 3 — re-observing the hidden driver does NOT remove the fake (on calcium)
Observe 50%, find the fakes, add their hidden common drivers back and re-solve.
- **Synthetic VAR:** the specific drivers cut fakes to ~26% of baseline vs ~48%
  for the same number of random hidden neurons — a clear causal effect.
- **Real calcium (N=1250):** null — baseline 6.6e-3, +drivers 6.5e-3, +random
  6.4e-3. Adding the culprit did no more than a random neuron. The clean synthetic
  effect vanishes under calcium + single-lag inference — consistent with Way 2.

## Multi-lag: diagnostic vs estimator

- **Diagnostic** (tested): a post-hoc *label* on an existing estimate — inspect each
  edge's coupling-vs-lag shape. Needs to *read* the timing, which the calcium blur
  erases → fails on calcium.
- **Estimator** (UNTESTED): a joint multi-lag VAR that predicts each neuron from
  several past lags at once, changing the estimate to *produce* fewer fakes. It
  *uses* the lags mechanically (doesn't need clean timing), so it could deconfound
  where the diagnostic can't — this is what Way 2 motivates. **But** the ~3 ms
  resolution floor means lag-1/lag-2 of the deconvolved feed are largely redundant,
  so its benefit is likely capped. Not built. The real lever is a faster indicator
  (smaller τ), also untested.

## Thresholding vs shared-input analysis (conceptual)
Thresholding moves along a *fixed* ROC/PR curve — it trades false positives for
missed edges and **cannot fix confounding** (a strong fake and a strong real edge
sit at the same |weight|). Shared-input analysis (symmetry) adds a second axis and
can shift the *whole* frontier — the only thing that reduces confound-driven error,
modestly. They are complementary: use the confound-aware score, then threshold for
your precision/recall priorities. (Distribution-fit thresholding — Pareto/mixture —
would give a principled, density-agnostic cut and per-edge confidence, but inherits
the same confounding floor. Not pursued.)

## Bottom line for the write-up
Excitatory recovery is limited by shared-input confounding from partial
observation (R.7). Directionality (symmetry) buys a modest, honest gain. Timing is
sound in principle but below the temporal resolution calcium imaging affords. The
residual confound is diffuse and multi-lag — not attributable to or removable via
individual hidden drivers under calcium + single-lag inference. Report edges with
graded (asymmetry-based) confidence, and calibrate expected error on a matched
simulation; certainty about symmetric edges requires wider observation or
intervention, not post-processing.
