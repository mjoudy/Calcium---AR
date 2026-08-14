# Experiment Notebook

Plain-language record of what we ran and what we learned. This is **Layer 2** of
the experiment record — the *conclusions*. The numbers live in the ledger
(`results/ledger.csv`, one row per run; rebuild/query with `scripts/ledger.py`).

**Workflow:** Claude appends a dated entry after each experiment campaign; you
review and correct. Newest entries go at the top of the Log. The
*Findings at a glance* table below is kept current so a one-line question like
"what does lag do?" is answerable without reading the whole log.

---

## Findings at a glance

One line per parameter. Updated as evidence comes in. "—" = not yet tested.

| Parameter | Effect on recovery | Best setting found | Evidence (entry) |
|---|---|---|---|
| `lag_ms` | **Dominant knob.** Pearson peaks sharply at the **1.5 ms synaptic delay** (0.37); off-peak at 1.0 ms it is ≈ 0. Spikes give a knife-edge peak at exactly 1.5 ms; calcium smears it into a usable 1.4–1.6 ms window. | **1.5 ms (= 15 samples, = synaptic delay)** | 2026-06-18 |
| `smooth_window_ms` | not yet swept; held at 0.5 ms (5 samples), which worked fine | 0.5 ms (provisional) | 2026-06-18 |
| `lam` (Elastic-Net L1) | **Big win for detection, at a cost.** F1 0.30 → 0.62, precision 0.18 → 0.84 as L1 rises. But Pearson falls (0.35 → 0.24) and inhibition is zeroed first (E/I → 0). Useful range is *tiny* (1e-3…1e-2); ≥ 3e-2 zeros 92 %. | L1 ≈ 3e-3 (balanced) | 2026-06-18 |
| `lam_l2` (Elastic-Net L2) | barely matters here — L1 does the work; neurons not very collinear at this lag | small / either | 2026-06-18 |
| post-processing (rescaling) | Alone: fixes magnitude (oracle E/I 0.29 → 6.0, Pearson → 0.52) but hurts detection. **Regularize→rescale breaks the trade-off**: F1 0.30→0.51, Pearson 0.35→0.55, E/I→5.2 together. Needs L1≈3e-3 (keep inhibition) + true types (oracle). | regularize(L1≈3e-3)→rescale | 2026-06-18 |
| Dale's-law post-processing | **Best unsupervised lever for detection+type.** EN+Dale+mixture: F1 0.30→0.49, type_acc 0.83→0.92, Spearman→0.50, perfect Dale — no ground truth. Does NOT fix magnitude. Full table: [methods_overview.md](methods_overview.md) | EN→Dale→mixture | 2026-06-19 |
| balance-rescale (magnitude) | **Unsupervised magnitude works.** Estimate g from network balance → rescale; matches oracle Pearson (0.553) with no ground truth, F1→0.48. E/I ratio only partial (1.63 vs 5, limited by type-guess). Doesn't compose with Dale yet (median/zeros bug). | EN→balance-rescale | 2026-06-19 |
| `sim_time` (T) | **Split result.** Precision & Pearson still *rising* at 50 k (variance-limited → regularization can help). But the E/I ratio *worsens* toward ~0.3 with more data (bias-limited → regularization can't fix it; needs post-processing). | more data still helps precision | 2026-06-18 |
| `sigma_extra` (noise) | **small effect at the optimum** — recording noise costs only ~0.02 Pearson (0.370 → 0.353) | minor | 2026-06-18 |
| input stage (spikes vs calcium) | **Calcium is not the bottleneck**, but preprocessing is still needed. Preprocessed feed recovers ~95 % of spike-level Pearson. RAW calcium (no prep) gives a *higher but fake* Pearson (0.48) with worse AUC/type/precision — prep removes that confound. | preprocess (don't skip) | 2026-06-18 |
| camera frame rate (R.2, dye τ vs camera dt, swept separately) | **Deconvolution trades noise for rank.** At a fast camera (dt≲2ms) deconvolved AUC > raw, but deconvolved *correlation* < raw — differentiation amplifies noise fastest exactly when frames are close together. Crosses over (deconv wins both) around dt≈5–10ms. Every other figure in the project (R.1/4/5/7/8) implicitly assumes an infinitely-fast camera (dt=0.1ms, no downsampling) — R.2 is the only place a realistic frame rate is modeled at all. | slow enough camera (≳5–10ms) for deconvolution to pay off on magnitude too | 2026-08-06 |
| network size N, fixed connection PROBABILITY (R.4, ε=0.1 always, C_E grows with N) | **Correlation converges with N; excitatory recall/precision do NOT.** At matched absolute T, correlation converges to ~0.8–0.85 by 5M ms. Excitatory recall/precision stay separated by N (recall: N=1250 ~0.77 vs N=12500 ~0.46 at the same T) — structural under THIS scaling convention, driven by in-degree C_E=ε·N growing with N. | more data helps correlation broadly; does not fix excitatory recall/precision under fixed probability | 2026-08-07 |
| network size N, fixed IN-DEGREE (R.4b, C_E=100 always, ε shrinks with N — more biologically motivated) | **The N-gap above is a scaling-convention artifact, not unavoidable — and it fully REVERSES.** Full grid, 3 sizes confirmed: recall/precision 0.86/0.85 (N=2500) → 0.94/0.93 (N=5000) → 0.99/0.98 (N=12500). N=12500-ci beats even N=1250's fixed-probability ceiling. Bigger networks go from structurally worse to structurally better once in-degree, not probability, is held fixed. Correlation converges either way (data-amount, not confound). Caught+fixed a threshold-calibration bug along the way — see 2026-08-08/2026-08-10 log entries. | fixed in-degree, not fixed probability, if biological realism is the goal | 2026-08-10 |

---

## Open questions

- ~~Is the smallness/overlap of `Â` caused by the calcium pipeline or by the
  regression itself?~~ **Answered (2026-06-18): the regression itself.** Even on
  perfect spikes the ceiling is Pearson ≈ 0.37 / AUC ≈ 0.90 — calcium adds almost
  no further damage.
- ~~Is the E/I magnitude ratio (`g=5`) preserved in `Â`?~~ **Answered: no — it is
  compressed/inverted** (≈ 0.3 at the optimal lag; inhibitory ends up *weaker*
  than excitatory in the estimate). Needs its own handling (per-class).
- ~~Does the lag optimum sit near the 1.5 ms synaptic delay?~~ **Answered: yes,
  exactly.** Optimum = 1.5 ms.
- NEW: the ceiling is only Pearson ≈ 0.37 (though AUC ≈ 0.90). Detection is good;
  magnitude correlation is the weak part. This is the target for
  regularization / post-processing — and we now know it is a *regression* problem,
  not a calcium problem.
- NEW: does `smooth_window_ms` interact with the lag window? (held fixed so far)
- NEW (current bottleneck): **identifying inhibitory neurons unsupervised.** Naive
  sum/majority ≈ chance; strongest-entry rule reaches 0.70. A good unsupervised type
  detector would unlock the fully-unsupervised regularize→rescale pipeline.
- NEW: can we estimate g (the E/I weight ratio) from the data (e.g. network balance) so
  the unsupervised rescale can target the true 5, not just equalise to 1?
- NEW (R.2, 2026-08-06): does widening the Savitzky-Golay smoothing window (or lowering
  polyorder) at fast camera rates fix the deconvolved-correlation deficit? Plausible
  (derivative noise gain shrinks with window length; τ=100ms leaves lots of room before
  a wider window would blur real dynamics) but not yet tested — needs a small SMOOTH_MS/
  polyorder sub-sweep at dt≲2ms.

---

## Log (newest first)

<!--
Entry template — copy for each new campaign:

### YYYY-MM-DD — <short title>
**Question:** what we set out to learn.
**Setup:** what changed vs the previous baseline (config deltas only), data size,
which params were held fixed.
**Result:** key numbers (metric = value), and the run_dir(s) / plots.
**Conclusion:** the plain-language takeaway — the sentence you'll want in a week.
**Next:** follow-up ideas.
-->

### 2026-08-11 — Checked the professor's own papers: they use fixed probability, not fixed in-degree
**Question:** given fixed in-degree is the more biologically-motivated scaling
convention (see 2026-08-10 discussion), does the professor's own published
work (Pernice/Rotter) support fixed probability or fixed in-degree when
network size varies?

**Checked directly** (4 PDFs: Pernice 2011 PLoS CB, Pernice 2012 PhysRevE,
Pernice & Rotter 2013 J Stat Mech, Schiefer/Pernice/Rotter 2018 PLoS CB) — full
writeup in `docs/theory/shared_input_theoretical_grounding.md` Section 6.
**All four use fixed connection probability whenever they touch network size
at all** — most direct precedent: Pernice 2012 Fig. 4/5 explicitly scales N
"assuming a uniform connection probability p." This corrects an earlier
in-chat claim (that balanced-network theory typically fixes in-degree when
scaling N) — that classical argument (van Vreeswijk & Sompolinsky's dilute
limit) is real but is a different tradition than what these specific,
directly-relevant papers do.

**Best guess why (unconfirmed, worth asking Prof. Rotter directly):**
mathematical convenience — fixed-p (Erdős–Rényi) graphs are the ensemble
random matrix theory and the mean-field self-consistency equations in these
papers are built for; none of the four papers actually pose R.4's specific
question ("grow the same kind of network, keep the dynamics matched, see what
happens to inference").

**Conclusion:** citing the professor's own papers as precedent for fixed
probability is accurate, but it's precedent for a different question than R.4
asks. Recommend presenting both conventions in the report rather than picking
one as "correct," and asking Prof. Rotter directly why fixed probability
became the group's standard.

---

### 2026-08-08 — Fixed-in-degree scaling explains most of the N-gap (confirmed, all 3 sizes)
**Status: CONFIRMED at N=2500/5000/12500-ci. Supersedes the same-day PRELIMINARY
entry below it, which had a real bug in the precision comparison — corrected here.**

`figures/fig_R4_CI` (5 seeds each N=2500/5000, 3 seeds N=12500, fixed-in-degree
C_E=100 vs. the original fixed-probability ladder), density-matched scoring
(see bug below):

| size | AUC_E | E_recall | E_precision |
|---|---|---|---|
| N=2500-ci | 0.996 | 0.86 | 0.85 |
| N=5000-ci | 1.000 | 0.94 | 0.93 |
| N=12500-ci | 1.000 | 0.99 | 0.98 |

**Both recall AND precision improve with N under fixed in-degree** — the
original R.4 "excitatory recall gap widens with N" finding was substantially a
**fixed-probability scaling-convention artifact**, not an unavoidable property
of bigger networks. Fixed-probability N=2500's best precision (0.70) is beaten
by fixed-in-degree N=2500 (0.85) once scored correctly.

**Bug found and fixed along the way, worth remembering:** `analyze_run.py`
thresholds at a FIXED 10%-density cutoff by default. That's correctly
calibrated for the fixed-probability ladder (true density ~10% everywhere),
but wrong for fixed-in-degree, where true density shrinks with N (ε=0.05 at
N=2500, 0.025 at N=5000, 0.01 at N=12500) — forcing a 10% cutoff on a ~5%-dense
network guarantees ~half the "positive" calls are padding, independent of
confound severity. First-pass numbers (precision plateauing ~0.48 for
N=2500-ci) were entirely this artifact — AUC_E (threshold-free) already showed
fixed in-degree was *better* (0.996 vs 0.965), which is what caught it. Fix:
`analyze_run.py --density <true epsilon for that network>`, not the default.
**Any future scoring of the ci ladder (or any non-10%-density network) must
pass the matching `--density`, not rely on the default.**

**Report implication:** the N-scaling story needs a real split now — under the
project's original (fixed-probability) scaling convention, bigger networks are
structurally worse at excitatory recovery (R.4's original finding, still true
*for that convention*). Under the more biologically-motivated fixed-in-degree
convention, that gap is not structural at all — it *improves* with N. State
both, and be explicit about which scaling convention is being discussed.

**UPDATE (2026-08-10) — full grid rescored, all 6 checkpoints x 3 sizes, result
holds and is even stronger than the single-checkpoint check suggested:**
`figures/fig_R4_CI` now shows the complete curves, correctly thresholded
throughout. **The order doesn't just close, it fully reverses**: by T=5M ms,
N=12500-ci reaches the HIGHEST recall (0.99) and precision (0.98) of every
curve in the figure — including beating N=1250's fixed-probability ceiling
(0.77/0.75). Under fixed in-degree, bigger networks go from "structurally
worse" to "structurally better." Correlation converges normally regardless of
which scaling convention is used (data-amount effect, not confound-severity).

**Also resolved the same day:** pushed N=1250 to 20M ms and N=12500 to 10M ms
(fixed-probability ladder) to test whether the original R.4 gap was a true
plateau or just not-yet-converged (see the 2026-08-07 entry's open question).
Answer: **N=1250 has genuinely plateaued** — flat at recall≈0.77 from 5M to
20M ms (4x more data, zero improvement). N=12500 is still slowly climbing at
10M ms (0.47→0.52) but decelerating, and still far below N=1250's ceiling —
consistent with a real, data-independent ceiling under fixed probability, not
merely insufficient data (though N=12500 alone hasn't fully flattened yet).
Interestingly correlation behaves oppositely: N=1250 plateaus (~0.77) but
N=12500 keeps climbing PAST it (~0.86 by 10M ms) — bigger N eventually wins on
correlation with enough data, but not on excitatory recall/precision.

**Next:** none outstanding for R.4/R.4b. Both the plateau question and the
scaling-convention question are now answered with real data. Good place to
write this section of the report.

---

### 2026-08-08 — PRELIMINARY (SUPERSEDED, see corrected entry above): fixed-in-degree scaling may explain most of the N-gap
**Superseded 2026-08-08 — the precision numbers here were a threshold-mismatch
bug, not a real finding. Kept for the record of how it was caught, not as a
citable result.**

`figures/fig_R4_CI` (5 seeds, N=2500 fixed-in-degree C_E=100 vs. the original
fixed-probability N=2500): **excitatory recall reaches ~0.99 by T=5M ms under
fixed in-degree — not just closing the gap with fixed-probability N=2500
(0.70), but exceeding even N=1250's fixed-probability recall (0.77).**
Excitatory precision is more nuanced: fixed-in-degree N=2500 plateaus ~0.48
while fixed-probability N=2500 keeps climbing to ~0.70 — the benefit isn't
symmetric across both metrics. Correlation is roughly unchanged between the
two scaling conventions (consistent with it being variance- not bias-limited).

If this holds at N=5000/12500 too: the R.4 "excitatory recall gap widens with
N" finding was **substantially a scaling-convention artifact** (fixed
connection probability, not fixed in-degree) rather than an unavoidable
property of larger networks — a big revision to how the report should frame
the N-scaling result. See [[shared_input_findings.md]] for the mechanism this
tests (in-degree C_E=eps*N drives the confound).

**Next:** wait for N=5000-ci and N=12500-ci (running), confirm the pattern
holds at those sizes before updating the "Findings at a glance" table or
drawing report conclusions from this.

---

### 2026-08-07 — R.4 extended to a full 6-point T grid; excitatory recall doesn't converge with N

**Question:** Professor's feedback on fig_R4: (1) drop the right "vs T/N, curves
collapse" column — it was a somewhat trivial consequence of the sweep design
(every N tested at matched T/N=800/1600/4000, so of course those points line
up); (2) each N's line only covered a narrow, disjoint slice of the "vs
recording length" x-axis (e.g. N=1250: 100k–500k ms; N=12500: 1000k–5000k ms) —
add more points so all 4 lines stretch across the full axis.

**Setup:** N=1250/2500/5000/12500, matched AI regime (~14Hz, CV~1.0, g=8,
V_reset=10, J·√C_E=4.743, per-N eta). Extended every size to the shared grid
T=100k/200k/500k/1M/2M/5M ms. N=12500 was cheap to extend (cached ground truth,
same max_T=5M already simulated — just added 3 more checkpoints). N=1250/2500/
5000 each needed a fresh full resimulation up to the new 5M ms max (longer than
their old max, so the ground-truth cache key changed). 4 separate SLURM jobs
(`slurm/run_r4_n12500.slurm` + 3 new `run_r4_n{1250,2500,5000}_extend.slurm`),
all completed same night (16min–2h each, well inside budget). Plot:
`scripts/fig_r4_plot.py` → `figures/fig_R4` (now single column, no collapse
panel).

**Result:**
1. **Correlation converges with N.** At matched absolute T (not just matched
   T/N), all 4 sizes climb toward ~0.8–0.85 by T=5M ms — the size gap seen at
   short T mostly closes with enough data.
2. **Excitatory recall does NOT converge.** At the same T=5M ms: N=1250≈0.77,
   N=2500≈0.70, N=5000≈0.60, N=12500≈0.46 — a clean, monotonic, persistent gap
   by size that more data does not close.
3. **Excitatory precision shows the same persistent gap** (added as a 3rd row,
   free — already in the cached `metrics.csv`, no new runs). At T=5M ms:
   N=1250≈0.75 down to N=12500≈0.55, same clean monotonic-by-N separation as
   recall. So it's not just that big-N estimators miss real excitatory edges
   (recall) — their positive calls are also proportionally less trustworthy
   (precision). Both directions of the confound persist with N regardless of
   data amount.

**Conclusion:** this is the same shared-input-confounding mechanism already
established in R.7/R.8 ([[shared_input_findings.md]]), now shown to be
structural rather than a data-amount effect: fixed connection probability
(ε=0.1) means in-degree C_E=ε·N_E scales with N, so bigger networks have
proportionally more shared common input per neuron regardless of recording
length. Correlation (dominated by inhibition + overall fit) isn't very
sensitive to this and converges; excitatory recall AND precision are both
directly capped by it and don't. Report implication: don't present "more data
fixes everything" — split the claim by metric.

**Next:** none planned; this closes out the "does more data fix the N-scaling
gap" question definitively (no, not for excitatory recall). Could check if the
excitatory-recall gap scales cleanly with C_E specifically (linear? sqrt?) if
useful for the report.

**Addendum — bias vs. variance framing (2026-08-07):** why precision and
recall moved *together* (both worse at large N) instead of trading off, and
why correlation converges while they don't, is cleanest explained as a
bias/variance split, worth keeping for the report write-up:
- Precision/recall trade off when sliding the *threshold* on one fixed
  classifier. Here we're comparing *different* classifiers (by N) at the same
  fixed threshold — when the underlying separation itself degrades, precision
  and recall fall together (more false positives AND more false negatives at
  once), they don't trade off.
- **Correlation is a variance problem** — more recording lets OLS converge
  more precisely everywhere, so it shrinks with data and converges across N.
- **Excitatory precision/recall are a bias problem** — shared-input
  confounding is a real feature of the true data (two neurons sharing a driver
  genuinely do correlate, synapse or not), not sampling noise. More data
  measures that fake correlation more precisely; it doesn't remove it. Bias
  doesn't shrink with T, and here it grows with N specifically (in-degree
  C_E=ε·N_E), so the gap is structural, not fixable by recording longer.
- One-liner: *"More data fixes noise, not confounding."*

**Addendum — matched-T/N confirms the split directly (2026-08-07):** built an
appendix figure (`figures/fig_R4_TN`, `scripts/fig_r4_plot.py --out-tn`, zero
new compute — same cached CSVs, just plotted vs T/N instead of raw T) to test
the prediction. Confirmed exactly: **correlation collapses** onto one curve
across all 4 N when plotted vs samples-per-neuron (T/N) — it really is a
variance/data-per-parameter effect. **Excitatory recall and precision do NOT
collapse** — same clean N-ordered separation persists even at matched T/N
(e.g. at T/N≈4000: recall 0.77 (N=1250) vs 0.46 (N=12500), same gap as vs raw
T). Clean, direct, one-figure proof that T/N is the right variable for the
correlation story and the wrong variable for the excitatory story — that one
is driven by raw N (in-degree), not by any per-neuron data ratio.

---

### 2026-08-06 — R.2 restructured: dye τ and camera rate swept one at a time (not as a ratio)

**Question:** R.2 originally collapsed two different physical knobs — dye decay
time constant τ (a property of the calcium indicator) and camera frame interval
dt (a property of the recording setup) — onto one shared `dt/τ` "blur" x-axis, to
test whether only the ratio matters. Professor's feedback: this makes it
impossible to tell which physical change is driving a given point; vary one at a
time instead. Separately: why is R.2's correlation so much lower than the
equivalent figure in the 2021 thesis presentation (page 15, "Effect of time
constant")?

**Setup:** N=1250, `n1250ai` regime (g=8, η=1, J_ex=0.8, ~8Hz AI), T=100k ms fixed
recording length (deliberately short — R.2 isolates the *observation* effect from
the *amount of data* effect). Two sweeps run separately, each holding the other
knob fixed:
  - **camera-rate sweep**: dye τ fixed at 100ms, camera frame interval swept
    0.1–1000ms.
  - **dye-τ sweep**: camera fixed at a realistic 33ms (~30Hz, not an idealized
    infinite-speed camera), τ swept 0.5–1600ms.
Both compare `raw` vs `deconvolved` (Savitzky-Golay smooth+derivative,
`SMOOTH_MS=3.1`, `polyorder=3` fixed, window widens automatically once camera dt
pushes the sample-count floor). Scripts: `scripts/fig_r2_compute.py` (compute,
ran on cluster) → `scripts/fig_r2_plot.py` (plot, local). Output:
`figures/fig_R2` (primary, one-variable-at-a-time) + `figures/fig_R2_ratio`
(secondary collapse view, kept for reference only).

**Result:**
1. **Camera rate matters more than τ in this regime.** AUC/corr both fall
   steadily as the camera slows past ~5ms and bottom out near chance by ~100ms+
   (both raw and deconvolved). The τ sweep (camera fixed) is much flatter —
   AUC/corr barely move across 0.5–1600ms once the camera itself is the
   bottleneck.
2. **Deconvolution helps AUC but hurts correlation at fast camera rates**, and
   flips to helping both past dt≈5–10ms. Numbers (dt=0.1→10ms): deconv AUC
   0.81→0.62 vs raw AUC 0.70→0.61 (deconv wins throughout); deconv corr
   0.43→0.43 (dips to 0.35 at dt=1) vs raw corr 0.55→0.43 (raw wins until the
   crossover). Mechanism: deconvolution = smoothing + numerical derivative;
   differentiation amplifies noise, worst when frames are close together
   because the real signal barely changes between adjacent samples while sensor
   noise doesn't shrink. AUC (rank-based) survives that noise much better than
   Pearson correlation does.
3. **The R.2 vs 2021-thesis correlation gap is NOT about tau/dt** — it's an
   apples-to-oranges setup difference. Found the actual 2021 config
   (`_arxiv/scripts-hpc/dask_pipeline/network_config.json`): g=6/η=2, **J=8.0**
   (current: J_ex=0.8, 10× weaker), **sim_length=1,000,000ms** (current: 100k ms,
   10× less data). Proof it's not the calcium manipulation: the spikes-only
   reference curve (no calcium involved at all) also collapsed, from
   AUC≈0.99/corr≈0.85–0.9 (2021) to AUC≈0.78/corr≈0.54 (now) — same gap, zero
   calcium effect, so the cause is the weaker synapses + 10× less data, not
   anything R.2 tests.
4. **R.2 is the only figure in the whole project that models a realistic camera
   at all.** Checked `calcium_ar/simulation/calcium_signal.py`, `wrapup_run.py`,
   all preprocessing code — zero mentions of downsampling/frame rate anywhere
   else. Every other figure (R.1/4/5/7/8, wrapup) implicitly assumes an
   infinitely-fast camera (dt=0.1ms, no downsampling step at all). Their
   headline recovery numbers are therefore best-case relative to a real ~30Hz
   camera.

**Conclusion:** the one-variable-at-a-time restructuring plus the extended
ranges (past the old endpoints, so curves visibly reach their floor) shows
camera rate is the dominant limiting factor in this regime, not dye kinetics.
Deconvolution is a rank-order tool more than a magnitude-recovery tool at fast
frame rates — worth stating carefully in the report rather than claiming
deconvolution is unconditionally better. The low absolute correlation numbers
vs. the 2021 thesis are a data-amount/coupling-strength artifact, not a finding
about calcium observation — don't compare the two directly without matching
setups.

**Next:** (a) small SMOOTH_MS/polyorder sub-sweep to test whether a wider
smoothing window recovers deconvolved correlation at fast camera rates
(hypothesis: yes, since τ=100ms leaves lots of room before a wider window would
blur real dynamics — not yet tested); (b) optionally rerun R.2 with the 2021
config (stronger synapses, longer recording) for a true like-for-like
comparison, if the report needs one.

**Addendum (same day):** trimmed the primary `figures/fig_R2` down to 2 columns
(camera rate | binned-spikes reference) — the dye-τ column was flat/uninformative
next to the camera panel in this regime and crowded the main read. The τ panel
(finding #2 above still holds) now lives in its own appendix figure,
`figures/fig_R2_tau`, so nothing is lost — just demoted out of the headline
figure. `figures/fig_R2_ratio` (the dt/τ collapse view) unchanged.

---

### 2026-06-22 — Dale-regularization candidates: does anything beat hard in-solver Dale?
**Question:** Can soft Dale, iterative type-refinement, EM-soft, or a no-guess
min()-purity penalty beat the current champion C1 (hard in-solver Dale, strongest-entry
types)?

**Setup:** N=100 feed, lag 1.5 ms, EN base (L1=3e-3, L2=1e-3). All reuse the FISTA core.
Script: `scripts/dale_candidates_test.py` → `results/dale_candidates_test/ledger.csv`.

**Result (detection F1 / type_acc / dale / spearman; magnitude unchanged, pearson ≈ 0.34):**
| method | f1 | type_acc | dale | spearman |
|---|---|---|---|---|
| C1 hard (champion) | 0.515 | 0.920 | 1.000 | 0.470 |
| C1 hard (TRUE types, ceiling) | 0.525 | 1.000 | 1.000 | 0.486 |
| C2 soft (λ_D→0.1) | 0.515 | 0.920 | 1.000 | 0.470 |
| C3 iterative hard | 0.515 | 0.920 | 1.000 | 0.470 |
| C4 EM soft | 0.499 | 0.890 | 0.917 | 0.464 |
| C5 min() no-guess (best, λ_D=0.01) | 0.501 | 0.870 | 0.953 | 0.459 |

**Conclusion — nothing beats C1:**
1. **Soft Dale (C2) converges to hard as λ_D grows** — hard is the correct limit; no softer
   sweet spot exists.
2. **Iterative refinement (C3) = C1 exactly** — the strongest-entry types are already a
   fixed point, so iteration does NOT break the inhibitory-ID ceiling.
3. **EM-soft (C4) and no-guess min() (C5) are slightly worse**; C5's E/I collapses to 0 at
   λ_D ≥ 0.03 (the predicted over-purify/global-shrinkage failure of the no-guess route).
4. The type ceiling (0.92 vs true 1.00) is **intrinsic to the strongest-entry signal** —
   only TRUE types lift it, and even then F1 only 0.515→0.525. C1 is at the ceiling.

**Decision:** Freeze **C1 (hard in-solver Dale, strongest-entry types)** as the
regularization step. Proceed with the balance rescale + HPC scale-up.

**Next:** none for Dale. The residual inhibitory-ID gap needs external information
(cell-type labels), not better optimization.

---

### 2026-06-19 — FULL unsupervised pipeline (composition fixed)
**Question:** Make detection (daleReg) and magnitude (balance) compose into one
unsupervised pipeline — fix the median/zero bug.

**Fix:** `rescale_balance_nz` estimates per-group magnitude from NON-ZERO entries only,
so it survives Dale's zeros. Script: `scripts/methods_overview.py`.

**Result — one pipeline, all three directions:**
| method | spearman | pearson | auc | f1 | type_acc | ei | dale |
|---|---|---|---|---|---|---|---|
| OLS | 0.42 | 0.35 | 0.87 | 0.30 | 0.83 | 0.29 | 0.54 |
| **EN_daleReg+balance_nz** | 0.47 | **0.585** | 0.86 | 0.49 | 0.92 | 0.92 | 1.00 |
| EN+dale+mixture+balance_nz | **0.50** | 0.54 | 0.86 | 0.49 | 0.92 | 0.61 | 1.00 |
| EN+oracle (GT ceiling) | 0.39 | 0.553 | 0.81 | 0.51 | 0.84 | 5.20 | 0.69 |

**Conclusion:**
1. **Composition solved.** A single unsupervised pipeline now gives detection (F1
   0.30→0.49), type (0.83→0.92, perfect Dale), AND magnitude (Pearson 0.35→**0.585**).
2. **Beats the oracle on Pearson** (0.585 vs 0.553): cleaning wrong-sign noise first
   (daleReg) lets the balance rescale overshoot raw per-class rescaling.
3. E/I *ratio* still partial (0.92, not 5) — capped by 70% inhibitory type-ID.
4. Two deployable pipelines: `daleReg+balance_nz` (best magnitude) vs
   `dale+mixture+balance_nz` (best rank).

**Recommended deployable pipeline:** preprocess → lag 1.5 ms → EN(L1≈3e-3) → Dale-reg →
balance_nz rescale. Detection + type + magnitude, fully unsupervised, near/above the
ground-truth ceiling.

---

### 2026-06-19 — Dale as regularization (in-solver vs post-hoc)
**Question:** Does enforcing single-sign columns DURING the fit (sign-constrained
Elastic Net) beat the post-hoc Dale cleanup?

**Theory:** minimise EN cost subject to sign(A[i,j]) = t_j per column; solved by adding
a per-column sign-projection to the FISTA prox: A_ij ← t_j·max(0, t_j·soft(A_ij)).
Types t_j from initial EN (strongest-entry). Hard constraint = λ_Dale→∞ limit of the
penalty Σ[−t_j·A_ij]₊. Script: `scripts/dale_reg_test.py`; also in the master overview.

**Result:**
| method | spearman | auc | f1 | precision | type_acc |
|---|---|---|---|---|---|
| EN+dale (post-hoc) | 0.461 | 0.857 | 0.494 | 0.370 | 0.920 |
| **EN_daleReg (in-solver, unsup)** | 0.470 | 0.861 | **0.515** | 0.394 | 0.920 |
| EN_daleReg (true types) | 0.486 | 0.878 | 0.525 | 0.399 | 1.000 |

**Conclusion:**
1. **In-solver Dale beats post-hoc** on every detection metric (F1 0.494→0.515, precision
   0.370→0.394, AUC, Spearman) — confirms the theory: constraining during the fit lets
   correct-sign entries re-grow to explain variance, instead of deleting it.
2. Best unsupervised detection method so far. Doesn't touch magnitude (as expected).
3. Combination bug persists: `daleReg+balance` = no magnitude change (sparse columns →
   median-based rescale breaks). Same median/zero issue.

**Next:** fix the median/zero composition so detection (daleReg) + magnitude (balance)
combine; iterative type refinement; soft-vs-hard Dale (λ_Dale sweep).

---

### 2026-06-19 — Balance-based magnitude rescale (unsupervised) — updates the magnitude verdict
**Question:** Can g estimated from network balance drive an unsupervised magnitude
rescale, instead of the oracle?

**Balance check:** g ≈ (N_E·νE)/(N_I·νI). With true types → g≈4.4; with inferred
(strongest-entry) types → g≈6.3; true g = 5. Ballpark (low bias from dropping external
drive), good enough to use.

**Result (`EN+rescale_balance`, unsupervised):**
| | spearman | pearson | f1 | ei |
|---|---|---|---|---|
| EN(3e-3) | 0.42 | 0.34 | 0.42 | 0.21 |
| **EN+rescale_balance** | 0.40 | **0.553** | 0.48 | 1.63 |
| EN+oracle (GT) | 0.39 | **0.553** | 0.51 | 5.20 |

**Conclusion — corrects the earlier "magnitude is GT-dependent" verdict:**
1. Balance-based rescale matches the **oracle's Pearson (0.553) with NO ground truth**,
   and improves F1 (0.30→0.48). Magnitude agreement IS recoverable unsupervised.
2. But the E/I *ratio* only reaches 1.63 (not 5) — the ~4 missed inhibitory neurons
   (type-guess 70%) don't get boosted. So weight *correlation* is recovered; exact
   *ratio* is still limited by type-inference quality.
3. **Combination bug to fix:** `dale→balance` does nothing (Dale's zeros break the
   median-based rescale). Fix: rescale on non-zeros, or rescale before Dale.

**Next:** better type inference would lift both ei and the dale/balance combo; or fix
the median issue so detection (dale+mixture) and magnitude (balance) compose cleanly.

---

### 2026-06-19 — Master methods overview + Dale/mixture post-processing
**Question:** Put every method in one table (detection/type/magnitude + unsupervised
quality) to select a final pipeline; and test Dale + 3-class mixture as unsupervised
post-processing. Finish the bottleneck (unsupervised strongest-entry rescale).

**Setup:** All methods at lag 1.5 ms on the real feed. Script:
`scripts/methods_overview.py` → table in `docs/experiments/methods_overview.md`.

**Result (unsupervised unless noted GT):**
| method | spearman | auc | f1 | precision | type_acc | ei | dale |
|---|---|---|---|---|---|---|---|
| OLS | 0.42 | 0.87 | 0.30 | 0.18 | 0.83 | 0.29 | 0.54 |
| EN(3e-3) | 0.42 | 0.85 | 0.42 | 0.29 | 0.82 | 0.21 | 0.69 |
| **EN+dale+mixture** | **0.50** | 0.86 | **0.49** | 0.37 | **0.92** | 0.10 | **1.00** |
| EN+rescale_strongest | 0.42 | 0.85 | 0.40 | 0.27 | 0.81 | 0.35 | 0.69 |
| EN+oracle (GT) | 0.39 | 0.81 | 0.51 | 0.39 | 0.84 | 5.20 | 0.69 |

**Conclusion:**
1. **Detection + type SOLVED unsupervised:** `EN + Dale-cleanup + mixture` is the best of
   all (incl. oracle on Spearman/type): F1 0.30→0.49, type 0.83→0.92, Spearman→0.50,
   perfect Dale. Dale's law is the strongest unsupervised lever.
2. **Magnitude still GT-dependent:** unsupervised strongest-entry rescale only lifts E/I
   0.21→0.35; only the oracle reaches ~5. Matches the bias-limit finding — magnitude
   needs types we can't get reliably.
3. **Tunable on real data:** the ground-truth-FREE scores (daleianity, overlap) pick the
   same winner → method-selection works without ground truth. (Caveat: `overlap` is in
   raw weight units → meaningless for rescaled matrices; compare within non-rescaled.)

**Recommended final pipeline (deployable):** preprocess → lag 1.5 ms → EN(L1≈3e-3) →
Dale-cleanup → mixture-threshold. Magnitude/E-I left as a bonus needing future work.

**Next:** improve unsupervised magnitude (estimate g from balance?) OR move to the
network-config direction the user set aside.

---

### 2026-06-18 — Unsupervised rescale: the type-detector is the bottleneck
**Question:** Does regularization make the unsupervised type-guess reliable enough to
replace the oracle?

**Setup:** EN(L1=3e-3) matrix, infer types from the estimate (3 rules), rescale,
compare to oracle. Script: `scripts/unsup_rescale_test.py` (+ inline rule comparison).

**Result — type-inference accuracy (identifying INHIBITORY neurons):**
| rule | overall | inhibitory-only |
|---|---|---|
| column-sum | 0.87 | 0.45 (chance) |
| majority-of-survivors | 0.81 | 0.15 |
| **strongest-entry** | **0.92** | **0.70** |

- Regularization did NOT rescue the column-sum guess (inhibitory 0.50 → 0.45).
- Unsupervised `typeeq` (sum-based) = no change vs reg-only.
- For an inhibitory neuron only **42%** of surviving entries are negative — most are
  positive noise — so sum/majority fail. But its single **strongest** entry is negative
  **70%** of the time (true inhibition is 5× stronger → the dominant connection reveals
  the type).

**Conclusion:**
1. The blocker for a fully unsupervised pipeline is **identifying inhibitory neurons**;
   the oracle's power was entirely in knowing types.
2. The naive sum/majority detector is ~chance for inhibition; the **strongest-entry**
   rule recovers it to 0.70 (overall 0.92) — a promising deployable detector (untested
   end-to-end).
3. Refines earlier "good at regions ~85%": that was carried by EXCITATORY neurons;
   inhibitory identification has been the weak spot at every stage.

**Next:** full unsupervised pipeline using the strongest-entry type rule for rescaling;
see how close it gets to the oracle. Also: an unsupervised rescale can only equalise
(E/I → 1), not hit the true 5, without external g — may need to estimate g from balance.

---

### 2026-06-18 — Regularize-then-rescale (the synthesis)
**Question:** Does regularizing FIRST (prune noise) then rescaling (lift survivors)
break the detection-vs-magnitude trade-off?

**Setup:** OLS feed at lag 1.5 ms → FISTA Elastic Net (L1 ∈ {1e-3,3e-3,1e-2}) →
oracle / colnorm rescale. Script: `scripts/combine_test.py`.

**Result — the combination wins on all three at once:**
| method | F1 (detect) | Pearson (mag) | E/I (truth 5) | spearman |
|---|---|---|---|---|
| OLS | 0.30 | 0.35 | 0.29 | 0.42 |
| regularize only (EN3e-3) | 0.42 | 0.34 | 0.21 | 0.42 |
| rescale only (OLS+oracle) | 0.39 | 0.52 | 6.0 | 0.37 |
| **EN3e-3 + oracle** | **0.51** | **0.55** | **5.2** | 0.39 |

**Conclusion:**
1. **The trade-off breaks.** Regularize→rescale is the only path good on detection AND
   magnitude AND E/I simultaneously (F1 0.30→0.51, Pearson 0.35→0.55, E/I→5.2). Prune
   noise first, then nothing bad gets amplified.
2. **L1 ordering matters:** L1=1e-2 zeroes inhibition entirely → rescale can't recover it
   (E/I 0.08). Sweet spot L1≈3e-3 (prune noise, keep inhibition alive).
3. **Caveats:** rescaling used TRUE types (oracle = ceiling); unsupervised colnorm only
   partly works and hurts F1. Spearman/AUC dip slightly (rescaling disturbs ranking).

**Next / open problem:** a deployable UNSUPERVISED type-and-scale estimator. Idea:
regularization cleans the estimate (EN raises spearman 0.42→0.48), so column-sign type
inference may now be reliable enough to replace the oracle — test typeeq ON the
regularized matrix.

---

### 2026-06-18 — Post-processing (rescaling) test
**Question:** Can rescaling the OLS estimate undo the inhibition bias (the part
regularization can't)?

**Setup:** OLS at lag 1.5 ms, then 4 post-processing methods. Headline = Spearman.
Script: `scripts/postprocess_test.py`.

**Result:**
| method | spearman | pearson | auc | f1 | E/I |
|---|---|---|---|---|---|
| baseline | 0.417 | 0.353 | 0.870 | 0.300 | 0.29 |
| colnorm (unsup, per-neuron) | 0.411 | 0.446 | 0.881 | 0.215 | 0.68 |
| typeeq (unsup, type-aware) | 0.417 | 0.349 | 0.869 | 0.302 | 0.27 |
| oracle (true types) | 0.366 | **0.518** | 0.819 | 0.389 | **6.01** |

**Conclusion — rescaling fixes magnitude but fights detection:**
1. **It can fix magnitude** (oracle: E/I 0.29 → 6.0, Pearson 0.35 → 0.52). The bias IS
   rescalable if types are known.
2. **But it costs detection:** oracle Spearman 0.42 → 0.37, AUC 0.87 → 0.82 — because
   amplifying inhibition amplifies its noise too.
3. **Unsupervised versions limited:** `colnorm` partly fixes magnitude but hurts F1;
   `typeeq` does nothing because type-guessing fails on the compressed inhibitory
   neurons it needs to fix (**circular**).
4. **Big picture:** regularization (helps detection, hurts magnitude) and rescaling
   (helps magnitude, hurts detection) pull opposite ways. Root cause: inhibition's
   signal and noise are tangled; no linear reweighting separates them.

**Next:** the synthesis — **regularize FIRST** (zero the noise), **THEN rescale**
(amplify only the surviving clean signal). May get both detection and magnitude.

---

### 2026-06-18 — Regularization (Elastic Net) test
**Question:** Can Elastic Net clean up the exact connections (precision), as the
data-size test (variance-limited) predicted? And does it hurt the E/I magnitude
(bias), as predicted?

**Setup:** Real preprocessed feed, lag = 1.5 ms, FISTA Elastic Net. L1 swept
0…3e-2, L2 ∈ {1e-3, 1e-1}. Baseline = plain OLS. Script:
`scripts/regularization_test.py`.

**Result (L2=1e-3):**
| L1 | precision | recall | F1 | Pearson | E/I | %zeroed |
|----|----|----|----|----|----|----|
| OLS | 0.18 | 0.86 | 0.30 | 0.35 | 0.29 | 0% |
| 3e-3 | 0.29 | 0.77 | 0.42 | 0.34 | 0.21 | 42% |
| 1e-2 | 0.50 | 0.65 | 0.57 | 0.30 | 0.00 | 77% |
| 3e-2 | 0.84 | 0.49 | 0.62 | 0.24 | 0.00 | 92% |

**Conclusion — both predictions confirmed:**
1. **Regularization works for detection:** F1 0.30 → 0.62, precision 0.18 → 0.84.
   (The variance part, exactly as the data-size test said.)
2. **It worsens magnitude/inhibition:** Pearson 0.35 → 0.24 and E/I → 0.00 — inhibition
   is zeroed *first*. So regularization **cannot** fix inhibition; post-processing must.
3. **Why earlier Lasso failed:** wrong lag + λ too large. Useful λ range is tiny
   (1e-3…1e-2); ≥ 3e-2 zeros 92 %.
4. **L2 barely matters** — L1 does the work; not very collinear at this lag.
- Sweet spot ≈ L1 = 3e-3 (F1 0.42, keeps some inhibition) to 1e-2 (F1 0.57, inhibition gone).

**Next:** post-processing for the E/I magnitude bias — the one thing regularization
provably can't fix.

---

### 2026-06-18 — Data-size test: variance or bias?
**Question:** Is the ~0.37 ceiling caused by too little data (variance, fixable by
more data / regularization) or built into the method (bias)? This decides
regularization vs post-processing.

**Setup:** Same N=100 dataset, fixed lag = 1.5 ms (15 samples), recording truncated
to 2.5k / 5k / 10k / 20k / 35k / 50k samples. Inputs: spikes and preprocessed feed.
Script: `scripts/data_size_test.py`.

**Result:**
- Precision (exact connections): feed 0.10 → 0.18, **still rising** at 50 k.
- Pearson: feed 0.24 → 0.35, spikes 0.16 → 0.37, **still rising** (slowing).
- E/I ratio (truth 5.0): feed 0.47 → **0.29**, monotonically *worsening* toward ~0.3.
- (Side note: at small T the calcium feed beats raw spikes — its smoothing acts like
  extra averaging; spikes overtake only at 50 k.)

**Conclusion — it's BOTH, split by problem:**
1. **Exact-connection quality (precision, Pearson) is variance-limited** — still
   improving with data, so **regularization should help it** (same effect as more data).
2. **The inhibition-too-weak magnitude (E/I ratio) is bias-limited** — converges to a
   wrong ~0.3 regardless of data, so **regularization can't fix it** (would make it
   worse). Needs **post-processing** (separate E/I rescaling).

So regularization and post-processing are **not** alternatives — they target different
defects. Regularization → precision; post-processing → magnitude/E-I.

**Next:** pick a half to attack — (a) regularization at lag 1.5 ms for precision, or
(b) post-processing for the E/I magnitude bias.

---

### 2026-06-18 — Oracle ladder: where does the signal die?
**Question:** Are the small/overlapping estimated weights caused by the calcium
pipeline, or by the regression itself? And what is the effect of lag?

**Setup:** Existing N=100 dataset (`results/solver_comparison_N100/dataset`,
NE=80/NI=20, g=5 → true weights +10/−50, τ=100 ms, T=50 k, σ_extra=0.05). No new
simulation. Same centred-OLS estimator on three inputs — true spikes,
clean-calcium→feed, noisy-calcium→feed — swept over lag. Smoothing window fixed at
5 samples. Script: `scripts/oracle_ladder.py`. Headline metric: Pearson.

**Result (Pearson at best lag = 1.5 ms = 15 samples = synaptic delay):**
- spikes **0.370**, clean-calcium **0.370–0.385**, noisy-calcium **0.353**
- AUC at best lag: spikes **0.896**, noisy **0.870**
- Off-peak (lag 1.0 ms) Pearson ≈ 0 for all rungs.
- E/I magnitude ratio at best lag ≈ **0.3** (ground truth = 5.0) — compressed and inverted.

**Conclusion:**
1. **Lag is the dominant knob and the optimum is exactly the synaptic delay
   (1.5 ms).** The previous default of 10 samples (1.0 ms) sat off-peak at ≈ 0 —
   this likely explains a lot of earlier "regularization didn't work": it was being
   tested at the wrong lag.
2. **The calcium pipeline is not the bottleneck.** Noisy-calcium recovers ~95 % of
   the spike-level Pearson; the preprocessing is near-lossless at the right lag.
   Interesting side effect: calcium's low-pass *broadens* the usable lag window
   (spikes work only at exactly 1.5 ms; calcium works across 1.4–1.6 ms).
3. **The ceiling is intrinsic to the regression (~0.37 Pearson, ~0.90 AUC).**
   Detection is good; magnitude/correlation is the weak part — consistent with the
   collinearity + partialling-out story. So regularization/post-processing should
   target the regression, and we now know calcium is innocent.
4. **E/I ratio is not preserved** — needs per-class handling, not a single global scale.

**Next:** (a) finer characterisation of the lag peak width; (b) does more data (T)
shrink the overlap / raise the ceiling? (c) now-justified regularization test,
fixed at lag = 1.5 ms.

**Addendum (same day) — two follow-up checks (lag = 1.5 ms):**

*Does preprocessing matter?* Yes — earlier "calcium is innocent" was too strong.
Feeding RAW calcium with no preprocessing gives a *higher* Pearson (0.48) but
*worse* everything that counts: AUC 0.75 (vs 0.87), neuron-type accuracy 0.66
(vs 0.83), precision 0.11 (vs 0.18). The high raw Pearson is fool's gold — slow
calcium dynamics fake a correlation. The preprocessing removes that confound. So
the preprocessing does real work; it is just near-lossless *relative to spikes*,
not removable.

*Region vs exact-connection (the "duality"):* both confirmed with numbers.
Neuron-type (E/I) accuracy = **0.83** (good — regions right), but per-edge
precision = **0.18**, recall 0.86, F1 0.30 (poor — exact wiring unreliable).
Cause = averaging: one entry has SNR ≈ 1, but a whole row/column averages the
noise away (SNR ≈ √k). So the heat map looks right by region but no single cell
is trustworthy. Implication: post-processing/regularization should target the
*exact entries*, since the regions are already correct.

---

### 2026-08-14 — R.4/R.5 review: recall vs correlation "contradiction" explained; PR-AUC added; grid unification started

**Question:** `fig_R4` shows correlation up to ~0.8+ and excitatory recall up
to ~0.77 (N=1250), but `fig_R5_conf`'s confusion matrices show ~49% of real
excitatory edges landing in "predicted none" at N=12500. Is this a
contradiction? Is it fixable by more data alone?

**Answer — not a contradiction, two different operating points + two
different metrics:**
1. Correlation is an aggregate over ALL pairs (E/I/none combined); the huge
   "none" majority dominates it, so it stays high even when excitatory
   recall specifically is weak. High correlation never implied good
   excitatory recall.
2. R4's ~0.77 recall is N=1250's *long-recording plateau*; R5_conf's ~49-53%
   is N=12500 — a different, much larger network, not the same point on the
   same curve.
3. Pulled the actual numbers behind fig_R4 (see table below): N=12500's
   excitatory recall IS still climbing with more data, but decelerating hard
   — each doubling of T buys less (1M→2M: +0.087, 5M→7.5M: +0.036,
   7.5M→10M: +0.020) — while N=1250 is fully flat from 5M to 20M ms. **More
   data helps some, but the trend argues against N=12500 ever reaching
   N=1250's ~0.77 ceiling from data alone** — consistent with the
   established shared-input-confounding mechanism (in-degree grows with N
   under fixed connection probability), not a pure sample-size deficit.

| T (ms) | N=1250 E_rec | N=12500 E_rec |
|---|---|---|
| 1,000,000 | 0.724 | 0.274 |
| 5,000,000 | 0.770 | 0.466 |
| 7,500,000 | 0.779 | 0.502 |
| 10,000,000 | 0.778 | 0.522 |
| 20,000,000 | 0.771 | *(not yet run — see below)* |

**PR-AUC added (zero new compute):** `scripts/analyze_run.py`'s `score()` has
computed `pr_ap` (average precision / PR-AUC, connected-vs-none) all along —
it just wasn't plotted. Added as a 4th row to `fig_R4`/`fig_R4_TN`/`fig_R4_CI`
and a 3rd row to `fig_R5`/`fig_R5_lambda`. Confirms the ROC-vs-PR asymmetry
under class imbalance directly: at N=12500's longest recording, correlation
reaches ~0.86 but PR-AUC only reaches ~0.67 — same run, ~0.19 gap purely from
metric choice (see `docs/theory/metrics_glossary.md`, written up the same
day, for the full ROC/PR/imbalance explanation destined for the report).

**Also fixed:** `fig_R4`'s error bars are now a shaded ±1 SD band
(`fill_between`) instead of error-bar caps, and the point grid is unified
across all 4 sizes to one canonical 10-point list (100k…20M ms) via a
`GRID_MS` filter in `fig_r4_plot.py` — this silently drops the old
single-seed exploratory points (N=2500 @ 400k, N=5000 @ 400k/800k) that were
never part of the main grid, rather than plotting them inconsistently
alongside the properly-seeded ones.

**In progress (submitted, not yet returned):** extending N=2500/5000/12500 to
the same 20M-ms ceiling N=1250 already has, 5 seeds each, so every fig_R4
curve spans the full x-axis. N=12500 is by far the most expensive step here
(never run past 10M ms before) — split into a single-seed pilot
(`run_r4_n12500_extend20m_pilot.slurm`) to validate the (extrapolated,
unmeasured) ~10-12h/seed and ~650G/seed cost estimates before committing to
the full 5-seed array job. R5 stays single-seed / N=1250+12500 only (decided
against expanding its scope) but will pick up the same longer x-axis once
R4's moments exist at those checkpoints — cheap, since it only re-solves
cached moments (`run_r5_extend20m.slurm`), and N=1250's side of that can run
immediately (its 20M-ms moments already exist from an earlier job).

**Next:** run the N=12500 pilot, confirm cost, then the seeds-2-5 array job;
re-run `run_r5_extend20m.slurm` once both are done; regenerate all fig_R4/R5
figures with the complete grid.

---

### 2026-08-14 — Analytic linear (OU) ground truth: shared-input FPs vanish under full observation

**Question (from the professor):** a fully-observed regression should condition
on everything, so shared-input confounding shouldn't survive full observation
in principle — if it does (Way 2, `n1250_r4`), that argues the LIF network's
nonlinearity (or the calcium pipeline) is the real cause, not a fundamental
limit of regression. Test: swap in a ground truth that's genuinely, exactly
linear, and see if the Way 2 pattern goes away.

**Method:** `scripts/ou_linear_ground_truth.py`. Same connectivity as
`best_moments/n1250r4/adj_true.npy`, but the dynamics are a multivariate OU
process (`dx = A x dt + dW`, `A = -I/tau + s*G`, `G = adj_true.T`, s chosen for
a 20% stability margin) instead of spiking LIF neurons. Stationary `Cxx` and
lagged `Cyx` (lag = 0.1τ) are solved **analytically** (continuous Lyapunov
equation + matrix exponential) — no simulation, no finite-sample noise at all,
the cleanest possible version of "infinite data." Runs in ~10s locally, no
cluster. Fed straight into the existing `fig_way2.py` / attribution pipeline,
unmodified.

**Result — direct comparison, same N=1250, same connectivity, same procedure:**

| | LIF+calcium (real pipeline) | OU (exact linear) |
|---|---|---|
| 50% observed, FDR | 32.9% | 51.1% |
| 50% observed, hidden-driver slope | rises (5.2e-5) | rises (3.7e-5) |
| 50% observed, observed-driver slope | rises too (3.2e-5) | **flat/negative** (-1.2e-5) |
| 100% observed, FDR | 30.6% | **4.7%** |
| 100% observed, % FPs above true-negative median | 78% (28pp excess over chance) | **49% (~0pp excess — chance level)** |

**Interpretation:** on the exactly-linear system, full observation makes the
shared-input signature in false positives disappear almost completely (excess
over chance goes from +28pp to ~0pp; FDR drops 6x). This is what the
professor's argument predicts for a correctly-specified linear model — and
it's the opposite of what we measured on the real LIF+calcium data, where the
pattern persists at essentially the same strength whether 50% or 100% of the
network is observed. So: regression *can* fully remove shared-input confounding
under full observation, provided the model is actually linear and correctly
specified. The residual confound seen on real data is therefore attributable to
LIF nonlinearity and/or the single-lag/temporal-memory mismatch (see
`docs/theory/shared_input_theoretical_grounding.md` §4-5, "confound is diffuse
and multi-lag"), not to a fundamental inability of full-state regression to
remove shared input. Directly supports testing the joint multi-lag estimator
(`scripts/multilag_estimator.py`, built 2026-08-04, never run) and/or a
PIF/high-input-resistance spiking network as the next disambiguating step.

Even at 50% observed, note the interesting asymmetry: OU's *observed*-driver
slope is flat/negative (textbook — OLS conditions it away), while the real
data's observed-driver slope still rises almost as much as its hidden-driver
slope. That contrast, present even at 50% observed, is the same story in
miniature.

---
