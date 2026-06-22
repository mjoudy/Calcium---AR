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
