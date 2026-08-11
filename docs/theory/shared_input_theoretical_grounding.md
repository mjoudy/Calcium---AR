# Theoretical grounding for the shared-input confound (Pernice et al.)

Notes from a working session connecting the project's empirical shared-input
findings (`docs/experiments/shared_input_findings.md`, i.e. R.7/R.8/Way2/Way3)
and the regression write-up (`docs/theory/connectivity_regression_chapter.tex`)
to two ancestor papers from the professor's group:

- Pernice, Staude, Cardanobile, Rotter (2011) *How Structure Determines
  Correlations in Neuronal Networks*, PLoS Comput Biol 7(5): e1002059.
- Pernice, Staude, Cardanobile, Rotter (2012) *Recurrent Interactions in
  Spiking Networks with Arbitrary Topology*, Phys. Rev. E 85, 031916.
- Pernice & Rotter (2013) *Reconstruction of Sparse Connectivity in Neural
  Networks from Spike Train Covariances*, J. Stat. Mech. P03008.
- Schiefer, Niederbühl, Pernice, Lennartz, Hennig, LeVan, Rotter (2018) *From
  Correlation to Causation: Estimating Effective Connectivity from Zero-Lag
  Covariances of Brain Signals*, PLoS Comput Biol 14(3): e1006056.

Purpose: a reference for later report/write-up sections that need to justify
*why* shared-input analysis is a reasonable lens for a regression-based
connectivity estimator, not just a leftover concern from correlation-era
methods. Dated 2026-08-02, extended 2026-08-11 (Section 6).

## 1. Correlation is a sum over network motifs (Pernice 2011)

The full covariance matrix decomposes exactly as
`C = Σ_{n,m} G^n · Y · (G^T)^m`. Each term `G^(n,m)` is a specific motif: paths
of length `n` from a common ancestor `k` reaching neuron `i`, length `m`
reaching neuron `j`, weighted by `k`'s rate. The project's "common presynaptic
driver count" (R.7's `shared_panel`, Way 2) measures exactly one term of this
series, `g^(1,1)` — **direct (1-hop) shared input**. Indirect shared input
(`(2,2)`, `(3,3)`, ...) is not counted at all.

**Untested idea motivated by this:** extend the R.7/Way2 driver-count metric to
2-hop (or weighted, multi-hop) common ancestors and see if it explains more
false-positive variance than the 1-hop count alone. Not built yet.

### Symmetric motifs explain the R.8 symmetry filter, not just describe it
`G^n·Y·(G^T)^n` (n=m) is symmetric by construction; the direct-edge term
`G^1·Y·(G^T)^0` is not. This is a structural fact of the linear model, not an
empirical coincidence: common-input motifs are inherently symmetric, direct
synaptic motifs are inherently asymmetric. It's the theoretical reason R.8
phase 1 (symmetry filter) works.

### Two different kinds of "cancellation" — resolves a specific question raised
- **Same-order (direct) cancellation does not happen.** In a Dale's-law
  network, a single driver `k`'s weights to any two targets share the same
  sign (`w_ki·w_kj ≥ 0` always), so direct shared drivers only ever add,
  never cancel.
- **Cross-order cancellation is real** (Pernice 2011, Eq. 23): in
  inhibition-dominated networks, successive orders of the motif expansion
  alternate in sign, each partially cancelling the previous. This is a
  plausible source of the non-monotonic wiggle seen in the generated
  `fig_way2_n1250` plot (driver-count bins don't rise monotonically).

### Timing/delay is a shape feature, not a magnitude feature
Pernice 2011 (Methods): "differences in the shape of interaction kernels which
do not alter the integral do not affect our results... delays only shift
interaction kernels in time." The confound magnitude lives in the
**integrated** covariance; delay is layered on top. Consistent with the
project's R.8b finding (timing separates real/fake on spikes but calcium blur
erases it) and Way 2 (single-lag OLS is an imperfect deconfounder) — the
ceiling isn't really about resolving delay better, it's about the estimator
only capturing first-order motifs given a single lag.

## 2. Regression vs correlation are not different paradigms (Pernice 2013)

This directly answers a live objection from the user's professor: *"shared
input is a correlation/covariance concept, your approach is regression-based,
why is it relevant?"*

The project's estimator, `A = Cyx @ inv(Cxx)`, **is** a linear regression, and
its coefficients are computed entirely from covariance matrices — this is not
incidental, OLS coefficients are always `Cov(y,x)·Cov(x,x)⁻¹`. Regression is a
particular way of using the covariance matrix (conditioning away other
neurons), not a different object from it.

Pernice & Rotter (2013) derive the inverse covariance explicitly:
`Ĉ⁻¹ = Ŷ − Ĝ*Ŷ − ŶĜ + Ĝ*ŶĜ`. The last term, `Σ_k Ḡ_ki·Y_kk·G_kj`, is exactly
a shared-input artifact ("marrying parents of joint children"). Plain
regression / partial-coherence-style estimates do **not** remove this term —
their whole paper exists to add an extra assumption (L1 sparsity) to suppress
it. This is the literal theoretical prediction of what the project's Way 2
found empirically (single-lag OLS leaks observed shared input into fakes).

**Talking point for the professor:** "The regression estimator is literally
`Cyx·Cxx⁻¹`, a covariance-derived quantity. Pernice & Rotter (2013) show this
class of estimator retains a shared-input term in its inverse-covariance
expansion unless extra structure (sparsity) is imposed. Shared-input analysis
isn't borrowed from a different framework — it's the established lens for
what my own estimator is and isn't removing."

## 3. Functional / effective / structural connectivity

Standard three-way distinction (Friston-era systems neuroscience vocabulary),
useful for framing the whole project:

- **Structural (anatomical) connectivity** — real synapses. `adj_true` in this
  project's ground truth.
- **Functional connectivity** — plain statistical dependency (correlation),
  symmetric, no model of who influences whom. This is what shared input
  contaminates.
- **Effective connectivity** — a directed, model-based estimate of causal
  influence, obtained by fitting a generative model (Hawkes, linear response,
  the project's regression) to the data. Meant to approximate structural
  connectivity better than raw functional connectivity.

Framing: the regression step's whole purpose is to strip functional-
connectivity contamination (shared input) and recover something closer to
effective/structural connectivity. Shared-input analysis is literally asking
how much functional-connectivity leakage survives into the effective-
connectivity estimate — the natural diagnostic for this method, not an
import from a different one.

## 4. Uniqueness of the regression solution vs the correlation-only problem

Apparent tension, resolved: Pernice 2013's zero-lag (fully time-integrated,
`ω=0`) covariance matrix is symmetric, so it under-determines a directed `G`
— many `B` matrices (related by unitary transformations) reproduce the same
`Ĉ`. They resolve this by imposing sparsity (L1 minimization) to pick one.

The project's regression, by contrast, uses **lagged** covariance
(`Cyx` at lag `l > 0`), which is generally *not* symmetric and already encodes
direction from time order. For a **fixed specification** (fixed lag, fixed
predictor set, OLS), the solution is unique and closed-form — no search, no
sparsity tie-break needed. This is a genuine advantage over Pernice 2013's
zero-lag-only reconstruction problem.

**But uniqueness ≠ correctness.** The single closed-form answer is exactly
right only if the assumed model (in particular: a single lag matching the true
delay structure, linear dynamics) matches the true generative process. Way 2
shows shared-input confounding spreads across multiple lags, so a fixed
single-lag regression's unique answer is still systematically biased — one
right answer to a slightly wrong question, not "ambiguous" the way Pernice
2013's problem is, but wrong in a specific, reproducible way.

Separately, *different specification choices* (which lag, which regularizer,
which neurons observed) each give their own unique-but-different answer — this
is a researcher-degrees-of-freedom question, not the same kind of
non-uniqueness as Pernice 2013's unitary-group ambiguity.

## 5. Regression-as-causality and its limits

Predicting each neuron's state from *other neurons' past* activity is Granger
causality — using temporal precedence to break the symmetry pure correlation
can't break. Real and valid (Pernice 2013 discussion: "non-zero couplings are
equivalent to... linear Granger-causal relation").

**Caveat (well-established in the causal-inference literature, not project-
specific):** Granger-style causality is still vulnerable to common-driver
confounding when the driver has *different delays* to different targets. If
driver `k` reaches `j` faster than `i`, `j`'s past will look like it
Granger-causes `i`'s future — both are just downstream echoes of `k` at
different lags. Same shared-input problem, dressed up as directional. Time
order protects against *simultaneous* confounds, not confounds spread across
time — exactly the regime Way 2 found the project is in.

## 6. Network-size scaling convention: what the professor's own papers actually do

Directly relevant to R.4/R.4b (does the excitatory-recall gap between network
sizes reflect biology, or the scaling convention used to grow N?). Checked
against the actual PDFs (2026-08-11), not recollection:

**All four papers above, whenever they touch more than one network size at
all, use FIXED CONNECTION PROBABILITY `p`, not fixed in-degree.**
- Pernice 2011: `p` is the primary variable; only one N is simulated, but the
  paper notes that for large random (fixed-`p`) networks, each neuron's
  *realized* in-degree still concentrates tightly around the mean (law of
  large numbers on a Binomial degree distribution) — so "regular" (exactly
  fixed-degree) and "random" (fixed-`p`) networks behave similarly *within
  one size*. This does not by itself resolve how to scale *across* sizes.
- Pernice 2012 (Phys. Rev. E), Section VI: explicitly scales network size N
  (Fig. 4/5) "assuming a uniform connection probability `p`" — fixed `p`,
  degree grows with N. This is the clearest direct precedent, and it
  contradicts what an earlier chat session claimed (that "balanced-network
  theory typically fixes in-degree when scaling N") — that classical
  argument (van Vreeswijk & Sompolinsky's original dilute-limit derivation)
  is a *different, more mathematical* tradition than what this group's own
  papers actually do.
- Pernice & Rotter 2013, Schiefer et al. 2018: both sweep `p` at a couple of
  *fixed* N values (e.g. n=50 vs n=300) to explore performance across a grid
  — N and `p` are used as independent knobs to characterize an algorithm, not
  as a coupled "how do I grow the network" scaling law. Note 2013's Fig. 3:
  **larger N performed *better* at their reconstruction task** (weights not
  too strong), the opposite direction from this project's original R.4
  finding — though not a directly comparable setup (different method, no
  attempt to hold the dynamical regime fixed across sizes).

**Best guess (not confirmed, worth asking the professor directly) for *why*
fixed probability is the group's standard, despite fixed in-degree being the
more biologically-motivated choice:** likely mathematical convenience. Fixed-`p`
random graphs are i.i.d. Bernoulli (Erdős–Rényi) — the ensemble that random
matrix theory (circular law, spectral radius formulas) and the mean-field
self-consistency equations throughout these papers are built for. Regular
(exactly fixed-degree) graphs don't fit that toolbox as cleanly. That's a
plausible reason for a *methods/theory* paper to prefer fixed `p` — it may not
have been a deliberate claim about which convention is more biologically
faithful when *scaling N*, since none of these papers actually pose that
specific question (they study one size, or treat N/`p` as independent
exploratory axes — this project's R.4 ladder is the one asking "make the same
kind of network bigger and see what happens").

**Implication for this project:** citing "the professor's own papers use fixed
probability" is accurate and defensible as precedent — but it's precedent for
a *different question* than R.4/R.4b's. Worth presenting both conventions
explicitly in the report (as already done — see `docs/experiments/notebook.md`
2026-08-08/2026-08-10 entries) rather than picking one as unambiguously
"correct," and worth asking Prof. Rotter directly why fixed probability became
standard in the group's work — the mathematical-convenience explanation above
is this project's best guess, not a confirmed fact.

## Bottom line / how to use this in reports

Shared-input confounding is not a correlation-specific artifact that
regression sidesteps. It is a property of the network's causal structure (a
real driver feeding multiple targets) and it leaks into *any* estimator built
on statistical dependency — raw correlation, partial coherence, sparsity-
regularized inversion, or lagged regression — unless the estimator actually
conditions on the true driver at the true lag(s) it acts on. R.7/Way2/Way3
are testing exactly that, and their empirical findings (single-lag OLS is an
imperfect deconfounder; fakes track observed drivers almost as much as hidden
ones) are independently predicted by this theory, not just consistent with
it.

See also: `docs/experiments/shared_input_findings.md` (empirical record),
`docs/theory/connectivity_regression_chapter.tex` (regression derivation).
