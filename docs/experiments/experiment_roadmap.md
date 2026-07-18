# Experiment Roadmap — scaling & generalization study

Forward-looking plan for the paper/thesis experiments at scale (N=1250 / N=12500).
Complements the older files: `notebook.md` (tested findings, N=100 era),
`open_directions.md` (untested ideas from the N=100 pipeline work),
`methods_overview.md` (method scoreboard). Created 2026-07-18.

**Not everything here will be run.** This is a record of the possibilities so
nothing is lost; priorities mark what actually carries the story.

Priority: **P1** = core claims · **P2** = strengthens significantly · **P3** = revision/optional.

---

## Where we stand (the story so far)

Three results already in hand shape everything below:

1. **Regime matters.** A regime scan over (g, η, J) at N=1250 located a clean
   asynchronous-irregular state at **g=8, η=1.0, J=0.8** (~8 Hz, CV 0.80,
   synchrony 0.027) — vs the old g=5/η=2 (~42 Hz, regular). `scripts/regime_scan.py`
2. **The AI regime is data-limited, not uninferable.** Recovery curve at N=1250
   (50k→2M ms): **OLS ROC-AUC 0.72 → 0.99**, excitatory recall 0.30 → 0.89 —
   while EN/EN+Dale stay flat at ~0.60. `scripts/recovery_curve.py`
3. **Bias vs variance explains it.** OLS is unbiased → more data removes its error.
   L1 adds bias (over-shrinks the weak AI-regime weights) → data cannot undo it.
   λ=1e-4 was tuned for the strong-signal g=5 regime and is far too strong here.

Governing idea: accuracy is set by **(i) how correlated the regime is, (ii) how much
data you have relative to network size, (iii) whether the estimator is matched to that
data richness.**

---

## A. Data length & scaling

| ID | Experiment | Answers | Cost | Pri | Status |
|----|------------|---------|------|-----|--------|
| A1 | Recovery curve N=1250 (50k→2M ms) | Does more data rescue the AI regime? | done | P1 | ✅ 0.72→0.99 |
| A2 | Recovery curve N=12500 (1M/2M/5M ms) | Does it hold at full Brunel scale? | ~2.5–3 h | P1 | next |
| A3 | Add a mid-size N (2500 or 5000) → **required-recording-length vs N law** | "How long must I record for a network of size N?" | medium | P2 | proposed |

Note: scaling is **worse than proportional** — N=1250 at T/N≈400 gives AUC 0.725,
but N=12500 at T/N≈800 gives 0.620. Bigger networks need *more* than 10× the data
for 10× the neurons. A3 turns this into a quotable law.

## B. Estimator × data richness (bias–variance)

| ID | Experiment | Answers | Cost | Pri | Status |
|----|------------|---------|------|-----|--------|
| B1 | λ sweep at **short** T | Does *tuned* regularization beat OLS when data-poor? | cheap | P1 | proposed |
| B2 | λ sweep at **long** T | Confirms OLS wins / L1 bias hurts | cheap | P1 | partly |
| B3 | **Crossover figure**: OLS vs tuned-EN vs T | The headline bias–variance result | cheap | P1 | proposed |
| B4 | Does Dale (sign constraint) help more when data-poor? | Value of post-processing vs data | cheap | P2 | proposed |

B1 is **untested and important**: our EN/Dale never beat OLS, but λ was frozen at a
mis-scaled value. Best value-per-CPU-hour in the plan (all N=1250, short runs).

## C. Dynamical regime

| ID | Experiment | Answers | Cost | Pri | Status |
|----|------------|---------|------|-----|--------|
| C1 | Regime scan (g, η, J) → rate/CV/synchrony | Where is clean AI? | done | P1 | ✅ |
| C2 | **Inference quality across the regime map** (fixed T) | Which regimes are inferable? | medium | P1 | proposed |
| C3 | **AUC vs synchrony / CV / rate** | *Why* regimes differ — the mechanism | cheap (reuses C2) | P1 | proposed |
| C4 | Deviate from AI (bursty / synchronous) | More correlation = easier, but more confounded? | medium | P2 | 2 points so far |

C2+C3 generalize the "OLS wins in AI" observation from an anecdote to a statement.

## D. Signal model & preprocessing

| ID | Experiment | Answers | Cost | Pri | Status |
|----|------------|---------|------|-----|--------|
| D1 | **Synaptic-delay recovery**: sweep true delay (1.5/3/5 ms), check the inferred optimal lag tracks it | Can the method *read out* the true delay? | cheap | P2 | proposed |
| D2 | Deconvolution on/off | Why deconvolution matters | cheap | P2 | partly |
| D3 | τ mis-estimation sensitivity | Robustness to a wrong calcium τ | cheap | P3 | |
| D4 | Noise level (σ) sweep | Robustness to imaging noise | cheap | P3 | |
| D5 | **Realistic frame rate** (dt 0.1 ms → 30–100 Hz imaging) | Does it survive real imaging resolution? | cheap | P1 | proposed |

D1 framed as *delay recovery* (not just a lag sweep) is a causal-validity claim.
D5 is a major practical gap: we simulate at 0.1 ms; two-photon imaging is ~30 Hz.

## E. Realism of the observation model

| ID | Experiment | Answers | Cost | Pri | Status |
|----|------------|---------|------|-----|--------|
| E1 | **Partial observation**: record a subset (e.g. 100–1000 of 12500) | Real experiments see a *fraction* of the network | medium | P1 | proposed |
| E2 | Connection density ε sweep (sparser, cortex-like) | Generalization beyond ε=0.1 | medium | P2 | |
| E3 | Non-random topology (clustered / small-world) | Does structure help or hurt? | medium | P3 | |
| E4 | Wrong assumed density at threshold | Sensitivity of the 10% density rule | cheap | P3 | |

E1 is arguably the most important realism gap: unobserved neurons become **hidden
confounders** — the same mechanism that caps excitatory recovery. It is the bridge
from "works in simulation" to "works on real data".

## F. Robustness / bookkeeping

| ID | Experiment | Answers | Cost | Pri | Status |
|----|------------|---------|------|-----|--------|
| F1 | Multiple seeds at N=12500 | Error bars at scale | expensive | P3 | deferred (revision) |
| F2 | Seeds at N=1250 for all headline curves | Error bars on the main claims | cheap | P2 | partly (3–5 seeds) |

---

## Suggested ordering

1. **Now:** HPC infra (on-cluster analysis, float32, speedup) → **A2**.
2. **Cheap high-value batch** (all N=1250, fast): **B1–B3**, **D5**, **D1**.
3. **Then:** **C2/C3** (regime map), **E1** (partial observation).
4. **Revision/thesis:** A3, C4, D2–D4, E2–E4, F1.

Resulting narrative: *accuracy is governed by regime correlation, data-per-parameter,
and estimator–data matching — and here is the recipe.*
