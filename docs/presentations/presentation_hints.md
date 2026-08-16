# Presentation Hints

Running cheat sheet of things to have instantly ready when presenting.
Short by design — add new entries below as they come up, don't expand old ones.

---

## 1. Axon / dendrite convention in the adjacency matrices

- **Axon** = output (sends). **Dendrite** = input (receives). Synapse = axon → dendrite.
- **Field convention** (and Dale's law): **column = axon/sender**, **row = dendrite/receiver**.
  An inhibitory neuron = a negative *column* (all of its outgoing weights share one sign).
- **`adj_true`** (raw, as saved from NEST) is stored the *opposite* way — `adj_true[source, target]`,
  i.e. row = axon. This just mirrors NEST's `GetConnections(source=, target=)` field order.
- **`A`** (every solver's estimate) is already column = axon, matching the field convention —
  standard regression form `x(t) = A · x(t−1)`, row = neuron being predicted (receiver).
- **Fix applied everywhere:** every plot/metric/comparison uses `adj_true.T` (or index-swap
  `adj[j, i]`) before touching `A`, so what you see in a figure is always in the field-standard,
  column = axon orientation. Verified numerically (toy AR sim): un-transposed correlation ≈ 0,
  transposed correlation = 1.0. No results are affected — this is bookkeeping, not a bug.
- **One-liner if asked:** *"Field convention puts the sender on the column. NEST's raw dump comes
  out sender-on-row instead, so we transpose once before comparing or plotting — everything you
  see is already field-standard."*

---

## 2. Ground truth vs. Brunel 2000 — how faithful, what changed, why

- **Exactly Brunel only at N=12500**: ε=0.1, J=0.1mV, V_reset=10mV, g=6/η=4 = Fig 8B (AI). Smaller N are *regime-matched variants*, not literal Brunel — say this plainly if asked.
- **g=8 not canonical g=6** in the scaling ladder: a 2-D probe (g × η) showed g=6 caps CV~0.63 at a realistic ~14Hz rate; g=8 (J rescaled) reaches CV~1 (textbook AI) at the same rate. Documented, not arbitrary.
- **ε (connection probability) was held fixed at 0.1 in every regime search this project has ever done** (the 2-D probe, all η-tuning, the whole R.4 ladder) — confirmed by reading the code. C_E grows with N as a result.
- **Tested the alternative** (fixed in-degree instead, ε shrinks with N) — motivated by biological realism (dendritic-tree-limited in-degree, ~N-independent). Preflight-checked: same CV/sync as the original ladder, only ~1Hz rate offset — not a different regime. **Confirmed result: fixes the excitatory-recall/precision N-gap, and reverses it** (bigger N gets better, not worse) — see hint 3.
- **One-liner if asked "is this really Brunel?"**: *"Structurally yes — same neuron model, same wiring rule. Parametrically exact only at N=12500. Every departure (g=8, per-size η, the in-degree test) is a probed, documented choice, not an arbitrary one."*

---

## 3. Fixed probability vs. fixed in-degree — the N-scaling result, and what the professor's own papers do

- **The finding:** under fixed connection probability (ε=0.1 always, in-degree grows with N), excitatory recall/precision get structurally *worse* with N — a real, data-independent ceiling (checked with 20M-ms recordings, doesn't converge). Under fixed in-degree (C_E=100 always, ε shrinks with N), the gap not only closes, it **reverses** — N=12500 ends up *best* (recall/precision ~0.99/0.98), beating even N=1250's fixed-probability ceiling (~0.77/0.75).
- **Mechanism:** fixed probability means in-degree (shared-input pathways per neuron) grows with N — more confounding as networks get bigger. Fixed in-degree caps that at a constant, biologically-plausible level regardless of N.
- **Checked against Prof. Rotter's own papers (Pernice 2011/2012/2013, Schiefer et al. 2018):** all of them use fixed connection probability whenever they touch network size at all (clearest case: Pernice 2012 Fig. 4/5, explicit N-scaling at fixed p). So there's real precedent for fixed probability in the group's own work — **but none of those papers actually ask R.4's specific question** (grow the same network, hold the dynamics matched, see what happens to inference) — they either study one size, or sweep N and p as independent exploratory knobs, not as a coupled "how should I scale" law.
- **Best guess why fixed probability is standard there (unconfirmed — worth asking directly):** likely mathematical convenience — fixed-p (Erdős–Rényi) graphs are the ensemble their random-matrix-theory/mean-field results are built for, not a deliberate claim about biological realism at scale.
- **One-liner if asked "which is correct?"**: *"Both have a place: fixed probability matches this group's own precedent and NEST's classic example; fixed in-degree matches how real cortex actually scales — a neuron's input count is capped by its dendritic tree, not by how big the surrounding tissue sample is. We show both explicitly because the choice turns out to change the conclusion completely, which is itself a finding worth reporting."*

## 4. Linearity boundary — LIF vs PIF ladder vs OU vs Hawkes

- **Two independent linear model classes, not one, both confirm the hypothesis:** shared-input
  false-positive rise at 100% observed — **LIF (real) 14.2pp, PIF ×10 6.9pp, PIF ×100 −0.3pp,
  Hawkes+calcium (linear point process) 3.3pp, OU (exact linear, no sim noise) −0.1pp.** A
  continuous exactly-linear process (OU) AND a real linear point process (Hawkes, matching
  Pernice et al. 2011's own model) both show the confound shrink; the PIF ladder shows it shrink
  *monotonically* as leak is removed. Directly confirms the professor's hypothesis, twice over.
- **But overall detection quality is a SEPARATE axis — don't conflate the two.** AUC / precision:
  LIF 0.943/0.694, PIF×10 0.730/0.371, PIF×100 0.709/0.341, Hawkes 0.578/0.185, OU 1.000/0.953.
  Every linear alternative to LIF has *worse* overall detection, not better — confound-structure
  (which mistakes) and raw signal strength (how many mistakes) are different things. LIF's
  strength on the second axis reflects this project's own extensive tuning toward a strongly-
  correlated realistic regime; PIF/Hawkes were tuned to test linearity, not maximize SNR, and hit
  real, separate, identified ceilings trying to close that gap (PIF: burstiness is *forced* by
  removing the leak, CV 0.98→1.94→6.14→47 at ×1/×10/×100/×10000, not fixable by re-tuning eta;
  Hawkes: hard stability ceiling, spectral radius must stay <1, unlike LIF's threshold-reset
  which has no such linear cap).
- **A large relative rate increase can still be a small share of total errors.** New attribution
  metric: of LIF's false positives, only ~2.8% are attributable to shared-input exposure in
  absolute-count terms (baseline-rate-implied vs actual), despite a 5.4× relative rate rise —
  because most non-edges simply don't have many shared drivers. Worth citing if asked "so is
  shared input really the main problem" — the mechanism is confirmed real, but concentrated on a
  minority of pairs, not the majority of the error count.
- **Partial observation is a separate problem, not the same one:** the exact-linear OU system
  shows the SAME steep rise in false-positive rate at 50% observed as LIF does. So "nonlinearity"
  and "hidden neurons" are two independent causes, not one — R.7 already showed hidden neurons
  matter in aggregate; this isolates that from the linearity question cleanly.
- **Multi-lag conditioning (joint VAR, several lags at once) gives a real, measured improvement**
  on calcium feed (precision 0.474→0.594, false positives −32%) — but the false positives it
  fixes have the *same* average shared-driver exposure as the ones that persist (attribution
  ratio ≈1.00). Real effect, not proven to be a shared-input-specific fix — say "a real lever,"
  not "solves the confound."
- **Do NOT claim "regression is less prone to shared-input confounding than correlation" from
  this work** — every arm above used the same regression estimator; only the ground truth varied.
  That's a real, different, buildable comparison (swap the estimator, not the ground truth) —
  not run. If asked, say plainly it hasn't been tested, and regression conditioning on observed
  neurons is a real but partial mitigation (Pernice & Rotter 2013 show a shared-input term
  survives explicitly in the regression solution — that's *why* their paper adds sparsity on top).
- **One-liner if asked "does more data / more neurons fix this?"**: *"No — tested directly. A
  perfectly linear network, fully observed, shows the confound vanish. The real spiking+calcium
  system doesn't, even fully observed. That rules out 'just need more data' — it's a linearity
  ceiling, now isolated from the separate, also-real, partial-observation ceiling."*
- **One-liner if asked "so is linear better overall, not just for this one bias?"**: *"No, and
  that's a real, separate finding — linearity fixes the shared-input-specific bias but doesn't
  come with better overall detection for free. Every linear ground truth we could build had a
  real, identified reason (burstiness, or a hard stability ceiling) for weaker raw signal than
  real LIF, whose parameters this project already spent a lot of effort tuning. Two different
  axes, not a contradiction."*

---

## 5. Is the method "failed"? Who is this actually for?

- **No — a method with a mapped, understood limitation isn't broken, it's characterized.**
  Inhibition: solid everywhere. Excitatory: solid when the system is near-linear and
  well-observed. Outside that, the cause is now known (nonlinearity + partial observation,
  tested separately) rather than an unexplained ceiling.
- **Not unique to this project.** Pernice & Rotter (2013) — the professor's own cited paper —
  needed sparsity regularization added on top of plain regression specifically because plain
  regression alone retains a shared-input term. No one has solved this with a smarter regression
  alone; it's a known, field-wide limit of passive/observational connectivity inference generally
  (the neuroscience instance of unmeasured confounding in causal inference — unsolvable from
  observation alone without either full linearity+observation, or intervention).
- **Real audience:** (1) experimentalists — as a calibrated candidate-ranker ahead of expensive
  targeted follow-up (patch-clamp, opto), and for relative comparisons across conditions even
  without perfect absolute accuracy; inhibition alone is already usable as-is. (2) the field —
  knowing precisely where/why this class of method breaks redirects future effort (multi-lag,
  faster indicators, intervention-based methods) instead of assuming more data would fix it.
- **One-liner if asked "so what is this good for?"**: *"Not a magic wiring-diagram reader — a
  scoped, honest tool: reliable for inhibition and for excitatory connectivity in near-linear,
  well-observed regimes; outside that, a calibrated candidate-ranker rather than ground truth,
  with the ceiling's actual cause isolated instead of just observed."*

<!-- Add new hints below this line, one section per topic. -->
