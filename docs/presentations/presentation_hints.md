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

<!-- Add new hints below this line, one section per topic. -->
