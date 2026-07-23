# Brunel Network Configuration Reference

> **VERIFIED 2026-07-18 — measured on this codebase at N=12500. Do not re-derive
> or re-search these; they are simulation results, not recollection.**
>
> ### Which (g, eta) is which regime — Brunel 2000 Fig 8
> | Fig | g | eta | regime | measured here (N=12500, J=0.1, V_r=10) |
> |-----|---|-----|--------|-----------------------------------------|
> | 8B | **6** | **4** | **asynchronous irregular (AI)** | rate 58.7 Hz, CV 0.86, sync 0.010 |
> | 8C | 5 | 2 | synchronous irregular (SI, fast) | rate 37.4 Hz, CV 0.43, sync 0.017 |
> | —  | 6 | 1.5 | AI at a cortical rate (eta lowered) | rate 13.7 Hz, CV 0.62, sync 0.008 |
>
> **g=5, eta=2 is NOT the AI point.** It is Fig 8C. That is the default in the
> widely-copied NEST `brunel_delta_nest.py` example, which is why it gets
> mislabelled as "the Brunel network". The AI state is **Fig 8B: g=6, eta=4**.
>
> ### Firing rate is set by eta (balance argument)
> mean input = eta*theta + J*tau*nu*(C_E - g*C_I). Setting it at threshold:
> * g=6, J=0.1, C_E=1000:  **nu ~ 20*(eta - 1)**  -> eta=4 gives ~60 Hz, eta=1.5 gives ~10 Hz.
> * Lower rate also means smaller fluctuations (sigma ~ J*sqrt(C_E*nu*tau)), so CV
>   drops as you lower eta: 0.86 @ 59 Hz -> 0.62 @ 13.7 Hz. **At fixed g and J**
>   rate and irregularity cannot be maximised independently — but g and J are not
>   fixed, see the next block: raising them recovers CV ~1 at 14 Hz.
>
> ### VERIFIED 2026-07-23 — rate and irregularity are NOT in conflict
> 2-D probe over (g, eta) at N=1250, three J scales, 144 configs
> (`scripts/r4_probe_2d.py`, raw numbers in `results/regime2d/*.csv`,
> figures `figures/fig_regime2d_j*.pdf` and `fig_regime2d_ridge.pdf`).
>
> Walking along the **14 Hz iso-rate line** (eta re-fitted at each g):
>
> | J | g=5 | g=6 | g=8 | g=10 | g=12 |
> |---|-----|-----|-----|------|------|
> | 0.316 (x1)   | CV 0.61 | 0.65 | 0.74 | 0.83 | 0.95 |
> | 0.474 (x1.5) | 0.75 | 0.82 | **0.97** | 1.10 | 1.23 |
> | 0.632 (x2)   | 0.90 | 0.98 | 1.16 | 1.30 | 1.45 |
>
> CV rises with **both** g and J at fixed rate, while synchrony *falls* with g
> (0.017-0.034 at g=5 down to ~0.005 at g=12), and no neuron goes silent anywhere
> on the map. So CV ~0.63 was a property of the (g=6, J=0.316) corner, not a
> ceiling imposed by wanting a cortical rate.
>
> **Direction of CV vs eta flips between families.** At g=6, J x1 raising eta
> *lowers* CV (0.72 -> 0.55 over eta 0.9 -> 2.0); at g=8, J x1.5 it *raises* CV
> (0.61 -> 0.95 at N=12500). The high-g/high-J family stays fluctuation-driven, so
> pushing the drive up to reach a target rate also buys irregularity.
>
> ### VERIFIED 2026-07-23 — the R.4 scaling ladder (one regime, four sizes)
> g=8, V_reset=10, J fluctuation-scaled (J*sqrt(C_E) = 1.5*0.1*sqrt(1000) = 4.743),
> eta tuned per size to ~14 Hz (`scripts/r4_tune_regime.py --g 8 --j-scale 1.5`).
> 4 s sims, 1 s warm-up, seed 1:
>
> | N | C_E | J | eta | rate | CV | sync |
> |---|-----|---|-----|------|----|------|
> | 1250  | 100  | 0.474 | 1.30 | 14.8 Hz | 0.98 | 0.010 |
> | 2500  | 200  | 0.335 | 1.50 | 13.9 Hz | 0.98 | 0.008 |
> | 5000  | 400  | 0.237 | 1.90 | 14.4 Hz | 1.03 | 0.007 |
> | 12500 | 1000 | 0.150 | 2.60 | 13.9 Hz | 1.06 | 0.007 |
>
> Rate matched to +/-3%, CV 0.98-1.06, synchrony <=0.010 across a 10x size range —
> so in the R.4 sweep **N is the only variable**. Presets `n{1250,2500,5000,12500}_r4`
> in `scripts/wrapup_run.py`. Note eta must RISE with N to hold the rate under
> fluctuation-scaling (1.30 -> 2.60), and g=8 is a deliberate departure from
> Brunel's g=6 — the canonical point stays in the `n12500` preset.
>
> ### V_reset matters and was wrong in this codebase until 2026-07-18
> `BrunelNetwork` hardcoded **V_reset = 0**; Brunel uses **V_r = 10 mV** (as this
> document already stated). V_r=0 suppresses irregularity — measured at N=1250,
> g=5/eta=2: CV **0.58** at V_r=0 vs **1.67** at V_r=10. It shifts the whole phase
> diagram. `V_reset` is now a parameter (default 0.0 to keep old results
> reproducible); canonical presets set 10.0.
>
> ### "Canonical" only applies at N=12500
> Brunel's parameters are defined at N_E=10000, N_I=2500, eps=0.1 -> **C_E=1000**,
> J=0.1 mV. Downscaling forces a J-rescaling choice (mean- vs fluctuation-
> preserving) and no choice preserves every property, so smaller nets are
> *variants*, not canonical.
>
> ### Presets (scripts/wrapup_run.py)
> * `n12500` — canonical Fig 8B AI (g=6, eta=4, V_r=10)
> * `n12500_lowrate` — same but eta=1.5 (~14 Hz, cortical rate)
> * `n12500ai` — deprecated early variant (J=0.23, g=8, eta=1, V_r=0)
>
> Check any config cheaply with:
> `python scripts/regime_probe.py --scale n12500 --only brunel_AI_fig8B --sim-time 5000 --fig out.png`


## The Canonical Brunel 2000 AI State

Parameters from Brunel (2000) "Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons":

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| Excitatory neurons | NE | 8,000 | (in a 10,000-neuron network) |
| Inhibitory neurons | NI | 2,000 | NI = NE/4 |
| Connection probability | ε | 0.1 | sparse random |
| Exc. in-degree | CE | 800 | = ε × NE |
| Exc. synaptic weight | J | 0.1 mV | EPSP amplitude |
| Inh./Exc. weight ratio | g | 5–6 | g > 4 = inhibition-dominated |
| External drive ratio | η (= νext/νth) | 2.0 | moderate drive above threshold |
| Membrane time constant | τm | 20 ms | LIF neuron |
| Refractory period | τrp | 2 ms | |
| Transmission delay | D | 1.5 ms | |
| Firing threshold | θ (Vth) | 20 mV | |
| Reset potential | Vr | 10 mV | |

**AI state signature:** CV(ISI) ≈ 1.0, mean rate 30–60 Hz, raster shows "salt-and-pepper" (no synchrony)

---

## The Key Scaling Invariant

When you reduce N below 10,000, you must scale J to preserve network dynamics:

```
J_ex × C_E = 80   (conserved quantity)
```

Derived from Brunel 2000: C_E = 800, J = 0.1 mV → J × C_E = 80

Auto-scaling rule:
```python
J_ex = 80.0 / C_E   where C_E = int(epsilon * n_excitatory)
```

---

## Operating Points by Network Size

### N = 100 (Local testing / code validation only)

| Parameter | Value |
|-----------|-------|
| NE / NI | 80 / 20 |
| CE | 8 |
| J_ex (auto-scaled) | 10.0 mV |
| g | 5.0 |
| η | 2.0 |
| **CV(ISI)** | **≈ 2.7** |
| **Mean firing rate** | **≈ 130–160 Hz** |
| V_th / σ | 0.40 |
| **Regime** | **Few-synapse (catastrophic)** |

⚠️ **Not a valid Brunel AI state.** With CE=8, each inhibitory spike delivers −g×J = −50 mV (half the reset potential in one event). Mean-field theory breaks down completely. Use N=100 **only** to test code paths quickly — do not interpret the neuroscience.

---

### N = 1,250 (HPC target — primary research config)

| Parameter | Value |
|-----------|-------|
| NE / NI | 1,000 / 250 |
| CE | 100 |
| J_ex (auto-scaled) | 0.8 mV |
| g | 5.0 |
| η | 2.0 |
| sim_time | 10,000 ms (testing) / 1,000,000 ms (production) |
| **CV(ISI)** | **≈ 0.58** |
| **Mean firing rate** | **≈ 42 Hz** |
| V_th / σ | 1.27 |
| **Regime** | **Best approximation of AI state at this scale** |

✅ **Recommended for research.** All neurons active, irregular firing, reasonable rate. V_th/σ=1.27 (vs ideal 3.59) means it's more fluctuation-dominated than canonical Brunel, but it is a valid asynchronous network. This is the best achievable at N=1250.

---

### N = 10,000+ (Future HPC / canonical regime)

| Parameter | Value |
|-----------|-------|
| NE / NI | 8,000 / 2,000 |
| CE | 800 |
| J_ex | 0.1 mV |
| g | 5.0–6.0 |
| η | 2.0 |
| **CV(ISI)** | **≈ 1.0** |
| **Mean firing rate** | **≈ 40 Hz** |
| V_th / σ | 3.59 |
| **Regime** | **True Brunel AI state** |

✅ **Canonical config.** Requires N≥10,000 (CE=800 at ε=0.1). This is HPC-only.

---

## Why CV(ISI) < 1 at N=1250

The Brunel AI state emerges in the **diffusion limit** where many small EPSPs/IPSPs
sum to give roughly Gaussian input fluctuations. The key ratio:

```
σ = sqrt(J² × (τm/2) × (CE × νE + CI × νI + Cext × νext))
V_th / σ ≈ 3.59  (Brunel 2000, canonical)
V_th / σ ≈ 1.27  (N=1250, J=0.8)
```

At N=1250, σ is ~12× larger relative to threshold (each synapse is 8× stronger).
This pushes the network out of the diffusion regime → CV drops below 1.

---

## Calcium Signal Parameters (from archive)

These are the production values used in the archived HPC pipeline:

| Parameter | Value | Notes |
|-----------|-------|-------|
| tau (decay) | 100 ms | true calcium decay time constant |
| dt (calcium) | 1 ms | spikes rebinned from 0.1 ms NEST resolution |
| amplitude | 1.0 | ΔF per spike |
| sigma_intra | 0.01 | intracellular noise std |
| sigma_extra | 1.0 | recording noise std (⚠️ current default is 0.05) |

⚠️ **sigma_extra discrepancy:** The archive used `sigma_extra=1.0`; `ExperimentConfig` defaults to `0.05`. Update `config.sigma_extra = 1.0` to match archive behavior.

⚠️ **dt discrepancy:** Archive rebinned spikes to dt=1ms before calcium simulation. Current pipeline uses dt=0.1ms end-to-end. This changes the number of time steps T by 10×.

---

## Archive Bug Notes

The archived HPC notebooks (2023) contain a known bug in the Poisson input rate:

```python
# BUGGY (archived notebooks):
p_rate = 10000 * nu_ex * C_E   # 10× too high

# CORRECT (current code):
p_rate = 1000 * nu_ex * C_E    # 1000 = Hz → spikes/s conversion
```

The archive's J=8.0 was calibrated **for the buggy 10× drive**. With the correct
formula, J=8.0 gives burst/silence dynamics (CV≈3.86) rather than AI state.
The archive results are internally consistent but use non-standard parameters.
Do **not** use J=8.0 with the current corrected code for N=1250.

---

## Sanity Check Script

```bash
# Quick local test (N=100, ~30 seconds)
python scripts/sanity_check.py --ne 80 --ni 20

# HPC target sanity check (N=1250, ~2 minutes)
python scripts/sanity_check.py --ne 1000 --ni 250 --sim-time 10000

# Production-length run (N=1250, long)
python scripts/sanity_check.py --ne 1000 --ni 250 --sim-time 1000000 --threads 8
```

Outputs saved to `scripts/sanity_n{N}_output/`:
- `raster.png` — spike raster (check: no visible stripes/synchrony)
- `pop_rate.png` — population firing rate over time (check: roughly flat)
- `firing_rate_hist.png` — per-neuron rate distribution (check: unimodal, ~30–60 Hz)
- `isi_distribution.png` — ISI histogram (check: exponential-ish tail)
- `cv_isi.png` — CV(ISI) distribution (check: peak near 0.5–1.0 for N=1250)
- `calcium_traces.png` — sample fluorescence traces
- `adjacency_matrix.png` — true connectivity matrix
- `summary_stats.txt` — numerical summary

**Acceptance criteria for N=1250:**
- Active neurons: 100% (all 1250 neurons fire)
- Mean firing rate: 30–60 Hz
- Mean CV(ISI): ≥ 0.5 (ideally closer to 1.0)
- No obvious synchrony in raster

---

*Last updated: 2026-05-27*
*Network: Brunel 2000 (J. Comput. Neurosci. 8, 183–208)*
