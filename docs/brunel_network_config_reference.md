# Brunel Network Configuration Reference

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
