# Methods Overview

All methods at lag = 1.5 ms on the real preprocessed feed (N=100). **GT=Y**
means the method used ground truth to *produce* the estimate (oracle ceilings
only); all others are fully unsupervised. `daleianity` and `overlap` need NO
ground truth to compute — they are the knobs tunable on real data.
Regenerate: `python scripts/methods_overview.py`.

Directions: detection = f1/precision/recall/auc · type = type_acc ·
magnitude = spearman/pearson/ei · unsupervised-quality = daleianity/overlap
(lower overlap = better).

| method | kind | GT | spearman | pearson | auc | f1 | precision | recall | type_acc | ei | dale | overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OLS | baseline | · | 0.417 | 0.353 | 0.870 | 0.300 | 0.182 | 0.857 | 0.830 | 0.291 | 0.536 | 0.145 |
| EN(L1=3e-3) | regularize | · | 0.424 | 0.339 | 0.847 | 0.424 | 0.293 | 0.765 | 0.820 | 0.213 | 0.686 | 0.112 |
| EN(L1=1e-2) | regularize | · | 0.481 | 0.298 | 0.819 | 0.566 | 0.502 | 0.649 | 0.902 | 0.002 | 0.898 | 0.056 |
| OLS+colnorm | rescale | · | 0.411 | 0.446 | 0.881 | 0.215 | 0.121 | 0.958 | 0.830 | 0.678 | 0.536 | 11.208 |
| OLS+oracle | rescale* | Y | 0.366 | 0.518 | 0.819 | 0.389 | 0.254 | 0.828 | 0.840 | 6.006 | 0.536 | 226.840 |
| EN+colnorm | reg+rescale | · | 0.420 | 0.454 | 0.848 | 0.304 | 0.186 | 0.829 | 0.820 | 0.634 | 0.686 | 14.869 |
| EN+rescale_strongest | reg+rescale | · | 0.421 | 0.422 | 0.847 | 0.401 | 0.270 | 0.779 | 0.810 | 0.352 | 0.686 | 0.162 |
| EN+rescale_balance | reg+rescale | · | 0.404 | 0.553 | 0.826 | 0.482 | 0.357 | 0.741 | 0.810 | 1.631 | 0.686 | 0.960 |
| EN+dale | reg+dale | · | 0.461 | 0.335 | 0.857 | 0.494 | 0.370 | 0.746 | 0.920 | 0.097 | 1.000 | 0.087 |
| EN_daleReg | dale-reg | · | 0.470 | 0.336 | 0.861 | 0.515 | 0.394 | 0.744 | 0.920 | 0.105 | 1.000 | 0.090 |
| EN_daleReg+balance_nz | FULL pipeline | · | 0.470 | 0.585 | 0.860 | 0.490 | 0.363 | 0.757 | 0.920 | 0.921 | 1.000 | 0.335 |
| EN+mixture | reg+mixture | · | 0.460 | 0.338 | 0.851 | 0.424 | 0.293 | 0.765 | 0.820 | 0.213 | 0.712 | 0.109 |
| EN+dale+mixture | reg+combo | · | 0.501 | 0.334 | 0.855 | 0.494 | 0.370 | 0.746 | 0.920 | 0.097 | 1.000 | 0.085 |
| EN+dale+mixture+balance_nz | FULL pipeline | · | 0.501 | 0.540 | 0.856 | 0.490 | 0.362 | 0.756 | 0.920 | 0.612 | 1.000 | 0.243 |
| EN+oracle | reg+rescale* | Y | 0.393 | 0.553 | 0.814 | 0.505 | 0.387 | 0.727 | 0.837 | 5.203 | 0.686 | 197.922 |

*\* = uses ground truth (ceiling, not deployable).*

## How each method works (mechanism · assumptions)

- **OLS** — Plain multivariate AR regression A=C_yx·inv(C_xx) on centred feed.
  - *assumes:* Linear AR(1) model at the chosen lag; off-diagonal A ↔ G.T.
- **EN(L1=3e-3) / EN(L1=1e-2)** — OLS + Elastic-Net penalty (L1 sparsity + L2 stability) via FISTA.
  - *assumes:* Connectivity is sparse; adds shrinkage bias toward 0. Larger L1 = sparser.
- **OLS+colnorm / EN+colnorm** — Divide each neuron's column by its own std (per-neuron normalisation).
  - *assumes:* Every neuron has similar total outgoing strength. Amplifies noise in weak/quiet columns.
- **EN+rescale_strongest** — Guess type from each column's strongest entry; equalise inferred-I group to inferred-E (target ratio 1).
  - *assumes:* Dale (single-sign neurons) + the strongest entry reveals type. Targets ratio 1, not true g.
- **EN+rescale_balance** — As strongest, but set the target ratio to g estimated from network balance g≈(N_E·νE)/(N_I·νI).
  - *assumes:* Network is balanced (E≈I input); firing rates observable; type guess approx. Gives ballpark g (~4–6 vs true 5).
- **EN+dale** — Per column, zero every entry whose sign disagrees with the column's (strongest-entry) type.
  - *assumes:* Dale's law holds strictly (no genuine opposite-sign outgoing edges). Removes wrong-sign false positives.
- **EN_daleReg** — Dale as REGULARIZATION: sign-constrained Elastic Net — FISTA prox projects each column onto its type's sign DURING the fit (not after).
  - *assumes:* Dale's law strict; types from initial EN. Correct-sign entries re-grow to explain variance → beats post-hoc dale on detection.
- **EN+mixture** — Fit a 3-component Gaussian mixture to the off-diagonal weights; zero the component nearest 0 (unconnected).
  - *assumes:* Weights are a 3-class mixture (exc/unconnected/inh) separable by magnitude.
- **EN+dale+mixture / +balance** — Compositions: Dale sign-cleanup, then mixture threshold, then optional balance rescale.
  - *assumes:* Union of the component assumptions above.
- **OLS+oracle / EN+oracle** — Per-class least-squares slope mapping A onto G using TRUE neuron types.
  - *assumes:* USES GROUND TRUTH — ceiling/upper bound only, NOT deployable.

*Scores above use ground truth only to evaluate; methods (except \*) use only the estimate + observable rates.*

