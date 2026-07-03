# N=1250 wrap-up plan

The pipeline has **two clean stages** (corrected taxonomy):

1. **Estimate** — a ladder of increasing regularization:
   `OLS → Ridge → Elastic Net (L1+L2) → Dale-Elastic-Net (L1+L2+sign constraint)`.
   Dale is a *regularization* (a sign constraint inside the solver), not post-processing.
   Its post-hoc twin `dale_cleanup` is unused.
2. **Detect** — one post-processing step: a **threshold** on |weight| (the operating point).
   `balance` (magnitude) is dropped — magnitude isn't the current goal; |weight| is only a
   detection score.

Measurements, grouped by what they answer:
- **Detection** (does a connection exist? score = |weight|): AUC-ROC, PR / AP, precision, recall, F1.
- **Sign / 3-class** (E / none / I): type-accuracy, macro-F1, confusion matrix.
- **Distribution**: separation / overlap of the estimated-weight histograms per true class.

## Experimental narrative (paper Results structure)

**Stage 1 — Landscape + preprocessing** *(mostly done)*
Sweep lag / τ / T / λ, 5 seeds (error bars), Elastic Net solver, **deconv on vs off overlaid**
on the same curves (sharp-vs-broad, τ-robust-vs-sensitive). Main metrics only (AUC, F1).
Output: the fixed **operating point** (lag ≈ 2 ms, λ ≈ 1e-4, enough T) used by Stages 2–3.

**Stage 2 — Regularization-ladder effect** *(partly done)*
At the fixed operating point, compare `OLS → Ridge → EN → Dale-EN`. Show how each rung changes
the **confusion matrix, distribution, and measurements** (off-diagonal shrinks, E/none/I
histograms separate). Sign/3-class metrics belong here (Dale acts on sign).

**Stage 3 — Decision / threshold** *(to build — the centerpiece)*
Best estimator (Dale-EN): **sweep the threshold**, show precision/recall (FP/TP) as ROC/PR
curves (Swets operating-point choice). Then the key real-data result: an **unsupervised
threshold** (mixture on the distribution / Dale-consistency) lands near the **oracle** (GT-based
max-F1) threshold — so the method is usable without ground truth.

**Stage 4 — Best configs → full panels** *(tools ready)*
6-cell panel + 4 matrix views for the winners.

## Cross-cutting
- One **fixed operating point** for Stages 2–3 (change one thing at a time).
- **Statistics:** 5-seed error bars; paired test (Wilcoxon) across seeds for key claims
  ("Dale-EN > EN", "deconv on > off").
- **GT-free model selection** is the scientific centerpiece (Stage 3): validate that an
  unsupervised criterion picks the same settings the oracle would.

## Then N=10000
GPU env → CPU-vs-GPU correctness check at N=1250 → prep 10 000-neuron datasets → repeat the
sweeps on GPU. Compute-heavy scaling half.
