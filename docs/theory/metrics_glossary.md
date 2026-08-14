# Evaluation metrics glossary: confusion-matrix building blocks, recall/precision, ROC vs PR, and class imbalance

Reference notes for the report's methods/evaluation section, written up from a
working-session Q&A on 2026-08-14 about why `fig_R4`'s headline numbers
(correlation, excitatory recall/precision) and `fig_R5_conf`'s confusion
matrices can look like they disagree, and what ROC-AUC vs PR-AUC each
actually say about this project's results. See
`docs/experiments/shared_input_findings.md` for the mechanism behind *why*
excitatory recall stays low at large N — this doc is about how to *measure
and report* that fact, not why it happens.

## 1. The four building blocks

For any yes/no question about one edge — the running example here is "is this
pair excitatory?" — every prediction falls into exactly one of four buckets:

| | predicted **excitatory** | predicted **not excitatory** |
|---|---|---|
| **really excitatory** | TP (true positive) | FN (false negative) |
| **really not excitatory** | FP (false positive) | TN (true negative) |

- **TP** — a real excitatory synapse, correctly caught.
- **FN** — a real excitatory synapse, missed (called "none" or "I"). In
  `fig_R5_conf`, this is the **E → none** cell (plus the near-zero E → I
  cell).
- **FP** — a real "none" (or "I") pair, wrongly called excitatory. This is
  the **none → E** (and I → E) cells.
- **TN** — a real "none" pair, correctly left alone.

Common confusion (came up directly in the 2026-08-14 session): a real
excitatory edge predicted as "none" is a **false negative**, not a false
positive. "False positive" is the opposite direction — a real negative
mislabeled positive.

## 2. Recall and precision

    recall    = TP / (TP + FN)
    precision = TP / (TP + FP)

- **Recall** — "of all the *real* excitatory synapses, what fraction did we
  actually find?" Its denominator (TP + FN) is *all real positives*,
  regardless of what we predicted. A recall-only readout says nothing about
  false positives — a classifier that labels every pair excitatory gets
  perfect recall and useless precision.
- **Precision** — "of everything we *labeled* excitatory, what fraction was
  right?" Its denominator (TP + FP) is *everything we predicted positive*,
  regardless of what's actually true. A precision-only readout says nothing
  about false negatives — a classifier that only calls the single most
  obvious edge excitatory can get perfect precision and useless recall.

Neither number alone describes the FN/FP trade-off completely; both together
do. `fig_R4`/`fig_R5` already plot both as separate rows for exactly this
reason. Neither one involves **TN** at all — how well "none" pairs are
correctly left alone (already ~92–97% throughout this project's confusion
matrices) is invisible to both recall and precision. That's usually fine here
since it's not the bottleneck, but it's worth remembering it's not being
measured by either number.

**Why a big headline number and a bad-looking confusion matrix can both be
"correct" at once:** `fig_R4`'s ~0.8 correlation is an aggregate over *every*
pair (E, I, and the huge "none" majority combined) — it stays high even when
excitatory-specific recall is weak, because most pairs (the easy "none"
class) dominate the sum. A single recall/precision *value* is also always
tied to one specific (N, recording length) — comparing N=1250's long-recording
plateau (~0.77 recall) to N=12500's confusion matrix (~0.50 recall) is
comparing two different operating points, not two disagreeing metrics. See
the 2026-08-14 notebook entry for the actual numbers behind this.

## 3. Decision thresholds, ROC, and PR curves

Every one of the four numbers above depends on a **decision threshold** —
where along the inferred-weight axis do we draw the line between "predict
connected" and "predict unconnected"? Sweep that threshold from strict
("call almost nothing positive") to loose ("call almost everything
positive") and trace out two standard curves:

**ROC curve** — plots

    TPR (= recall) = TP / (TP + FN)     [y-axis]
    FPR            = FP / (FP + TN)     [x-axis]

**PR curve** — plots

    precision = TP / (TP + FP)   [y-axis]
    recall    = TP / (TP + FN)   [x-axis]

AUC (area under curve) summarizes the whole sweep in one number instead of
picking one threshold. `roc_auc` and `pr_ap` (average precision = the area
under the PR curve) are both already computed by `scripts/analyze_run.py`
(`score()`) for every run in this project — `pr_ap` was sitting unused in the
metrics CSVs until 2026-08-14, when it was added to `fig_R4`/`fig_R5` as a
plotted row.

## 4. Why they disagree under class imbalance — and this project IS imbalanced

The key difference is what each curve's "other" axis is normalized by:

- **FPR's denominator is (FP + TN) — all real negatives.** In this project,
  "none" pairs outnumber excitatory pairs roughly 10:1 or more. A large
  *absolute* number of false positives is still a tiny *fraction* of that
  huge negative pool, so FPR stays low and **ROC-AUC looks optimistic.**
- **Precision's denominator is (TP + FP) — everything actually predicted
  positive**, a much smaller set, not diluted by the huge negative class.
  The same false positives show up as a much bigger hit to precision, and
  **PR-AUC tells the harsher, more honest story.**

Measured effect (N=12500, longest recording, OLS, R4 ladder): correlation
reaches ~0.86, but PR-AUC (`pr_ap`) only reaches ~0.67 at the same point —
same run, same data, a ~0.19 gap purely from which metric is asked. This is
the quantified version of "the confusion matrix looks worse than the
headline curve" — it's not a bug in either figure, it's what class imbalance
does to ROC-style metrics specifically.

**Report-writing takeaway:** lead with PR-AUC (or recall+precision together)
when characterizing excitatory detection quality, not ROC-AUC alone — ROC-AUC
is the number most likely to overstate performance here, precisely because
"none" is the majority class by a wide margin.
