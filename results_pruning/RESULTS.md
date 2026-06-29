# Magnitude Pruning vs Fidelity — Results Summary

**Notebook:** `enformer_pruning.ipynb`  •  **Drivers:** `run_pruning_testset.py` (this run), `run_pruning.py`
**Module:** `enformer_pruning.py`  •  **Model:** `EleutherAI/enformer-official-rough` (251.2M), fp32, MPS, no fine-tuning
**Eval set:** **256 official held-out Enformer test intervals** (196,608 bp each), seed 0
**Sweep:** 0→95% sparsity in 5% steps × {attention, MLP, both}, plus structured head/channel pruning

Follow-up to the capacity-utilization phase: there we only *measured* under-used capacity; here we
**prune weights** and quantify the effect on the model's output.

---

## Q1 — What dataset did you use?

The **hg38 human reference genome** for sequence, evaluated on the **official Enformer held-out
test intervals** (the `Genentech/enformer-data` mirror of the Basenji2 test split — 1,937 regions;
we use a seeded **256-region subset**). Each interval is exactly **196,608 bp**, the model's native
input length. These are the genuine test regions the model was held out on, so the evaluation set
is proper, not arbitrary windows.

**Caveat that shapes everything below:** the repo / open mirrors contain the **sequence and the
interval definitions, but not the experimental target tracks** (the labels live in a requester-pays
GCS bucket). So we cannot score against experimental truth — see the next section.

---

## Q2 — Magnitude pruning, and how it affects "accuracy"

**Method.** Unstructured **L1 global magnitude pruning**: rank every weight in the target group by
|value|, zero the smallest fraction (one shared global threshold). Applied to **attention** Linears
(55), **MLP** Linears (22), and **both**. Each level prunes from clean pretrained weights (verified
bit-identical restore). We also run **structured** pruning — removing whole **attention heads** and
**MLP channels** by magnitude — for comparison. No fine-tuning anywhere.

**"Accuracy" = fidelity to the full model.** With no labels, each pruned model is scored by how
faithfully it reproduces the **unpruned** model's predictions on the test sequences:
- **Pearson r** per output track over the 896 bins (Enformer's scoring axis), averaged.
- **Normalized MSE** = mean((pruned − full)²) / mean(full²).

This is *functional degradation relative to the full model* — **not** accuracy against experimental
truth. It cannot say whether the model was right; only how well a pruned model imitates it.

---

## Headline result — the MLP is far more compressible than attention

Unstructured magnitude pruning, human-head Pearson r vs the full model:

| sparsity | attention | MLP | both |
|---|---|---|---|
| 0%  | 1.0000 | 1.0000 | 1.0000 |
| 30% | 0.9971 | 0.9999 | 0.9985 |
| 40% | 0.9906 | 0.9996 | 0.9941 |
| 50% | 0.9719 | 0.9986 | 0.9808 |
| 60% | 0.9421 | 0.9966 | 0.9542 |
| 70% | 0.9140 | 0.9927 | 0.9266 |
| 80% | 0.8919 | 0.9875 | 0.9033 |
| 90% | 0.8822 | 0.9796 | 0.8888 |
| 95% | 0.8811 | 0.9733 | 0.8809 |

**Sparsity at which human-head fidelity first drops below a threshold:**

| | r < 0.99 | r < 0.95 | r < 0.90 |
|---|---|---|---|
| **attention** | 45% | 60% | 80% |
| **MLP**       | 80% | never (0.973 at 95%) | never |
| **both**      | 45% | 65% | 85% |

Mouse-head tracks human within ~0.01. When pruning **both**, attention is the bottleneck — the
"both" curve hugs the attention curve, not the MLP one. Figures: `01_fidelity_vs_sparsity.png`,
`02_mse_vs_sparsity.png`. Full tables: `summary.txt`; raw: `pruning_metrics.json`.

---

## Where the pruning lands (ties to the capacity study)

`03_sparsity_by_layer.png`: at a 50% global budget, magnitude pruning removes the most from the
**early-layer MLPs** — L0–L2 at **0.61–0.64** realized sparsity, falling to ≈0.45 by the deep
layers — while attention is pruned almost uniformly (≈0.44–0.47). Those early MLP layers are
exactly the **dead-channel** layers the capacity study flagged. Two independent methods —
activation-based dead-channel counting and weight-magnitude pruning — point at the same wasted
capacity. This is the strongest cross-validation in the project.

---

## Structured vs unstructured pruning (`04_structured_vs_unstructured.png`)

Removing whole heads / channels by magnitude, vs zeroing individual weights, at matched sparsity:

- **MLP:** unstructured wins **everywhere** (e.g. at 50%: 0.999 vs 0.989; at 90%: 0.980 vs 0.956).
  Expected — unstructured has more freedom in *which* weights to drop.
- **Attention:** the two **cross at ~57% sparsity**. Below it, unstructured is better (more freedom);
  **above ~60%, structured head-removal degrades more gracefully** — it plateaus around r ≈ 0.91 at
  high sparsity while unstructured sinks to ≈0.88. Removing whole low-magnitude heads is a "cleaner"
  cut at aggressive budgets.

**The honest reading:** structured pruning is **not** more faithful in the practical regime. Its
real advantage is orthogonal to these curves — it removes whole units, so it yields **actual
speed/memory savings on ordinary hardware**, whereas unstructured sparsity needs specialized sparse
kernels to realize any gain. So "which is better" depends on *whether you can exploit unstructured
sparsity*: if not, structured pruning buys real savings at a modest fidelity cost.

---

## Interpretation / bottom line for the supervisor

1. **MLP weights are largely redundant; attention weights are not.** Half the MLP weights delete
   essentially for free (r = 0.999 at 50%), and even **95% MLP sparsity holds r ≈ 0.97**. Attention
   degrades from ~45% and bottoms near 0.88. ⇒ A single global sparsity setting is wrong: the two
   machines have very different budgets.
2. **This corroborates the capacity study directly on the MLP side** — early-MLP layers (the
   dead-channel layers) are exactly where global magnitude pruning concentrates its cuts.
3. **Practical recipe:** prune the MLP aggressively (structured channel removal for real savings, or
   unstructured for max fidelity); keep attention pruning light, and prefer structured head removal
   if you must go aggressive there.

---

## Caveats & next steps

- **Fidelity ≠ true accuracy.** All numbers are agreement with the full model, not experimental
  data. **Next:** with GCP access to the Basenji `basenji_barnyard` bucket, pull the target tracks
  and recompute these curves as Pearson vs ground truth (the loader already uses the official
  intervals, so only the target side is missing).
- **No fine-tuning** — these are one-shot prune-and-evaluate numbers; fine-tuning would lift every
  curve.
- **256 of 1,937 test regions** — curves are smooth; scaling to the full test set is expected to
  move numbers only marginally.

## Artifacts
`01–04_*.png` · `summary.txt` + `summary_structured.txt` · `pruning_metrics.json` +
`structured_metrics.json` · `overnight.log` · pipeline in `enformer_pruning.py`,
driver `run_pruning_testset.py`.
