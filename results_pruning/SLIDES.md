# Magnitude Pruning — Slide Notes

*Supervisor 1:1. Finding-first, terse. Figures do the talking. ~9 slides.*
*Each section = one slide. Bullets = on-screen; "Say:" = talking points.*
*Answers the two follow-ups: (1) what dataset, (2) prune attn/MLP by magnitude, effect on accuracy.*

---

## Slide 1 — Title

- **Magnitude pruning of Enformer: how much can we delete before predictions drift?**
- Follow-up to the capacity study — now we *act* on weights, not just measure.
- One-shot pruning, no fine-tuning · `enformer-official-rough` (251M) · eval on the held-out test set.

**Say:** Last time I mapped where capacity looked wasted. This time I prune weights with magnitude
pruning and measure the damage on the real test set — and it lines up with the earlier map.

---

## Slide 2 — Dataset (answers Q1)

- Sequence: **hg38**. Evaluation: **256 of the 1,937 official held-out Enformer test intervals**
  (Genentech mirror of the Basenji2 test split), each exactly 196,608 bp.
- These are the **genuine held-out regions** — a proper test set, not random windows.
- **Caveat:** the experimental target *tracks* (the labels) sit in a requester-pays bucket — not
  accessible. We have the sequences and intervals, not the ground-truth signal.

**Say:** I upgraded the eval from random windows to the actual Enformer test intervals. The one
thing we can't get without GCP billing is the experimental labels — which sets up the next slide.

---

## Slide 3 — What "accuracy" means here

- Method: **unstructured L1 global magnitude pruning** — zero the smallest-|weight| weights to a
  target sparsity. Attention, MLP, both; swept 0→95%.
- No labels ⇒ **"accuracy" = fidelity to the full model**: Pearson r + normalized MSE between
  *pruned* and *unpruned* predictions on the test sequences.
- Honest framing: measures *functional degradation vs the full model*, not correctness vs truth.

**Say:** Magnitude pruning is the textbook method — delete the weights nearest zero. With no labels,
I score how faithfully each pruned model reproduces the full model's own outputs. It tells us how
much pruning damages the model, not whether the model was right to begin with.

---

## Slide 4 — Headline finding (money slide)

> **The MLP is far more compressible than attention.**
> - MLP: r ≥ 0.99 up to **80%** sparsity; still **r ≈ 0.97 at 95%**.
> - Attention: below 0.99 by **45%**, below 0.90 by **80%**, bottoms ≈ 0.88.
> - Pruning **both** → attention is the bottleneck (the "both" curve tracks attention).

- Figure: `01_fidelity_vs_sparsity.png`

**Say:** This is the main result. The MLP barely notices losing half its weights — 95% sparsity
still holds r≈0.97. Attention falls apart from ~45%. So one global sparsity setting is wrong; the
two parts have very different budgets.

---

## Slide 5 — The curves

- Figures: `01_fidelity_vs_sparsity.png` (Pearson) + `02_mse_vs_sparsity.png` (error).
- Read the **knee**: MLP knee is out near 80%; attention knee is ~40–45%.
- Mouse head tracks human within ~0.01 — same story on both output heads.

**Say:** Two metrics, same conclusion. The knee is the practical budget — MLP bends late, attention
bends early.

---

## Slide 6 — Where the deletions land (ties to capacity study)

- Figure: `03_sparsity_by_layer.png` (realized per-layer sparsity at 50% global).
- Early-layer MLPs cut hardest: **L0–L2 ≈ 0.61–0.64**, falling to ≈0.45 deep. Attention flat ≈0.45.
- These early MLP layers are the **dead-channel** layers from the capacity study.

**Say:** Because the threshold is global, the layers full of tiny weights lose the most — and those
are the early MLPs, exactly the layers the dead-channel analysis flagged. Two independent methods
agreeing on the same wasted capacity — that's the strongest result in the project.

---

## Slide 7 — Structured vs unstructured pruning

- Figure: `04_structured_vs_unstructured.png` (whole heads/channels vs individual weights).
- **MLP:** unstructured wins everywhere (more freedom in what to drop).
- **Attention:** curves **cross at ~57%** — unstructured better below, structured head-removal
  degrades more gracefully above (plateaus ≈0.91 vs ≈0.88).

**Say:** Structured pruning isn't more *faithful* in the useful range — unstructured has more
freedom. The honest point is on the next slide: structured's advantage is a different axis.

---

## Slide 8 — Why structured still matters (the practical caveat)

- Unstructured sparsity needs **special sparse kernels** to actually run faster — zeros still occupy
  the matrix otherwise.
- Structured removal deletes **whole heads/channels** → smaller matrices → **real speed/memory wins
  on ordinary hardware**, at a modest fidelity cost.
- Recipe: **MLP — prune aggressively** (structured channels for real savings, unstructured for max
  fidelity); **attention — keep light**, prefer structured heads if going aggressive.

**Say:** So "which is better" depends on whether you can exploit unstructured sparsity. If you want
actual speedups without special hardware, structured pruning is the one that pays off.

---

## Slide 9 — Caveats & next steps

- **Fidelity ≠ true accuracy** → top next step: GCP access to the Basenji bucket for ground-truth
  tracks; the loader already uses the official intervals, only the labels are missing.
- **No fine-tuning** → one-shot numbers; fine-tuning would raise every curve.
- 256 of 1,937 test regions (curves already smooth); scale up if a reviewer wants it.

**Say:** The biggest gap is fidelity vs true accuracy — getting the label tracks is the top next
step, and we're one data download away since the eval intervals are already the official ones.
