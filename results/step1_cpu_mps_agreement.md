# Step 1 — CPU/MPS Numerical Agreement (Sanity Checkpoint)

**Verdict: PASSED.** The MPS (Apple Silicon) fp32 forward pass of pretrained Enformer
reproduces the CPU reference to within **~1e-5 relative-to-peak error** end-to-end. The
backend is trustworthy for the capacity-utilization measurements that follow.

Model: `EleutherAI/enformer-official-rough`, 251.2M params, fp32, `eval()`, seed-0 random
196,608 bp input. Reproduce with `python step1_sanity.py` and
`python investigate_divergence.py`.

---

## What is being checked, and why on *relative* error

We run the identical forward pass on CPU and on MPS and compare the two output heads. The
checkpoint passes/fails on **error relative to peak output** (`rtol = 5e-5`), *not* on
absolute difference.

This distinction is the whole point of the check. Absolute difference is meaningless as a
backend-agreement criterion because it scales with the magnitude of whatever tensor you
measure — a 0.14 absolute diff in the 1536-d trunk and a 5e-4 diff at the output head are
the *same* relative drift, just at different signal scales. Judging on absolute diff would
either reject benign runs (the trunk) or hide real problems (a small-magnitude head). An
earlier version of this check used `max_abs < 1e-4` and reported FAILED on a known-good
run purely because of this scale mismatch.

---

## Evidence 1 — output heads (`step1_sanity.py`)

| head  | shape           | max\|cpu−mps\| | mean\|cpu−mps\| | **rel-to-peak** | within rtol=5e-5 |
|-------|-----------------|---------------|----------------|-----------------|------------------|
| human | (1, 896, 5313)  | 5.627e-04     | 4.257e-06      | **5.13e-06**    | ✅ |
| mouse | (1, 896, 1643)  | 4.077e-04     | 3.691e-06      | **1.01e-05**    | ✅ |

Both heads agree to ~5–10 parts per million of peak output. The mouse head drifts ~2×
more than human only because its peak output magnitude is smaller, so the same per-element
fp32 error is a larger fraction of the peak — not a head-specific problem.

## Evidence 2 — layer-by-layer localization (`investigate_divergence.py`)

Divergence measured at every residual-stream boundary along the trunk:

| probe            | shape            | max\|d\|   | mean\|d\|  | rel-to-peak |
|------------------|------------------|-----------|-----------|-------------|
| stem             | (1, 768, 98304)  | 2.09e-04  | 7.50e-07  | 9.42e-06    |
| conv_tower       | (1, 1536, 1536)  | 1.42e-01  | 1.32e-04  | 3.78e-05    |
| tf00             | (1, 1536, 1536)  | 1.42e-01  | 1.52e-04  | 3.77e-05    |
| tf01 … tf05      | (1, 1536, 1536)  | ~1.40e-01 | →2.91e-04 | ~3.6e-05    |
| tf06             | (1, 1536, 1536)  | 1.48e-01  | 8.68e-04  | 3.75e-05    |
| tf07 … tf10      | (1, 1536, 1536)  | ~1.36e-01 | →1.22e-03 | ~3.3e-05    |
| crop_final       | (1, 896, 1536)   | 1.36e-01  | 3.38e-04  | 3.43e-05    |
| final_pointwise  | (1, 896, 3072)   | 1.61e-04  | 5.51e-07  | 1.75e-05    |
| head_human       | (1, 896, 5313)   | 5.63e-04  | 4.26e-06  | 5.13e-06    |
| head_mouse       | (1, 896, 1643)   | 4.08e-04  | 3.69e-06  | 1.01e-05    |

**Output-magnitude characterization (human head):** the 20 worst-diff cells sit at 24.0%
of the global max output magnitude, and `corr(|diff|, |output|) = 0.509` across all 4.76M
cells — errors are positively correlated with signal magnitude.

---

## Interpretation

1. **Relative error is bounded everywhere (~1e-5 to 4e-5), and never grows unbounded.**
   The largest relative drift appears once at the `conv_tower` (3.78e-5) and then stays
   *flat — even slightly decreasing* (3.78e-5 → 3.39e-5) — across all 11 transformer
   blocks. There is **no single boundary where the error jumps**, which is the signature
   we were looking for. A sudden jump at one op would have indicated a specific broken /
   unsupported kernel on MPS; we see the opposite.

2. **The pattern is classic fp32 accumulation, not a backend bug.** `mean|d|` grows
   monotonically through the stack (1.3e-4 → 1.2e-3) as rounding differences accumulate
   over depth, while the *relative-to-peak max* stays constant — exactly what
   order-of-summation differences between two fp32 backends produce. The `corr ≈ 0.51`
   between error and output magnitude confirms the largest absolute errors live where the
   largest activations live, rather than being localized blow-ups.

3. **The large mid-trunk absolute diffs (0.14) are an artifact of scale, not severity.**
   They shrink back to 5e-4 at the heads because `crop_final` + `final_pointwise` operate
   at a different signal scale. Relative-to-peak is stable across all of these, confirming
   the absolute number was never the right thing to threshold on.

---

## Why this matters for the capacity study

The entire measurement phase — dead-channel rates, attention entropy, attention sparsity,
residual norms — is collected on **MPS in fp32 with forward hooks**. If the MPS backend
disagreed materially with the reference forward pass, every downstream capacity statistic
would be partly an artifact of the hardware backend rather than a property of the
pretrained model. This checkpoint certifies that is not the case: MPS reproduces the
reference to ppm-level relative error, so the activity distributions we measure are the
model's, and any cross-device or future Borzoi comparison rests on a verified baseline.

### Caveat to carry forward
At the MLP activation sites (post-ReLU, 3072-d), backend disagreement is `mean|d| ~5e-7`
absolute — the *same order* as the dead-channel cutoff (`mean|act| < 1e-6`). This is safe
**because** dead channels are detected from the bimodal point-mass-at-exactly-zero
structure of ReLU outputs, not from borderline values near the threshold; truly dead
channels read ~0 on both backends. It would only become a concern if the dead criterion
were ever changed to a percentile/soft cutoff — flag it then.
