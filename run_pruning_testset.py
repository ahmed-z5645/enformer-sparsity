"""Overnight pruning experiment on the OFFICIAL held-out Enformer test intervals.

Upgrade over run_pruning.py: evaluates on the real test-set regions (Genentech/enformer-data
intervals) instead of random hg38 windows, with a finer/larger sweep, and adds a structured
(head / channel) pruning comparison alongside unstructured magnitude pruning.

Scoring is still fidelity-to-full-model (Pearson r + nMSE of pruned vs unpruned predictions),
because the experimental target tracks are gated behind a requester-pays bucket (no labels
locally). Writes everything to results_pruning/, saving incrementally so a late failure cannot
lose the earlier phases.

Usage: python run_pruning_testset.py [n_test]   (default 256)
"""
import sys
import numpy as np
import torch
torch.set_grad_enabled(False)

import enformer_capacity as ec
import enformer_pruning as ep

N_TEST = int(sys.argv[1]) if len(sys.argv) > 1 else 256
OUT = sys.argv[2] if len(sys.argv) > 2 else "results_pruning"
GRID = tuple(round(x, 2) for x in np.arange(0.0, 0.951, 0.05))   # 0.00 .. 0.95 step 0.05

device = ec.pick_device("mps")
ec.log(f"OVERNIGHT pruning on official test set | device={device} n_test={N_TEST} grid={GRID}")

model = ec.load_enformer()
windows = ep.load_test_windows("data/hg38.fa", "data/human_intervals.tsv", "test",
                               n=N_TEST, seed=0)
encoded = ec.encode_windows(windows)
ec.log(f"encoded {len(encoded)} test windows, each {tuple(encoded[0].shape)}")

ec.log("computing reference predictions (unpruned model)")
ref_pred = ep.predict_all(model, encoded, device)

# ---- Phase 1: unstructured magnitude pruning (attention / mlp / both) ----
ec.log("PHASE 1: unstructured magnitude pruning")
uns = ep.run_sparsity_sweep(model, encoded, device, ref_pred,
                            levels=GRID, targets=("attention", "mlp", "both"),
                            layermap_level=0.5)
uns["eval_set"] = f"official test intervals (n={len(encoded)})"
ep.make_pruning_figures(uns, outdir=OUT)
ep.write_pruning_summary(uns, outdir=OUT)
ep.save_results(uns, outdir=OUT, fname="pruning_metrics.json")
ec.log("PHASE 1 done + saved")

# ---- Phase 2: structured pruning (whole heads / channels) ----
ec.log("PHASE 2: structured head / channel pruning")
struct = ep.run_structured_sweep(model, encoded, device, ref_pred, fracs=GRID)
struct["eval_set"] = uns["eval_set"]
ep.save_results(struct, outdir=OUT, fname="structured_metrics.json")

# combined comparison figure: unstructured vs structured, per machine
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _series(res, tgt, key):
    return res["realized_sparsity"][tgt], [m[key] for m in res["metrics"][tgt]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, (mach, uns_t, st_t) in zip(
        axes, [("Attention", "attention", "attention_heads"),
               ("MLP", "mlp", "mlp_channels")]):
    x1, y1 = _series(uns, uns_t, "pearson_human")
    x2, y2 = _series(struct, st_t, "pearson_human")
    ax.plot(x1, y1, "o-", label=f"unstructured (weights)")
    ax.plot(x2, y2, "s--", label=f"structured ({'heads' if mach=='Attention' else 'channels'})")
    ax.set_title(mach); ax.set_xlabel("realized weight sparsity"); ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
axes[0].set_ylabel("Pearson r vs full model")
fig.suptitle("Unstructured vs structured magnitude pruning (human head)")
fig.tight_layout(); fig.savefig(f"{OUT}/04_structured_vs_unstructured.png", dpi=150); plt.close(fig)

# structured summary table
lines = ["=" * 72, "STRUCTURED PRUNING (whole heads / channels) vs FIDELITY",
         f"eval: {struct['eval_set']}", "=" * 72]
for tgt in struct["targets"]:
    lines.append(f"\n[{tgt}]  realized-sparsity -> Pearson(human)/Pearson(mouse)/nMSE(human)")
    for real, m in zip(struct["realized_sparsity"][tgt], struct["metrics"][tgt]):
        lines.append(f"    s={real:5.3f}  r_h={m['pearson_human']:.4f}  "
                     f"r_m={m['pearson_mouse']:.4f}  nmse_h={m['nmse_human']:.3e}")
with open(f"{OUT}/summary_structured.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
ec.log("PHASE 2 done + saved. ALL COMPLETE.")
