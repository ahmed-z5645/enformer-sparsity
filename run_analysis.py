"""Driver for Steps 2-6. Usage: python run_analysis.py [n_windows] [batch_size]"""
import sys
import json
import numpy as np
import torch
torch.set_grad_enabled(False)

from enformer_capacity import (
    log, pick_device, load_enformer, sample_hg38_windows, encode_windows,
    build_enformer_spec, collect_activations, make_figures, write_summary,
)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 1
FASTA = "data/hg38.fa"

device = pick_device("mps")
log(f"device={device}  n_windows={N}  batch={BATCH}")

windows = sample_hg38_windows(FASTA, n_windows=N, seed=0)
encoded = encode_windows(windows)
log(f"encoded {len(encoded)} windows, each {tuple(encoded[0].shape)}")

model = load_enformer()
spec = build_enformer_spec(model)
log(f"spec: {spec.name}  layers={spec.n_layers} heads={spec.n_heads}")

stats = collect_activations(model, encoded, spec, device, batch_size=BATCH)

# sanity gates before drawing anything
log("sanity: checking stats are finite and in plausible ranges")
assert np.isfinite(stats["attn_entropy"]).all(), "non-finite entropy"
assert np.isfinite(stats["attn_mass_frac"]).all(), "non-finite mass frac"
assert np.isfinite(stats["mlp_abs_mean"]).all(), "non-finite mlp act"
assert np.isfinite(stats["resid_norm"]).all(), "non-finite resid norm"
maxent = np.log(stats["mlp_abs_mean"].shape[1])  # not the attn bound; just info
attn_maxent = np.log(1536)
log(f"  entropy range [{stats['attn_entropy'].min():.3f}, {stats['attn_entropy'].max():.3f}]"
    f"  (max possible ~{attn_maxent:.3f} nats)")
log(f"  mass-frac range [{stats['attn_mass_frac'].min():.3f}, {stats['attn_mass_frac'].max():.3f}]")
log(f"  resid-norm range [{stats['resid_norm'].min():.1f}, {stats['resid_norm'].max():.1f}]")
assert (stats["attn_entropy"] <= attn_maxent + 1e-3).all(), "entropy exceeds theoretical max"
assert ((stats["attn_mass_frac"] > 0) & (stats["attn_mass_frac"] <= 1)).all(), "frac out of (0,1]"

fig_meta = make_figures(stats, outdir="results")
write_summary(stats, fig_meta, outdir="results")

np.savez("results/stats.npz", **{k: v for k, v in stats.items()
                                 if isinstance(v, np.ndarray)})
log("saved figures + results/summary.txt + results/stats.npz")
