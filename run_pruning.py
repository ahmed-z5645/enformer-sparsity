"""Driver for the magnitude-pruning sweep. Usage: python run_pruning.py [n_windows].

Mirrors run_analysis.py. Reuses the hg38 64-window pipeline, computes the full-model
reference predictions, sweeps unstructured L1 magnitude pruning over attention / mlp / both,
and writes figures + summary + JSON to results_pruning/.
"""
import sys
import numpy as np
import torch
torch.set_grad_enabled(False)

import enformer_capacity as ec
import enformer_pruning as ep

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
FASTA = "data/hg38.fa"

device = ec.pick_device("mps")
ec.log(f"device={device}  n_windows={N}")

model = ec.load_enformer()
windows = ec.sample_hg38_windows(FASTA, n_windows=N, seed=0)
encoded = ec.encode_windows(windows)
ec.log(f"encoded {len(encoded)} windows, each {tuple(encoded[0].shape)}")

ec.log("computing reference predictions (unpruned model)")
ref_pred = ep.predict_all(model, encoded, device)

results = ep.run_sparsity_sweep(
    model, encoded, device, ref_pred,
    levels=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    targets=("attention", "mlp", "both"),
    layermap_level=0.5,
)

ep.make_pruning_figures(results, outdir="results_pruning")
ep.write_pruning_summary(results, outdir="results_pruning")
ep.save_results(results, outdir="results_pruning")
ec.log("saved figures + results_pruning/summary.txt + pruning_metrics.json")
