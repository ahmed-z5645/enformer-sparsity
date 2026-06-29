"""Enformer magnitude-pruning experiment.

Follow-up to the capacity-utilization measurement phase. Where that phase only *measured*
under-used capacity (entropy, dead channels) without touching weights, this module actually
**prunes** weights with unstructured L1 magnitude pruning and quantifies the effect.

Two supervisor questions are answered here:
  Q1 "What dataset did you use?"  -> hg38 reference genome, 64 windows of 196,608 bp, seed 0,
     N-heavy regions filtered (the *same* windows as the capacity phase; see sample_hg38_windows
     in enformer_capacity.py). No ground-truth experimental tracks are in the repo.
  Q2 "Prune attentions and MLPs using magnitude pruning. How does that affect accuracy?"

Because there are no ground-truth labels, "accuracy" here means **fidelity to the full model**:
how well the pruned model reproduces the *unpruned* model's predictions on those windows
(Pearson r per track across the 896 bins + normalized MSE). This measures functional
degradation relative to the full model, not accuracy against experimental truth.

fp32 throughout. Helpers (load, sampling, encoding, forward) are reused from enformer_capacity.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

from enformer_capacity import log, forward_human_mouse


# --------------------------------------------------------------------------------------
# Official held-out Enformer test intervals (Genentech/enformer-data mirror). These are the
# real test regions the model was evaluated on -- a proper eval set vs random hg38 windows.
# Each interval is exactly 196,608 bp, so it drops straight into the pipeline.
# --------------------------------------------------------------------------------------
def load_test_windows(fasta_path: str, tsv_path: str = "data/human_intervals.tsv",
                      split: str = "test", n=None, seed: int = 0):
    """Return list of (chrom, start, seq_str) for `split` intervals, sampled to n (seeded)."""
    import csv
    from pyfaidx import Fasta

    rows = []
    with open(tsv_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["split"] == split:
                rows.append((r["chrom"], int(r["start"]), int(r["end"])))
    log(f"{len(rows)} '{split}' intervals in {tsv_path}")
    if n is not None and n < len(rows):
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), size=n, replace=False))]
        log(f"sampled {n} (seed {seed})")

    fa = Fasta(fasta_path)
    out = []
    for chrom, start, end in rows:
        if chrom not in fa:
            continue
        out.append((chrom, start, str(fa[chrom][start:end])))
    log(f"extracted {len(out)} sequences from {fasta_path}")
    return out


# --------------------------------------------------------------------------------------
# Locating the prunable Linear layers (attention vs MLP), per transformer block.
# Resolved by isinstance(nn.Linear) so we never hard-code submodule indices: enformer-pytorch
# block i is (Residual(Sequential(LayerNorm, Attention=fn[1], ...)),
#             Residual(Sequential(LayerNorm, Linear, ..., ReLU=fn[3], Linear, ...)))
# --------------------------------------------------------------------------------------
def _linears_by_layer(model):
    """Return list over transformer blocks of (attention_linears, mlp_linears)."""
    out = []
    for blk in model.transformer:
        attn_mod = blk[0].fn[1]          # the Attention module (to_q/to_k/to_v/to_out live here)
        mlp_seq = blk[1].fn              # the feed-forward Sequential
        attn_lin = [m for m in attn_mod.modules() if isinstance(m, nn.Linear)]
        mlp_lin = [m for m in mlp_seq.modules() if isinstance(m, nn.Linear)]
        out.append((attn_lin, mlp_lin))
    return out


def get_prunable_linears(model, which: str):
    """Flat list of (name, module) Linear layers for which in {'attention','mlp','both'}."""
    by_layer = _linears_by_layer(model)
    named = []
    for li, (attn_lin, mlp_lin) in enumerate(by_layer):
        if which in ("attention", "both"):
            for j, m in enumerate(attn_lin):
                named.append((f"L{li}.attn.{j}", m))
        if which in ("mlp", "both"):
            for j, m in enumerate(mlp_lin):
                named.append((f"L{li}.mlp.{j}", m))
    if not named:
        raise ValueError(f"no prunable linears found for which={which!r}")
    return named


# --------------------------------------------------------------------------------------
# Magnitude pruning (unstructured, global L1) + sparsity bookkeeping.
# --------------------------------------------------------------------------------------
def apply_global_magnitude_pruning(linears, amount: float):
    """Zero the smallest-|weight| weights across the whole group to global `amount` sparsity.

    A single magnitude threshold is shared by all selected layers (the textbook definition of
    global magnitude pruning): the |weight| value at the `amount` quantile over every weight in
    the group, then every weight at or below it is set to 0, in place. Biases are left intact.

    The threshold is computed on CPU because torch's built-in global_unstructured indexes into a
    concatenation of all weights, which hits an out-of-bounds bug on the MPS backend at this
    scale. Zeroing is done in place; restore originals with load_state_dict afterwards.
    """
    if amount <= 0:
        return
    mags = np.concatenate([m.weight.detach().abs().reshape(-1).cpu().numpy() for _, m in linears])
    thr = float(np.quantile(mags, amount))
    for _, m in linears:
        w = m.weight.data
        w.masked_fill_(w.abs() <= thr, 0.0)


def group_sparsity(linears) -> float:
    """Realized fraction of exactly-zero weights across the group (after pruning)."""
    zeros = total = 0
    for _, m in linears:
        w = m.weight
        zeros += int((w == 0).sum().item())
        total += w.numel()
    return zeros / total if total else 0.0


def layer_sparsity_map(model):
    """Per-layer realized weight sparsity for attention and MLP groups (after pruning).

    Returns (attn_sparsity[L], mlp_sparsity[L]) -- used to show *where* magnitude pruning
    removes weights, for comparison with the capacity study's early-MLP / late-attention split.
    """
    by_layer = _linears_by_layer(model)
    L = len(by_layer)
    attn_s = np.zeros(L)
    mlp_s = np.zeros(L)
    for li, (attn_lin, mlp_lin) in enumerate(by_layer):
        attn_s[li] = group_sparsity([("", m) for m in attn_lin])
        mlp_s[li] = group_sparsity([("", m) for m in mlp_lin])
    return attn_s, mlp_s


# --------------------------------------------------------------------------------------
# Prediction + fidelity-to-full-model metrics.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict_all(model, encoded_seqs, device):
    """Forward over all windows (batch=1). Returns {head: np.array (n, 896, tracks)} on CPU."""
    heads = {}
    n = len(encoded_seqs)
    for i, x in enumerate(encoded_seqs):
        out = forward_human_mouse(model, x.unsqueeze(0), device)  # CPU dict
        for k, v in out.items():
            heads.setdefault(k, []).append(v.squeeze(0).numpy())
        if (i + 1) % 8 == 0 or i + 1 == n:
            log(f"    predicted {i + 1}/{n} windows")
    return {k: np.stack(v, axis=0) for k, v in heads.items()}


def _pearson_per_track(pred: np.ndarray, ref: np.ndarray) -> float:
    """Mean Pearson r between pred and ref, computed per track over the 896-bin axis.

    pred/ref: (n_windows, 896, n_tracks). Correlate along bins (axis=1), then average over
    tracks and windows. Tracks with zero variance (constant) are ignored (nan-skipped).
    """
    p = pred.astype(np.float64)
    r = ref.astype(np.float64)
    p = p - p.mean(axis=1, keepdims=True)
    r = r - r.mean(axis=1, keepdims=True)
    num = (p * r).sum(axis=1)
    den = np.sqrt((p * p).sum(axis=1) * (r * r).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / den                                  # (n_windows, n_tracks)
    return float(np.nanmean(corr))


def _norm_mse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Normalized MSE = mean((pred-ref)^2) / mean(ref^2) (relative L2 energy)."""
    num = float(((pred.astype(np.float64) - ref.astype(np.float64)) ** 2).mean())
    den = float((ref.astype(np.float64) ** 2).mean()) + 1e-12
    return num / den


def fidelity_metrics(pred: dict, ref: dict) -> dict:
    """Per-head fidelity of a pruned prediction set vs the full-model reference."""
    out = {}
    for head in ref:
        out[f"pearson_{head}"] = _pearson_per_track(pred[head], ref[head])
        out[f"nmse_{head}"] = _norm_mse(pred[head], ref[head])
    return out


# --------------------------------------------------------------------------------------
# The sweep: for each target group x sparsity level, prune -> predict -> score vs reference.
# Each level prunes from clean weights (no cumulative masks) via an orig_state snapshot.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def run_sparsity_sweep(model, encoded_seqs, device, ref_pred,
                       levels=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                       targets=("attention", "mlp", "both"),
                       layermap_level: float = 0.5):
    """Sweep magnitude pruning. Returns a results dict keyed by target.

    ref_pred: predictions of the *unpruned* model (from predict_all) -- the fidelity reference.
    layermap_level: the global sparsity at which per-layer sparsity maps are captured for fig 3.
    """
    orig_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    levels = list(levels)
    results = {"levels": levels, "targets": list(targets), "layermap_level": layermap_level,
               "metrics": {}, "realized_sparsity": {}, "layermap": {}}

    for tgt in targets:
        log(f"=== target group: {tgt} ===")
        results["metrics"][tgt] = []
        results["realized_sparsity"][tgt] = []
        linears = get_prunable_linears(model, tgt)
        for amount in levels:
            if amount > 0:
                apply_global_magnitude_pruning(linears, amount)
            realized = group_sparsity(linears)

            pred = predict_all(model, encoded_seqs, device)
            m = fidelity_metrics(pred, ref_pred)
            m["target"], m["amount"] = tgt, amount
            results["metrics"][tgt].append(m)
            results["realized_sparsity"][tgt].append(realized)
            log(f"  {tgt} amount={amount:.2f} realized={realized:.3f}  "
                f"pearson_human={m['pearson_human']:.4f} nmse_human={m['nmse_human']:.3e}")

            # capture per-layer sparsity map at the representative level (use 'both')
            if tgt == "both" and abs(amount - layermap_level) < 1e-9:
                attn_s, mlp_s = layer_sparsity_map(model)
                results["layermap"] = {"attention": attn_s.tolist(), "mlp": mlp_s.tolist()}

            if amount > 0:
                model.load_state_dict(orig_state)   # restore clean weights for next level
    return results


# --------------------------------------------------------------------------------------
# Figures + summary (mirror enformer_capacity.make_figures / write_summary).
# --------------------------------------------------------------------------------------
def make_pruning_figures(results: dict, outdir: str = "results_pruning"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    levels = results["levels"]
    targets = results["targets"]
    colors = {"attention": "tab:blue", "mlp": "tab:orange", "both": "tab:green"}

    def series(tgt, key):
        return [m[key] for m in results["metrics"][tgt]]

    # 1. fidelity (Pearson vs full model) vs sparsity
    fig, ax = plt.subplots(figsize=(7, 5))
    for tgt in targets:
        x = results["realized_sparsity"][tgt]
        ax.plot(x, series(tgt, "pearson_human"), marker="o", color=colors[tgt],
                label=f"{tgt} (human)")
        ax.plot(x, series(tgt, "pearson_mouse"), marker="s", ls="--", color=colors[tgt],
                alpha=0.6, label=f"{tgt} (mouse)")
    ax.set_xlabel("realized weight sparsity"); ax.set_ylabel("Pearson r vs full model")
    ax.set_title("Prediction fidelity vs magnitude-pruning sparsity")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/01_fidelity_vs_sparsity.png", dpi=150); plt.close(fig)

    # 2. normalized MSE vs sparsity (log y)
    fig, ax = plt.subplots(figsize=(7, 5))
    for tgt in targets:
        x = results["realized_sparsity"][tgt]
        ax.plot(x, series(tgt, "nmse_human"), marker="o", color=colors[tgt], label=f"{tgt} (human)")
        ax.plot(x, series(tgt, "nmse_mouse"), marker="s", ls="--", color=colors[tgt],
                alpha=0.6, label=f"{tgt} (mouse)")
    ax.set_xlabel("realized weight sparsity"); ax.set_ylabel("normalized MSE vs full model")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_title("Prediction error vs magnitude-pruning sparsity")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/02_mse_vs_sparsity.png", dpi=150); plt.close(fig)

    # 3. where pruning lands: per-layer realized sparsity at the representative global level
    lm = results.get("layermap")
    if lm:
        L = len(lm["attention"])
        x = np.arange(L); w = 0.4
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - w / 2, lm["attention"], w, label="attention", color=colors["attention"])
        ax.bar(x + w / 2, lm["mlp"], w, label="mlp", color=colors["mlp"])
        ax.axhline(results["layermap_level"], color="crimson", ls="--", lw=1,
                   label=f"global target {results['layermap_level']:.0%}")
        ax.set_xlabel("transformer layer"); ax.set_ylabel("realized weight sparsity")
        ax.set_title(f"Where magnitude pruning removes weights "
                     f"(both groups @ {results['layermap_level']:.0%} global)")
        ax.set_xticks(x); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{outdir}/03_sparsity_by_layer.png", dpi=150); plt.close(fig)


def write_pruning_summary(results: dict, outdir: str = "results_pruning"):
    os.makedirs(outdir, exist_ok=True)
    lines = []
    def emit(s=""):
        lines.append(s); print(s, flush=True)

    emit("=" * 72)
    emit("MAGNITUDE-PRUNING vs FIDELITY SUMMARY (Enformer)")
    emit(f"eval: {results.get('eval_set', 'hg38 windows')}; accuracy = fidelity to full model")
    emit("=" * 72)

    for tgt in results["targets"]:
        emit(f"\n[{tgt}]  sparsity -> Pearson(human) / Pearson(mouse) / nMSE(human)")
        for real, m in zip(results["realized_sparsity"][tgt], results["metrics"][tgt]):
            emit(f"    sparsity={real:5.3f}  r_h={m['pearson_human']:.4f}  "
                 f"r_m={m['pearson_mouse']:.4f}  nmse_h={m['nmse_human']:.3e}")
        # headline: sparsity at which human Pearson first drops below thresholds
        for thr in (0.99, 0.95, 0.90):
            crossed = next((real for real, m in zip(results["realized_sparsity"][tgt],
                                                    results["metrics"][tgt])
                            if m["pearson_human"] < thr), None)
            emit(f"    Pearson(human) drops below {thr:.2f} at sparsity "
                 f"{'%.2f' % crossed if crossed is not None else '>max (never)'}")

    if results.get("layermap"):
        lm = results["layermap"]
        emit(f"\n[where pruning lands @ {results['layermap_level']:.0%} global]")
        emit("    layer:  " + " ".join(f"{i:>4}" for i in range(len(lm['attention']))))
        emit("    attn :  " + " ".join(f"{s:4.2f}" for s in lm["attention"]))
        emit("    mlp  :  " + " ".join(f"{s:4.2f}" for s in lm["mlp"]))
    emit("=" * 72)

    with open(f"{outdir}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def save_results(results: dict, outdir: str = "results_pruning", fname: str = "pruning_metrics.json"):
    """Persist sweep results as JSON (human-readable) for later plotting / inspection."""
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/{fname}", "w") as f:
        json.dump(results, f, indent=2)


# --------------------------------------------------------------------------------------
# Structured magnitude pruning: remove whole attention HEADS / MLP CHANNELS (units), ranked
# by weight magnitude, globally. This is the "right tool for attention" comparison and links
# to the capacity study's head/channel candidates. Removing a unit zeroes its full parameter
# slice, so a head/channel contributes exactly nothing afterward.
# --------------------------------------------------------------------------------------
def _attn_head_geometry(model):
    """Return (heads, dim_head) and per-block (to_q,to_k,to_v,to_out) Linears."""
    blocks = []
    heads = dim_head = None
    for blk in model.transformer:
        att = blk[0].fn[1]
        h = att.heads
        inner = att.to_q.weight.shape[0]          # (inner_dim, dim)
        dh = inner // h
        heads, dim_head = h, dh
        # to_out may be wrapped in a Sequential; grab its Linear
        out_lin = next(m for m in att.to_out.modules() if isinstance(m, nn.Linear)) \
            if not isinstance(att.to_out, nn.Linear) else att.to_out
        blocks.append((att.to_q, att.to_k, att.to_v, out_lin))
    return heads, dim_head, blocks


def _head_scores(model):
    """Magnitude score per (layer, head): L2 norm of the head's q/k/v rows + out columns."""
    heads, dh, blocks = _attn_head_geometry(model)
    L = len(blocks)
    scores = np.zeros((L, heads))
    for li, (q, k, v, o) in enumerate(blocks):
        for h in range(heads):
            s = slice(h * dh, (h + 1) * dh)
            n2 = (q.weight.data[s].pow(2).sum() + k.weight.data[s].pow(2).sum()
                  + v.weight.data[s].pow(2).sum() + o.weight.data[:, s].pow(2).sum())
            scores[li, h] = float(n2.sqrt())
    return scores, heads, dh, blocks


def prune_attention_heads(model, frac: float):
    """Zero the smallest-magnitude `frac` fraction of all attention heads (global ranking)."""
    if frac <= 0:
        return 0.0
    scores, heads, dh, blocks = _head_scores(model)
    L = heads_total = scores.size
    k = int(round(frac * heads_total))
    if k <= 0:
        return 0.0
    order = np.argsort(scores, axis=None)            # ascending: smallest magnitude first
    kill = set(order[:k].tolist())
    for li, (q, kk, v, o) in enumerate(blocks):
        for h in range(heads):
            if li * heads + h in kill:
                s = slice(h * dh, (h + 1) * dh)
                for lin in (q, kk, v):
                    lin.weight.data[s] = 0
                    if lin.bias is not None:
                        lin.bias.data[s] = 0
                o.weight.data[:, s] = 0              # remove head's output pathway
    return k / heads_total


def _mlp_channel_linears(model):
    """Per-block (lin_in: dim->inter, lin_out: inter->dim) for the MLP intermediate channels."""
    out = []
    for blk in model.transformer:
        lins = [m for m in blk[1].fn.modules() if isinstance(m, nn.Linear)]
        lin_in = max(lins, key=lambda m: m.weight.shape[0])    # largest out_features = ->inter
        lin_out = max(lins, key=lambda m: m.weight.shape[1])   # largest in_features  = inter->
        out.append((lin_in, lin_out))
    return out


def prune_mlp_channels(model, frac: float):
    """Zero the smallest-magnitude `frac` fraction of all MLP intermediate channels (global)."""
    if frac <= 0:
        return 0.0
    pairs = _mlp_channel_linears(model)
    # score each channel = norm of its in-row + out-col
    scores = []
    for li, (lin_in, lin_out) in enumerate(pairs):
        n2 = lin_in.weight.data.pow(2).sum(dim=1) + lin_out.weight.data.pow(2).sum(dim=0)
        scores.append(n2.sqrt().cpu().numpy())
    flat = np.concatenate(scores)
    k = int(round(frac * flat.size))
    if k <= 0:
        return 0.0
    thr = np.partition(flat, k - 1)[k - 1]
    for li, (lin_in, lin_out) in enumerate(pairs):
        kill = torch.from_numpy(scores[li] <= thr).to(lin_in.weight.device)
        lin_in.weight.data[kill] = 0
        if lin_in.bias is not None:
            lin_in.bias.data[kill] = 0
        lin_out.weight.data[:, kill] = 0
    return k / flat.size


@torch.no_grad()
def run_structured_sweep(model, encoded_seqs, device, ref_pred,
                         fracs=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Structured sweep: remove whole heads / MLP channels by magnitude. Mirrors the
    unstructured results schema (target -> list of fidelity dicts) for combined plotting."""
    orig_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    fracs = list(fracs)
    results = {"levels": fracs, "targets": ["attention_heads", "mlp_channels"],
               "metrics": {}, "realized_sparsity": {}}
    prune_fns = {"attention_heads": prune_attention_heads, "mlp_channels": prune_mlp_channels}

    for tgt, fn in prune_fns.items():
        log(f"=== structured target: {tgt} ===")
        results["metrics"][tgt] = []
        results["realized_sparsity"][tgt] = []
        for amount in fracs:
            realized = fn(model, amount) if amount > 0 else 0.0
            pred = predict_all(model, encoded_seqs, device)
            m = fidelity_metrics(pred, ref_pred)
            m["target"], m["amount"] = tgt, amount
            results["metrics"][tgt].append(m)
            results["realized_sparsity"][tgt].append(realized)
            log(f"  {tgt} frac={amount:.2f} realized={realized:.3f}  "
                f"pearson_human={m['pearson_human']:.4f} nmse_human={m['nmse_human']:.3e}")
            if amount > 0:
                model.load_state_dict(orig_state)
    return results
