"""Enformer capacity-utilization analysis (measurement phase).

Modular functions for capturing intermediate activations / attention weights from a
pretrained Enformer (lucidrains/enformer-pytorch) and producing capacity-analysis figures.
Structured so the same collection + figure code can later be pointed at Borzoi via a
different ModelSpec.

fp32 throughout. Hooks only -- the model architecture is never modified.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# Device / setup
# --------------------------------------------------------------------------------------
SEQ_LEN = 196_608
HUMAN_TRACKS = 5313
MOUSE_TRACKS = 1643
TARGET_BINS = 896


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_device(prefer: str = "mps") -> torch.device:
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_enformer(pretrained: str = "EleutherAI/enformer-official-rough"):
    """Load pretrained Enformer in fp32 eval mode. Returns the model on CPU."""
    from enformer_pytorch import Enformer

    log(f"loading Enformer weights: {pretrained} (downloads ~1GB on first run)")
    model = Enformer.from_pretrained(pretrained)
    model = model.float().eval()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"loaded. parameters: {n_params/1e6:.1f}M")
    return model


# --------------------------------------------------------------------------------------
# Step 1: sanity check
# --------------------------------------------------------------------------------------
def random_onehot(batch: int = 1, seq_len: int = SEQ_LEN, seed: int = 0) -> torch.Tensor:
    """Random one-hot DNA. ONLY for the Step-1 sanity check -- never for real stats."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 4, (batch, seq_len), generator=g)
    return torch.nn.functional.one_hot(idx, num_classes=4).float()


@torch.no_grad()
def forward_human_mouse(model, x: torch.Tensor, device: torch.device):
    """Run a forward pass, return dict of head -> tensor on CPU."""
    model = model.to(device)
    x = x.to(device)
    out = model(x)  # enformer-pytorch returns dict with 'human'/'mouse' heads
    if isinstance(out, dict):
        res = {k: v.detach().float().cpu() for k, v in out.items()}
    else:  # single-head fallback
        res = {"human": out.detach().float().cpu()}
    return res


@torch.no_grad()
def sanity_check(model, device: torch.device, rtol: float = 5e-5) -> dict:
    """Shapes + CPU/MPS agreement. Returns a report dict. Raises on shape mismatch.

    Agreement is judged on error *relative to peak output* (rtol), not absolute diff:
    fp32 accumulation makes the absolute max-diff (~5e-4) scale with output magnitude,
    while rel-to-peak is the magnitude-independent measure of end-to-end drift
    (~5e-6 human, ~1e-5 mouse; see investigate_divergence.py)."""
    log("Step 1: building random test sequence (batch=1)")
    x = random_onehot(batch=1, seq_len=SEQ_LEN, seed=0)

    log(f"forward pass on {device}")
    out_dev = forward_human_mouse(model, x, device)
    for head, shape in (("human", (1, TARGET_BINS, HUMAN_TRACKS)),
                        ("mouse", (1, TARGET_BINS, MOUSE_TRACKS))):
        if head in out_dev:
            got = tuple(out_dev[head].shape)
            assert got == shape, f"{head} head shape {got} != expected {shape}"
            log(f"  {head} head shape OK: {got}")

    log("forward pass on cpu (for agreement check)")
    out_cpu = forward_human_mouse(model, x, torch.device("cpu"))

    report = {"device": str(device), "rtol": rtol, "heads": {}, "passed": True}
    for head in out_dev:
        if head not in out_cpu:
            continue
        diff = (out_dev[head] - out_cpu[head]).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        rel = max_abs / (out_cpu[head].abs().max().item() + 1e-12)
        ok = rel < rtol
        report["heads"][head] = {"max_abs": max_abs, "mean_abs": mean_abs,
                                 "rel_to_peak": rel, "within_rtol": ok}
        report["passed"] = report["passed"] and ok
        flag = "OK" if ok else "*** EXCEEDS RTOL ***"
        log(f"  {head}: max|cpu-{device.type}|={max_abs:.3e} mean={mean_abs:.3e} "
            f"rel={rel:.3e} {flag}")
    return report


# --------------------------------------------------------------------------------------
# ModelSpec: the only model-specific glue. Borzoi reuse = a new spec + loader.
# --------------------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """Resolves module paths/handles for hooks. Filled in at Step 3 after print(model)."""
    name: str
    n_layers: int
    n_heads: int
    # callables taking the model, returning ordered lists of (layer_idx, module)
    get_attention_modules: Callable
    get_mlp_act_modules: Callable
    get_block_output_modules: Callable
    # optional monkey-patch installer for attention-weight capture; returns an uninstall fn
    install_attn_capture: Optional[Callable] = None


# --------------------------------------------------------------------------------------
# Step 2: real biological sequence (hg38). Synthetic input is never used for stats.
# --------------------------------------------------------------------------------------
def sample_hg38_windows(fasta_path: str, n_windows: int = 64, seq_len: int = SEQ_LEN,
                        max_n_frac: float = 0.01, seed: int = 0):
    """Sample length-correct windows genome-wide from hg38, skipping N-heavy regions.

    Returns list of (chrom, start, seq_str). N-heavy windows (centromeres/telomeres)
    corrupt activity measurement, so they are filtered (see real-sequence requirement).
    """
    from pyfaidx import Fasta

    fa = Fasta(fasta_path)
    main = {f"chr{i}" for i in range(1, 23)} | {"chrX"}
    chroms = [c for c in fa.keys() if c in main and len(fa[c]) >= seq_len]
    lengths = np.array([len(fa[c]) for c in chroms], dtype=np.float64)
    probs = lengths / lengths.sum()
    rng = np.random.default_rng(seed)

    out, attempts, skipped = [], 0, 0
    log(f"sampling {n_windows} hg38 windows of {seq_len:,} bp (max N frac {max_n_frac})")
    while len(out) < n_windows and attempts < n_windows * 100:
        attempts += 1
        c = chroms[rng.choice(len(chroms), p=probs)]
        start = int(rng.integers(0, len(fa[c]) - seq_len))
        s = str(fa[c][start:start + seq_len])
        if s.upper().count("N") / len(s) <= max_n_frac:
            out.append((c, start, s))
            if len(out) % 10 == 0:
                log(f"  collected {len(out)}/{n_windows} (skipped {skipped} N-heavy)")
        else:
            skipped += 1
    if len(out) < n_windows:
        raise RuntimeError(f"only sampled {len(out)}/{n_windows} windows; check fasta")
    log(f"done: {len(out)} windows, {skipped} N-heavy windows skipped")
    return out


def encode_windows(windows):
    """One-hot encode using the package's own encoder (N -> all-zero row)."""
    from enformer_pytorch.data import str_to_one_hot
    return [str_to_one_hot(s).float() for (_, _, s) in windows]


# --------------------------------------------------------------------------------------
# Step 3: Enformer ModelSpec + attention-weight capture (monkey-patch).
# --------------------------------------------------------------------------------------
def _enformer_attn_modules(model):
    # transformer block i: blk[0].fn[1] is the Attention module
    return [(i, blk[0].fn[1]) for i, blk in enumerate(model.transformer)]


def _enformer_mlp_act_modules(model):
    # blk[1].fn[3] is the ReLU after Linear(dim -> 2*dim); its output is the 3072-d intermediate
    return [(i, blk[1].fn[3]) for i, blk in enumerate(model.transformer)]


def _enformer_block_modules(model):
    # whole block output = residual stream at that layer boundary
    return [(i, blk) for i, blk in enumerate(model.transformer)]


def _install_enformer_attn_capture(model):
    """Per-instance monkey-patch of Attention.forward to stash post-softmax weights.

    enformer-pytorch's Attention computes softmax internally and does not return it, so we
    faithfully re-run its forward (source pinned at enformer-pytorch 0.8.11) and set
    `module._cap_attn`. Returns an uninstall() that restores the original bound methods.
    """
    import types
    from einops import rearrange
    from torch import einsum
    from enformer_pytorch.modeling_enformer import get_positional_embed, relative_shift

    def patched_forward(self, x):
        h = self.heads
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        q = q * self.scale
        n = x.shape[-2]
        content_logits = einsum("b h i d, b h j d -> b h i j", q + self.rel_content_bias, k)
        positions = get_positional_embed(n, self.num_rel_pos_features, x.device,
                                         use_tf_gamma=self.use_tf_gamma,
                                         dtype=self.to_rel_k.weight.dtype)
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)
        rel_k = rearrange(rel_k, "n (h d) -> h n d", h=h)
        rel_logits = einsum("b h i d, h j d -> b h i j", q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)
        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        self._cap_attn = attn.detach()          # <-- capture (pre-dropout; dropout=0 in eval)
        attn = self.attn_dropout(attn)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    originals = []
    for _, mod in _enformer_attn_modules(model):
        originals.append((mod, mod.forward))
        mod.forward = types.MethodType(patched_forward, mod)

    def uninstall():
        for mod, fn in originals:
            mod.forward = fn
            if hasattr(mod, "_cap_attn"):
                del mod._cap_attn
    return uninstall


def build_enformer_spec(model) -> ModelSpec:
    return ModelSpec(
        name="enformer",
        n_layers=len(model.transformer),
        n_heads=model.transformer[0][0].fn[1].heads,
        get_attention_modules=_enformer_attn_modules,
        get_mlp_act_modules=_enformer_mlp_act_modules,
        get_block_output_modules=_enformer_block_modules,
        install_attn_capture=_install_enformer_attn_capture,
    )


# --------------------------------------------------------------------------------------
# Step 4: activity collection (model-agnostic given a ModelSpec). Bounded memory: all
# reductions happen per forward; no full attention matrices are retained.
# --------------------------------------------------------------------------------------
def _reduce_attention(attn: torch.Tensor, mass: float = 0.90):
    """attn: (b,h,i,j) softmax over j. Returns per-head (entropy, frac-for-90%-mass)."""
    a = attn.double()  # reduce on cpu fp64 for stable stats
    ent = -(a * a.clamp_min(1e-12).log()).sum(dim=-1)          # (b,h,i)
    ent_h = ent.mean(dim=(0, 2))                               # (h,)
    sorted_a, _ = a.sort(dim=-1, descending=True)
    cs = sorted_a.cumsum(dim=-1)
    k = (cs < mass).sum(dim=-1) + 1                            # #weights to reach >= mass
    frac_h = (k.double() / a.shape[-1]).mean(dim=(0, 2))       # (h,)
    return ent_h.float(), frac_h.float()


@torch.no_grad()
def collect_activations(model, encoded_seqs, spec: ModelSpec, device: torch.device,
                        batch_size: int = 1, mass: float = 0.90):
    """Run sequences with hooks, return dict of aggregated capacity stats (numpy arrays)."""
    model = model.to(device).eval()
    L, H = spec.n_layers, spec.n_heads

    cache = {"mlp": {}, "resid": {}}
    handles = []
    for i, mod in spec.get_mlp_act_modules(model):
        handles.append(mod.register_forward_hook(
            lambda m, inp, out, i=i: cache["mlp"].__setitem__(i, out.detach())))
    for i, mod in spec.get_block_output_modules(model):
        handles.append(mod.register_forward_hook(
            lambda m, inp, out, i=i: cache["resid"].__setitem__(
                i, (out[0] if isinstance(out, tuple) else out).detach())))
    uninstall_attn = spec.install_attn_capture(model)
    attn_mods = dict(spec.get_attention_modules(model))

    # accumulators
    ent_sum = np.zeros((L, H)); frac_sum = np.zeros((L, H)); seq_count = 0
    mlp_abs_sum = None; mlp_pos_count = 0
    resid_sum = np.zeros(L)

    n = len(encoded_seqs)
    log(f"collecting on {n} sequences, batch_size={batch_size}, device={device}")
    for b0 in range(0, n, batch_size):
        batch = encoded_seqs[b0:b0 + batch_size]
        x = torch.stack(batch, dim=0).to(device)        # (B, seq_len, 4)
        B = x.shape[0]
        _ = model(x)

        for i in range(L):
            attn = attn_mods[i]._cap_attn.to("cpu")     # (B,H,n,n) -> cpu for stable reduce
            ent_h, frac_h = _reduce_attention(attn, mass)
            ent_sum[i] += ent_h.numpy() * B
            frac_sum[i] += frac_h.numpy() * B

            relu = cache["mlp"][i].float()              # (B,n,3072)
            if mlp_abs_sum is None:
                mlp_abs_sum = [np.zeros(relu.shape[-1]) for _ in range(L)]
            mlp_abs_sum[i] += relu.abs().sum(dim=(0, 1)).to("cpu").numpy()

            res = cache["resid"][i].float()             # (B,n,d)
            resid_sum[i] += res.norm(dim=-1).mean().item() * B
        mlp_pos_count += B * cache["mlp"][0].shape[1]
        seq_count += B
        log(f"  {min(b0 + B, n)}/{n} sequences done")

    for h in handles:
        h.remove()
    uninstall_attn()

    stats = {
        "name": spec.name, "n_layers": L, "n_heads": H, "n_seqs": seq_count, "mass": mass,
        "attn_entropy": ent_sum / seq_count,                       # (L,H)
        "attn_mass_frac": frac_sum / seq_count,                    # (L,H)
        "mlp_abs_mean": np.stack([s / mlp_pos_count for s in mlp_abs_sum]),  # (L,3072)
        "resid_norm": resid_sum / seq_count,                       # (L,)
    }
    return stats


# --------------------------------------------------------------------------------------
# Step 5: figures (pure function of stats -> PNGs). Model-agnostic.
# --------------------------------------------------------------------------------------
def make_figures(stats: dict, outdir: str = "results", dead_thresh: float = 1e-6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    L, H = stats["n_layers"], stats["n_heads"]
    name = stats["name"]

    # 1. attention entropy heatmap
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(stats["attn_entropy"], aspect="auto", cmap="viridis")
    ax.set_xlabel("head"); ax.set_ylabel("layer"); ax.set_title(f"{name}: attention entropy (nats)")
    ax.set_xticks(range(H)); ax.set_yticks(range(L))
    fig.colorbar(im, ax=ax, label="mean entropy")
    fig.tight_layout(); fig.savefig(f"{outdir}/01_attention_entropy.png", dpi=150); plt.close(fig)

    # 2. per-layer dead-channel histograms. ReLU activations are bimodal (point mass at
    # exactly 0), so an absolute near-zero cutoff is used rather than a percentile.
    thr = dead_thresh
    ncol = 4; nrow = int(np.ceil(L / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow))
    axes = np.array(axes).flatten()
    dead_per_layer = []
    for i in range(L):
        v = stats["mlp_abs_mean"][i]
        dead = float((v < thr).mean() * 100)
        dead_per_layer.append(dead)
        ax = axes[i]
        ax.hist(np.log10(v + 1e-12), bins=40, color="steelblue")
        ax.axvline(np.log10(thr), color="crimson", ls="--", lw=1)
        ax.set_title(f"layer {i}: {dead:.1f}% dead", fontsize=9)
        ax.set_xlabel("log10 mean|act|", fontsize=8)
    for j in range(L, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{name}: per-layer MLP channel activity "
                 f"(dead = mean|act| < {thr:.0e}; left spike = exactly-zero channels)")
    fig.tight_layout(); fig.savefig(f"{outdir}/02_dead_channels.png", dpi=150); plt.close(fig)

    # 3. residual-stream norm by layer
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(L), stats["resid_norm"], marker="o")
    ax.set_xlabel("layer boundary"); ax.set_ylabel("mean L2 norm")
    ax.set_title(f"{name}: residual-stream norm by layer"); ax.set_xticks(range(L))
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/03_residual_norm.png", dpi=150); plt.close(fig)

    # 4. attention sparsity heatmap (fraction of weights for 90% mass)
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(stats["attn_mass_frac"], aspect="auto", cmap="magma")
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_title(f"{name}: frac of weights for {int(stats['mass']*100)}% attn mass")
    ax.set_xticks(range(H)); ax.set_yticks(range(L))
    fig.colorbar(im, ax=ax, label="fraction (low=concentrated)")
    fig.tight_layout(); fig.savefig(f"{outdir}/04_attention_sparsity.png", dpi=150); plt.close(fig)

    return {"dead_threshold": float(thr), "dead_per_layer": dead_per_layer}


# --------------------------------------------------------------------------------------
# Step 6: interpretation writeup -> stdout + results/summary.txt
# --------------------------------------------------------------------------------------
def write_summary(stats: dict, fig_meta: dict, outdir: str = "results"):
    L, H = stats["n_layers"], stats["n_heads"]
    ent = stats["attn_entropy"]; frac = stats["attn_mass_frac"]
    dead = np.array(fig_meta["dead_per_layer"]); resid = stats["resid_norm"]
    lines = []
    def emit(s=""):
        lines.append(s); print(s, flush=True)

    emit("=" * 70)
    emit(f"CAPACITY-UTILIZATION SUMMARY: {stats['name']}  "
         f"(n_seqs={stats['n_seqs']}, {L} layers x {H} heads)")
    emit("=" * 70)

    emit("\n[1] Attention entropy + [4] diffuseness (prune candidates)")
    ent_thr = np.percentile(ent, 75); frac_thr = np.percentile(frac, 75)
    cand = [(l, h) for l in range(L) for h in range(H)
            if ent[l, h] >= ent_thr and frac[l, h] >= frac_thr]
    emit(f"  high-entropy AND diffuse heads (top-quartile both): {len(cand)} heads")
    for l, h in cand:
        emit(f"    L{l:>2} H{h}: entropy={ent[l,h]:.3f} nats, "
             f"{int(stats['mass']*100)}%-mass frac={frac[l,h]:.3f}")
    lo = np.unravel_index(np.argmin(frac), frac.shape)
    emit(f"  most concentrated/specialized head (likely keep): "
         f"L{lo[0]} H{lo[1]} frac={frac[lo]:.3f}")

    emit("\n[2] Dead MLP channels per layer "
         f"(threshold = {fig_meta['dead_threshold']:.3e})")
    for l in range(L):
        emit(f"    layer {l:>2}: {dead[l]:5.1f}% dead")
    emit(f"  highest dead-channel layers: "
         f"{', '.join(f'L{l}({dead[l]:.0f}%)' for l in np.argsort(dead)[::-1][:3])}")

    emit("\n[3] Residual-stream norm by layer")
    emit(f"    range {resid.min():.1f} (L{resid.argmin()}) .. "
         f"{resid.max():.1f} (L{resid.argmax()})")
    growth = "grows with depth" if resid[-1] > resid[0] else "shrinks with depth"
    emit(f"    trend: {growth} ({resid[0]:.1f} -> {resid[-1]:.1f})")

    emit("\n[non-obvious axes]")
    emit(f"  - mean dead-channel rate {dead.mean():.1f}% suggests "
         f"{'width (MLP) pruning is promising' if dead.mean() > 10 else 'MLP width is well-used'}")
    layer_ent = ent.mean(axis=1)
    emit(f"  - layers with most diffuse attention overall: "
         f"{', '.join(f'L{l}' for l in np.argsort(layer_ent)[::-1][:3])} "
         f"(whole-layer attention-pruning candidates)")
    emit("=" * 70)

    with open(f"{outdir}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
