"""Step 1 investigation: localize CPU vs MPS divergence layer-by-layer.

If divergence grows gradually through the stack it's fp32 accumulation (benign).
A sudden jump at one boundary would indicate a specific problematic op.
Also characterizes the output-layer outliers (are they the largest-magnitude outputs?).
"""
import torch
torch.set_grad_enabled(False)
from enformer_capacity import log, random_onehot
from enformer_pytorch import Enformer

x = random_onehot(batch=1, seq_len=196_608, seed=0)

def probe_points(model):
    """Ordered (name, module) list along the trunk = the residual-stream boundaries."""
    pts = [("stem", model.stem), ("conv_tower", model.conv_tower)]
    for i, blk in enumerate(model.transformer):
        pts.append((f"tf{i:02d}", blk))
    pts += [("crop_final", model.crop_final), ("final_pointwise", model.final_pointwise)]
    for hname, hmod in model.heads.items():
        pts.append((f"head_{hname}", hmod))
    return pts

def run_capture(device_str):
    dev = torch.device(device_str)
    model = Enformer.from_pretrained("EleutherAI/enformer-official-rough").float().eval().to(dev)
    caps = {}
    handles = []
    for name, mod in probe_points(model):
        def mk(nm):
            def hook(m, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                caps[nm] = t.detach().float().cpu()
            return hook
        handles.append(mod.register_forward_hook(mk(name)))
    order = [n for n, _ in probe_points(model)]
    log(f"forward on {device_str}")
    _ = model(x.to(dev))
    for h in handles:
        h.remove()
    return order, caps

order, cpu = run_capture("cpu")
_, mps = run_capture("mps")

log("=" * 72)
log(f"{'probe':<16}{'shape':<22}{'max|d|':>12}{'mean|d|':>12}{'rel_peak':>12}")
log("-" * 72)
for name in order:
    a, b = cpu[name], mps[name]
    d = (a - b).abs()
    peak = a.abs().max().item() + 1e-12
    log(f"{name:<16}{str(tuple(a.shape)):<22}{d.max().item():>12.3e}"
        f"{d.mean().item():>12.3e}{d.max().item()/peak:>12.3e}")

# Characterize final human-head outliers: are the worst-diff elements the largest outputs?
log("=" * 72)
a = cpu["head_human"]; b = mps["head_human"]
d = (a - b).abs().flatten()
mag = a.abs().flatten()
k = 20
topd = torch.topk(d, k).indices
log(f"human head: top-{k} |diff| elements -> their output magnitude vs global max output")
gmax = mag.max().item()
log(f"  global max |output| = {gmax:.3f}")
log(f"  mean |output| at the {k} worst-diff cells = {mag[topd].mean().item():.3f} "
    f"(= {100*mag[topd].mean().item()/gmax:.1f}% of global max)")
# correlation between |diff| and |magnitude| (are big errors where big outputs are?)
corr = torch.corrcoef(torch.stack([d, mag]))[0, 1].item()
log(f"  corr(|diff|, |output|) across all {d.numel()} cells = {corr:.3f}")
