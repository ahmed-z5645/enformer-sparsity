"""Step 1 driver: load Enformer, verify shapes, check CPU vs MPS agreement."""
import json
import torch

from enformer_capacity import log, pick_device, load_enformer, sanity_check

torch.set_grad_enabled(False)

device = pick_device("mps")
log(f"selected device: {device}")

model = load_enformer()
report = sanity_check(model, device, rtol=5e-5)

log("=" * 60)
log("STEP 1 REPORT")
print(json.dumps(report, indent=2), flush=True)
log("PASSED" if report["passed"] else "FAILED -- agreement exceeded rtol")
