"""Compat alias — moved to code.perception.detector.model (RF-1).

See docs/refactor_plan.md. Old import paths (`import code.nx6_heatmap_model`,
`from code.nx6_heatmap_model import HeatmapDetector, ...`) keep working
unchanged: this file replaces itself in sys.modules with the real module
object, so both paths refer to the exact same module.
"""
import sys

from code.perception.detector import model as _real

sys.modules[__name__] = _real

# Direct-execution entry shim: `python code/nx6_heatmap_model.py` runs this
# file as __main__ -- delegate to the real module's own __main__ block logic
# so the old command keeps working verbatim.
if __name__ == "__main__":
    import torch

    m = _real.TinyHeatmapUNet()
    print("params:", m.num_params(), f"({m.num_params()/1e6:.3f}M)")
    x = torch.randn(2, 4, _real.TARGET_H, _real.TARGET_W)
    q = torch.zeros(2, _real.N_CLASS + _real.N_COLOR)
    q[:, 1] = 1.0
    q[:, _real.N_CLASS + 6] = 1.0
    h, d = m(x, q)
    print("heat", h.shape, "dist_resid", d.shape)
