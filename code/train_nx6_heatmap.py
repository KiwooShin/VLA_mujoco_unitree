"""Compat alias — moved to code.train.nx6_heatmap (RF-1)."""
import sys

from code.train import nx6_heatmap as _real

sys.modules[__name__] = _real

if __name__ == "__main__":
    _real.main()
