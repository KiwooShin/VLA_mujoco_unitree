"""Compat alias — moved to code.train.maneuver (RF-1)."""
import sys

from code.train import maneuver as _real

sys.modules[__name__] = _real

if __name__ == "__main__":
    _real.main()
