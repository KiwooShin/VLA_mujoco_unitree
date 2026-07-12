"""Compat alias — moved to code.train.dart_phase (RF-1)."""
import sys

from code.train import dart_phase as _real

sys.modules[__name__] = _real

if __name__ == "__main__":
    _real.main()
