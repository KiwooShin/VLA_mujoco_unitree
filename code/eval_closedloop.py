"""Compat alias — moved to code.eval.closedloop (RF-1)."""
import sys

from code.eval import closedloop as _real

sys.modules[__name__] = _real

if __name__ == "__main__":
    _real.main()
