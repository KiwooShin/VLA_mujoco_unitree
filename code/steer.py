"""Compat alias — moved to code.control.steer (RF-1). See docs/refactor_plan.md."""
import sys

from code.control import steer as _real

sys.modules[__name__] = _real

if __name__ == "__main__":
    _real._smoke_test()
    sys.exit(0)
