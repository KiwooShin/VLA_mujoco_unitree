"""Compat alias — moved to code.control.avoid (RF-1). See docs/refactor_plan.md."""
import sys

from code.control import avoid as _real

sys.modules[__name__] = _real

if __name__ == "__main__":
    from code.control.avoid._selftest import main as _main
    sys.exit(_main())
