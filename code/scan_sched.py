"""Compat alias — moved to code.control.scan_sched (RF-1). See docs/refactor_plan.md."""
import sys

from code.control import scan_sched as _real

sys.modules[__name__] = _real
