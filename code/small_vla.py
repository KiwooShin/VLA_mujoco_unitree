"""Compat alias — moved to code.policy.small_vla (RF-1). See docs/refactor_plan.md."""
import sys

from code.policy import small_vla as _real

sys.modules[__name__] = _real
