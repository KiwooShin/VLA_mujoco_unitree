"""Compat alias — moved to code.policy.action_stats (RF-1). See docs/refactor_plan.md."""
import sys

from code.policy import action_stats as _real

sys.modules[__name__] = _real
