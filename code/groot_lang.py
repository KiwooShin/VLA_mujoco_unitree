"""Compat alias — moved to code.policy.groot_lang (RF-1). See docs/refactor_plan.md."""
import sys

from code.policy import groot_lang as _real

sys.modules[__name__] = _real

if __name__ == "__main__":
    _real.main()
