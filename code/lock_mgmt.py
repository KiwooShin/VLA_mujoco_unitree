"""Compat alias — moved to code.perception.lock_mgmt (RF-1).

See docs/refactor_plan.md. Old import paths (`import code.lock_mgmt`,
`from code.lock_mgmt import LockGate, ReacquisitionScan`, etc.) keep working
unchanged: this file replaces itself in sys.modules with the real module
object, so both paths refer to the exact same module.
"""
import sys

from code.perception import lock_mgmt as _real

sys.modules[__name__] = _real
