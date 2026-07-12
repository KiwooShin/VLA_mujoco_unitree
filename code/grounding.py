"""Compat alias — moved to code.perception.grounding (RF-1).

See docs/refactor_plan.md. Old import paths (`import code.grounding`,
`from code.grounding import ground`, attribute access/monkeypatching on
`code.grounding.*`) keep working unchanged: this file replaces itself in
sys.modules with the real module object, so both paths refer to the exact
same module (same globals, same mutable state) -- no copying, no drift.
"""
import sys

from code.perception import grounding as _real

sys.modules[__name__] = _real

# Direct-execution entry shim: `python code/grounding.py` runs this file as
# __main__ (the sys.modules substitution above only affects the import
# system, not direct script execution) -- delegate to the real module's
# smoke test so the old command keeps working verbatim (docs/nx4_depth_split.md,
# docs/nx3_size_gate.md cite `python code/grounding.py` as a smoke check).
if __name__ == "__main__":
    _real._smoke_test()
