"""Compat alias — moved to code.runtime.inferencer (RF-1).

See docs/refactor_plan.md. Old import paths (`import code.inferencer`,
`from code.inferencer import Inferencer, RolloutResult, _compute_gt_goal,
_build_proprio, _apply_student_pd, _GaitPhaseTracker, _write_video,
FALL_HEIGHT, GROUNDING_PERIOD, ...`), including monkeypatching
`code.inferencer.classical_ground = ...` (code/gen_det_failcases.py,
eval/nx7_ep1_diag/*, eval/nx8_stall/*) and `mock.patch("code.inferencer.
Inferencer")` (code/apps/fancy/tests/test_cli.py), keep working unchanged:
this file replaces itself in sys.modules with the real module object, so
both paths refer to the exact same module (same globals, same mutable
state) -- no copying, no drift.
"""
import sys

from code.runtime import inferencer as _real

sys.modules[__name__] = _real

# Direct-execution entry shim: `python code/inferencer.py` runs this file as
# __main__ (the sys.modules substitution above only affects the import
# system, not direct script execution) -- delegate to the real module's
# smoke test so the old command keeps working verbatim.
if __name__ == "__main__":
    _real._smoke_test()
    sys.exit(0)
