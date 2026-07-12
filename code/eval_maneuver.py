"""Compat alias — moved to code.eval.maneuver (RF-1).

run_maneuver_rollout / HEADING_SUCCESS_THR / IMG_SIZE / ManeuverResult were
split out of this module into code.eval.maneuver_types /
code.eval.maneuver_rollout, but code.eval.maneuver re-imports (and thus
re-exports) all of them, so
``from code.eval_maneuver import run_maneuver_rollout, evaluate_maneuver,
HEADING_SUCCESS_THR, IMG_SIZE`` keeps working.
"""
import sys

from code.eval import maneuver as _real

sys.modules[__name__] = _real

if __name__ == "__main__":
    _real.main()
