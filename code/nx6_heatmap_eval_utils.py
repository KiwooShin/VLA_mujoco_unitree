"""Compat alias — moved to code.perception.detector.eval_utils (RF-1).

See docs/refactor_plan.md. Old import paths (`import
code.nx6_heatmap_eval_utils`, `from code.nx6_heatmap_eval_utils import
run_inference, ...`) keep working unchanged: this file replaces itself in
sys.modules with the real module object, so both paths refer to the exact
same module.
"""
import sys

from code.perception.detector import eval_utils as _real

sys.modules[__name__] = _real
