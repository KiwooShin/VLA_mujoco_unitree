"""Compat alias — moved to code.perception.detector.data (RF-1).

See docs/refactor_plan.md. Old import paths (`import code.nx6_heatmap_data`,
`from code.nx6_heatmap_data import SplitCache, ...`) keep working unchanged:
this file replaces itself in sys.modules with the real module object, so
both paths refer to the exact same module.
"""
import sys

from code.perception.detector import data as _real

sys.modules[__name__] = _real
