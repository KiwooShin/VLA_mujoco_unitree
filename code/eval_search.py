"""Compat alias — moved to code.eval.search (RF-1).

sample_search_scene / _run_search_rollout / STOP_R_SEARCH / MAXSTEPS_SEARCH /
SEARCH_FOV_HALF_DEG / SEARCH_DIST_MIN / SEARCH_DIST_MAX / SCAN_ALIGNED_THR_DEG
were split out of this module into code.eval.search_types /
code.eval.search_rollout, but code.eval.search re-imports (and thus
re-exports) all of them, so
``from code.eval_search import sample_search_scene, _run_search_rollout, ...``
keeps working.
"""
import sys

from code.eval import search as _real

sys.modules[__name__] = _real

if __name__ == "__main__":
    _real.main()
