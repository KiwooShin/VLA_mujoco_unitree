"""Compat alias — moved to code.sim.teacher (RF-1).

Also a thin CLI entry shim: docs/teacher.md documents
``MUJOCO_GL=egl python code/teacher.py`` as the smoke-test invocation, so
direct execution of this old path still runs the smoke test (now living at
code.sim.teacher_smoke), matching pre-RF-1 behavior.
"""
import sys
from code.sim import teacher as _real
sys.modules[__name__] = _real

if __name__ == "__main__":
    from code.sim.teacher_smoke import main
    sys.exit(main())
