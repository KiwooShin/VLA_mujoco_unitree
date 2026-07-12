"""Compat alias + CLI shim — moved to code.datagen.gen_stand_keyframe (RF-1)."""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from code.datagen import gen_stand_keyframe as _real
sys.modules[__name__] = _real

if __name__ == "__main__":
    _real.gen_keyframe()
