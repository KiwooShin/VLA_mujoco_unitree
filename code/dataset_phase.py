"""Compat alias — moved to code.data.dataset_phase (RF-1)."""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from code.data import dataset_phase as _real
sys.modules[__name__] = _real
