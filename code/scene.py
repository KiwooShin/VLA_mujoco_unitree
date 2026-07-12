"""Compat alias — moved to code.sim.scene (RF-1)."""
import sys
from code.sim import scene as _real
sys.modules[__name__] = _real
