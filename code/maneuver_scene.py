"""Compat alias — moved to code.sim.maneuver_scene (RF-1)."""
import sys
from code.sim import maneuver_scene as _real
sys.modules[__name__] = _real
