"""Compat alias — moved to code.sim.maneuver_expert (RF-1)."""
import sys
from code.sim import maneuver_expert as _real
sys.modules[__name__] = _real
