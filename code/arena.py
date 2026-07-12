"""Compat alias — moved to code.sim.arena (RF-1)."""
import sys
from code.sim import arena as _real
sys.modules[__name__] = _real
