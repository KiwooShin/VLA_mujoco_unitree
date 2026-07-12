"""Compat alias — moved to code.train.gaitfix (RF-1).

GaitFixLoss / _run_epoch / audit_velocity_head were split out of this module
into code.train.gaitfix_loss / code.train.gaitfix_epoch, but code.train.gaitfix
re-imports (and thus re-exports) all three, so
``from code.train_gaitfix import GaitFixLoss, _run_epoch`` keeps working.
"""
import sys

from code.train import gaitfix as _real

sys.modules[__name__] = _real

if __name__ == "__main__":
    _real.main()
