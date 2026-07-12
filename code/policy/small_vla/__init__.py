"""
code.policy.small_vla — GroundedNav student (Architecture A modular / C baseline).

RF-1 package split of the old flat code/small_vla.py (554 lines) into:
  - blocks.py : PatchEmbed, Attention, TransformerBlock, TinyViT, ProprioEncoder
                — generic from-scratch vision/proprio building blocks.
  - heads.py  : GroundingHead, VelocityHead, ActionHead, DoneHead — the
                per-modality output heads.
  - model.py  : DEFAULTS, GroundedNav — the top-level student that wires the
                blocks and heads together.

This __init__ re-exports the full flat public surface of the old module so
`from code.small_vla import GroundedNav, DEFAULTS` (and the individual
block/head classes) keep working unchanged for every one of GroundedNav's
importers (train_maneuver.py, trainer.py, train_gaitfix.py, eval_gaitfix.py,
train_dart_phase.py, demo.py, eval_maneuver.py, inferencer.py,
train_grounding.py, train_bakeoff.py, train_velproprio.py,
record_showcase.py, eval_grounding.py) via the code/small_vla.py old-path
compat alias — see docs/refactor_plan.md. GroundedNav's checkpoints store a
plain state dict (tensor name -> tensor), never a pickled class reference,
so only this import path needs to keep resolving; no checkpoint format
changes with this split.
"""

from code.policy.small_vla.blocks import (
    Attention,
    PatchEmbed,
    ProprioEncoder,
    TinyViT,
    TransformerBlock,
)
from code.policy.small_vla.heads import (
    ActionHead,
    DoneHead,
    GroundingHead,
    VelocityHead,
)
from code.policy.small_vla.model import DEFAULTS, GroundedNav

__all__ = [
    "Attention",
    "PatchEmbed",
    "ProprioEncoder",
    "TinyViT",
    "TransformerBlock",
    "ActionHead",
    "DoneHead",
    "GroundingHead",
    "VelocityHead",
    "DEFAULTS",
    "GroundedNav",
]
