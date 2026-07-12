"""code.sim — G1Nav simulation layer (RF-1).

Owns the MuJoCo arena (construction, camera math, rendering), the scene
samplers (goto/search + maneuver), the scripted maneuver FSM expert, and the
WBC locomotion teacher. This package is the RF-1 home for what used to be
five flat modules directly under ``code/``:

    code/arena.py            -> code.sim.arena (+ arena_build/arena_cameras/arena_render)
    code/scene.py             -> code.sim.scene
    code/maneuver_scene.py    -> code.sim.maneuver_scene
    code/maneuver_expert.py   -> code.sim.maneuver_expert
    code/teacher.py           -> code.sim.teacher (+ teacher_smoke)

Every old flat path is preserved as a compat-alias module (sys.modules
substitution) so existing imports (``from code.arena import build_arena``,
``from code.scene import sample_scene``, etc.) keep working unchanged.
"""
