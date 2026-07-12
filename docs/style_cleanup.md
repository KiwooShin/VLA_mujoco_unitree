# Style Cleanup — Ship Report

Date: 2026-07-11
Agent: SHIP (style cleanup)
Source repo: the source working copy (not under git)
Deploy repo: this repositorys published set

## Scope

The style-cleanup pass touched every `.py` file in `unitree_vla/code/` that already
had a published counterpart in `VLA_mujoco_unitree/code/`. That intersection is
**37 files** (the task brief said 38; the actual overlap counted at ship time was
37 — see Notes below). Files that exist only in the source working tree (dev/debug
scripts such as `debug_ep7.py`, `diagnose_hsv*.py`, `bench_*`, one-off `check_*`
scripts, training-corpus generators not yet promoted, etc.) were intentionally
**not** copied, per instructions — only filenames already present in the deploy
repo were synced.

Before any file in the deploy repo was overwritten, the pre-cleanup version was
byte-copied to `VLA_mujoco_unitree/code_backup_style/<name>.py` (created fresh for
this ship step) so the deploy repo has its own revert path, in addition to the
pre-existing `unitree_vla/code_backup_style/` snapshot of the pre-cleanup source
originals (made by the earlier gate/build pass, timestamps ~02:05–02:09).

## What was standardized

Diffing `unitree_vla/code_backup_style/*.py` (pre-cleanup originals) against the
shipped `unitree_vla/code/*.py` shows the changes are mechanical/stylistic only —
no control-flow or numeric-logic edits were evident in the sampled files
(`action_stats.py`, `check_env.py`, `steer.py`, `scene.py`, `small_vla.py` inspected
line-by-line; all 37 files diffed for line-count deltas, ranging from a handful of
lines in `maneuver_scene.py`/`gen_stand_keyframe.py` up to ~540 changed lines in
`fancy_demo.py`, the largest file in the set). Recurring patterns:

- **Type hints modernized**: `Optional[X]` → `X | None`, `Tuple[...]` → builtin
  tuple syntax, added missing `-> None` / `-> str` / etc. return annotations on
  functions that previously had none, added variable annotations to module-level
  constants (e.g. `MAX_VX: float = 0.55`) and containers (`dict[str, Any]`,
  `list[str]`).
- **Import hygiene**: stdlib imports separated from third-party imports with a
  blank line (PEP 8 grouping), combined imports split onto their own lines
  (`import sys, os as _os` → two `import` statements), unused imports removed
  (e.g. a now-dead `Optional` import after the `X | None` conversion).
- **Docstring style**: NumPy-style `Parameters\n----------` / `Returns\n-------`
  docstrings converted to Google-style `Args:` / `Returns:` / `Raises:` sections;
  bare one-line docstrings collapsed to a single summary line instead of a
  padded multi-line block; a few previously undocumented functions/classes
  gained a short docstring (e.g. `_make_instruction` in `scene.py`, the
  `Attention` class in `small_vla.py`).
- No renames of public functions/classes, no reordering of logic, no changed
  constants/thresholds were observed in the spot-checked diffs.

Per-file diff size (added/removed lines, `diff -u` count):

| file | + | - |
|---|---|---|
| action_stats.py | 46 | 23 |
| arena.py | 129 | 76 |
| avoid.py | 85 | 48 |
| check_env.py | 72 | 10 |
| dataset_maneuver.py | 31 | 11 |
| dataset_phase.py | 20 | 8 |
| dataset.py | 34 | 14 |
| demo.py | 78 | 56 |
| eval_closedloop.py | 38 | 11 |
| eval_maneuver.py | 73 | 14 |
| eval_nx6_heatmap.py | 85 | 12 |
| eval_search.py | 54 | 18 |
| fancy_demo.py | 418 | 124 |
| gen_dart_dataset.py | 105 | 37 |
| gen_dataset.py | 74 | 46 |
| gen_det_dataset.py | 195 | 41 |
| gen_maneuver_dataset.py | 53 | 19 |
| gen_stand_keyframe.py | 17 | 7 |
| groot_lang.py | 62 | 22 |
| grounding.py | 126 | 81 |
| inferencer.py | 148 | 47 |
| __init__.py | 0 | 0 |
| lock_mgmt.py | 23 | 11 |
| maneuver_expert.py | 23 | 16 |
| maneuver_scene.py | 12 | 5 |
| nx6_heatmap_data.py | 80 | 21 |
| nx6_heatmap_eval_utils.py | 64 | 14 |
| nx6_heatmap_model.py | 108 | 35 |
| scan_sched.py | 27 | 11 |
| scene.py | 32 | 28 |
| small_vla.py | 27 | 22 |
| steer.py | 55 | 42 |
| teacher.py | 59 | 17 |
| train_dart_phase.py | 55 | 7 |
| train_gaitfix.py | 133 | 24 |
| train_maneuver.py | 76 | 17 |
| train_nx6_heatmap.py | 67 | 7 |

## Gate results (input to this ship step)

The task brief stated "Gates all passed" for the style cleanup prior to this ship
step. This SHIP agent did not have direct access to a saved gate-agent report
file (searched `unitree_vla/docs/`, the scratchpad directory, and common report
naming conventions — none found), so the specific list of any hunks the gate
agent may have reverted before declaring gates green is **not reproduced here**.
The existing `code_backup_style/` snapshots (in both repos) remain the audit
trail / revert path if that history needs to be reconstructed later.

## Publish step (this ship)

1. **Byte-copy**: 37/37 matched files copied from `unitree_vla/code/` to
   `VLA_mujoco_unitree/code/`, preserving filenames. No files were added to or
   removed from the deploy `code/` directory (file-set diff before/after: none).
2. **Skew sweep**: `cmp` of every deploy `code/*.py` against its source-repo
   counterpart — **0 mismatches** across all 37 shared files.
3. **Fresh-clone runtime check**, run in the pre-staged clone at
   `.../scratchpad/vr1_clone` (a git clone of the deploy repo with
   symlinked `checkpoint/`, `runs/`, `dataset/` fixtures from an earlier "VR-1"
   session), after rsyncing the newly-synced deploy `code/` into its `code/`:
   - `code/eval_closedloop.py --checkpoint checkpoint/goto_best.pt --difficulty easy --n 2 --device cuda --no-render --seed 999`
     → **2/2 (100%) success**, matching the VR-1 precedent's success behavior
     (mean survival 237.5 steps, mean forward displacement 1.76 m, no falls,
     no lost-target/wrong-object failures). Log/summary:
     `scratchpad/ship_eval_closedloop.log`,
     `vr1_clone/eval/style_ship_check/summary_archA_classical_predicted_easy.json`.
   - `code/fancy_demo.py --smoke --n-smoke 1 --maxsteps 300 --no-render --device cuda`
     (bounded scripted episode) → ran to completion cleanly (no traceback/
     exception); outcome was a controlled `scan_timeout` on an out-of-FOV
     long-range target (purple ball, 5.3 m, 133.6° bearing) — the same failure
     mode the pre-existing VR-1 precedent log recorded for this scenario shape
     at a similar step budget, i.e. expected harness behavior, not a crash.
     Log/summary: `scratchpad/ship_fancy_demo.log`,
     `vr1_clone/eval/style_ship_fancy_check/fancy_showcase_summary_fd2.json`.

   Both invocations completed without Python tracebacks or non-zero-looking
   failures (only benign EGL/OpenGL context-teardown warnings on interpreter
   exit, which are pre-existing and unrelated to this change).

## Notes / discrepancies

- Task brief said "38 cleaned files"; the actual filename intersection between
  `unitree_vla/code/` (85 `.py` files) and `VLA_mujoco_unitree/code/` (37 `.py`
  files pre-sync) was **37**. All 37 deploy files matched a same-named source
  file, and no deploy file was left unmatched, so nothing was skipped that
  should have been synced — the count is simply 37, not 38.
- No git was used anywhere in this ship step, per instructions. Reverts, if
  ever needed, go through `VLA_mujoco_unitree/code_backup_style/` (new, made by
  this ship step) or `unitree_vla/code_backup_style/` (pre-existing, made
  before this step).
