# RF-1 SHIP — deploy sync report

Ship pass for the RF-1 structural refactor (gate: `docs/rf_gate.md` in the
source project, all_pass=y). Scope: sync `VLA_mujoco_unitree/code/` to the
source's (`unitree_vla/code/`) published closure, update the deploy README,
and run a fresh-clone rehearsal of the new tree.

## 1. Rebuild — closure sync

**Closure rule applied:** a file ships iff it is (a) one of the 37
pre-refactor published paths (confirmed via `git ls-tree -r HEAD -- code` in
the deploy repo — read-only, no writes), now a `sys.modules` alias or thin
CLI shim, (b) a new-home module under one of the eleven shipped packages
(`sim/`, `perception/` incl. `detector/`, `control/` incl. `avoid/`,
`policy/` incl. `small_vla/`, `data/`, `datagen/`, `train/`, `eval/`,
`runtime/`, `apps/repl/`, `apps/fancy/`), (c) a package `__init__.py`, or (d)
a `tests/` file under a shipped package. Verified the eleven package
directories in the source tree contain **zero** stray debug/diag/bench/
`nx*judge*`/scratch files (grepped for those patterns — no hits), so every
non-test file under them ships in full. The 48 non-published top-level
scripts in the source `code/` (bench_*, diagnose_*, debug_*, check_ep0/fg/
goal_*, gen_grounding_*, gen_rotation_dart, gen_det_failcases, nx6_judge_*/
nx6_infer_centernet/nx14_judge_compare, record_showcase, render_showcase_*,
render_deliverable, smoke_easy/fix, train_bakeoff/centernet/grounding/
velproprio, trainer.py, verify_*, loss.py, model_centernet.py,
centernet_utils.py, dataset_det.py, combine_rotation_dart.py, deploy_eval.py,
eval_dart_phase/gaitfix/grounding/keyframe.py) were excluded — none are in
the pre-refactor 37, matching the explicit exclusion list in the task.

**Ship manifest:** 246 files = 37 top-level aliases/shims + 209 package
files (new-home modules + package `__init__.py` + `tests/`) across the 11
packages.

**Result: 0 files added, 0 files removed.** The deploy `code/` tree was
already an exact match of the ship manifest — every one of the 246 files was
already present at the correct path with the correct content. No stray
`__pycache__` remained after cleanup (0 dirs).

### Two-direction cmp sweep

- Manifest → deploy: 246/246 present (0 missing).
- Deploy → manifest: 0 extra files (nothing in deploy `code/` outside the
  246-file manifest).
- Content: `cmp -s` on all 246 manifest files, source vs. deploy —
  **246/246 byte-identical, 0 diffs**.

Repository tree shipped (concise — full per-package detail in the README):

```
code/
├── __init__.py, check_env.py                  # top-level, unmoved
├── <37 old-path aliases/shims>                 # e.g. arena.py, demo.py, eval_closedloop.py, ...
├── sim/            (+ tests/)                  # MuJoCo world + WBC teacher
├── perception/      (+ detector/, tests/)       # grounding, lock_mgmt, nx6 heatmap detector
├── control/         (+ avoid/, tests/)           # steer, scan_sched, avoidance
├── policy/          (+ small_vla/, tests/)       # action_stats, groot_lang, student model
├── data/            (+ tests/)                   # dataset loaders
├── datagen/         (+ tests/)                   # rollout/dataset generators
├── train/           (+ tests/)                   # training entry points
├── eval/            (+ tests/)                   # closed-loop evaluators
├── runtime/         (+ tests/)                   # deploy rollout harness (inferencer)
└── apps/
    ├── repl/        (+ tests/)                   # demo.py split
    └── fancy/       (+ tests/)                   # fancy_demo.py split
```

## 2. README — Repository Layout / Running the tests

- **Repository Layout**: rewrote the tree to one line per package (was
  multi-line for `sim/`, `perception/`, `policy/`, `datagen/`), each line now
  states its contents and ends with "own tests/" so the sibling-`tests/`
  convention is visible per-package instead of only in a trailing paragraph.
- **Running the tests**: dropped the `-v` flag so the documented command is
  exactly `python -m unittest discover -s code -p "test_*.py"`; added the
  concrete count ("~1000 tests (1039 in the reference run)") and a note that
  a handful skip gracefully without EGL rendering or the external
  checkpoint/third_party assets, referencing the Prerequisites section.
- Verified every `code/*.py` command path referenced anywhere in the README
  (19 unique entry files: `check_env.py`, `demo.py`, `fancy_demo.py`,
  `eval_closedloop.py`, `eval_maneuver.py`, `eval_nx6_heatmap.py`,
  `eval_search.py`, `gen_dataset.py`, `gen_dart_dataset.py`,
  `gen_det_dataset.py`, `gen_maneuver_dataset.py`, `gen_stand_keyframe.py`,
  `groot_lang.py`, `train_dart_phase.py`, `train_maneuver.py`,
  `train_nx6_heatmap.py`, plus `arena.py`/`grounding.py`/`lock_mgmt.py` cited
  as alias examples) resolves to a real file in the synced deploy tree —
  19/19 present.

## 3. Fresh-clone verification

Clone dir: `/tmp/claude-1000/-home-kiwoos-work/72d96c87-05a3-4831-8aea-c89e6c85ac18/scratchpad/vr1_clone`
(scratch, per VR-1 precedent — `docs/vr1_rehearsal.md`). `rsync -a --delete`
of the deploy `code/` (246 files landed). The two README-documented external
prerequisites, plus the checkpoint/detector artifacts the eval/demo commands
load, were symlinked in read-only from the source project (never writing
into it), matching the README's own Prerequisites paths:
`third_party/Isaac-GR00T`, `checkpoints/GR00T-N1.6-3B`,
`checkpoint/goto_best.pt`, `checkpoint/maneuver_best.pt`,
`runs/nx6_heatmap_B`.

| Check | Result |
|---|---|
| `python -m unittest discover -s code -p "test_*.py"` | **Ran 1039 tests — OK (skipped=5)**. Matches the gate's 1039-test count (skip count 5 vs. the gate's 4 — one extra environment-dependent skip, not a regression; run was fully green either way). |
| `eval_closedloop.py --difficulty easy --goal-source classical --n 2 --seed 999` | **2/2 = 100%** (both SUCCESS, final_dist 0.57m each). |
| `fancy_demo.py --smoke --n-smoke 1 --maxsteps 450 --no-render` | Clean exit (0). `purple ball` 6.20m/66.5° out-of-FOV → `FAIL(scan_timeout)` steps=450 dist=6.597m — same scenario/mechanism as the gate's reference run (`docs/rf_gate.md` §5a, dist=6.575m); the small numeric delta is the same GPU-nondeterminism category already documented for this repo, not a regression. |

## Verdict

Ship complete: closure synced (0 added/0 removed, cmp-clean), README updated,
fresh clone green on all three checks.
