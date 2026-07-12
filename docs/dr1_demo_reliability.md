# DR-1 — Interactive Demo (fancy_demo.py) Wild-Usage Reliability Sweep

**Date:** 2026-07-10
**Agent:** DR-1
**Scope:** `code/fancy_demo.py`'s public-facing interactive-demo flow (terminal
`_terminal_loop` / Flask `/execute` → `_do_rollout`), NOT the gated eval protocols
(`eval_closedloop.py`/`eval_search.py`, whose published numbers are unaffected by
anything in this doc). Pure defaults throughout: no env toggles set
(`GROUND_NET=1`/detector-v2 and `AVOID=1` are already the module-level defaults —
confirmed in `code/grounding.py`/`code/avoid.py` — and every episode's log line
confirms `GROUND_NET=1: loaded detector .../nx6_heatmap_B/model_best.pt`).
Checkpoint: `checkpoint/goto_best.pt` (unchanged). **Measure, don't fix** — no
behavioral changes were made; see §5 for the one crash-check result (none found).

---

## TL;DR

- **Headline finding (not a crash, but the single biggest reliability gap in the
  live demo):** in the actual interactive paths — `_terminal_loop()` and the Flask
  `/execute` → `_do_rollout()` handler — **the typed instruction text is never
  parsed to select a target.** Both call sites pull the target directly from
  `scene_cfg['objects'][scene_cfg['target_index']]` and pass the user's string
  through only as display text (`prompt=instruction`, drawn on the BEV status
  banner). Confirmed two ways: (1) source reading — `run_fancy_rollout()` reads
  `target_color`/`target_shape` from the scene object, never from its `prompt` arg;
  (2) a live behavioral test: the exact same scene (target = purple cone) driven
  with two prompts that name objects **not even in the scene** ("find the red
  ball", "gibberish zzz not english") produced **byte-identical rollout metrics**
  both times (`spotted=False, final_dist=4.936m, steps=600`) — proof the text has
  zero effect. A user who types a mismatched instruction will watch the robot
  confidently narrate progress toward the wrong object with no error, no
  clarification, and no visible sign anything is wrong.
- **The only instruction parser that exists anywhere in the file,
  `_parse_multi_goal_fancy()`, is dead code** — grepped every call site; it is
  defined at line 1058 and never invoked anywhere else, including inside
  `run_smoke()` (which builds multi-goal sub-goals directly from scene object
  indices, not from parsing text). So the multi-goal "find X then find Y" chaining
  shown in `docs/showcase.md`/`docs/cam_p3_demo.md` demo reels is **only reachable
  via the internal `--smoke` scripted path** — a live user typing that exact phrase
  into the web textbox gets treated as a single (ignored) instruction, not a
  2-goal chain.
- **Single-goal rollout mechanics (Sweep A, 30 episodes, full reliable-color ×
  shape matrix, 3–9m, out-of-FOV starts):** 25/30 (83%) succeeded within a 2400-step
  cap; re-running the 5 that hit the cap with 5000 steps recovered 2 more (**27/30 =
  90% true success**). **Zero falls, zero exceptions, zero true scan-timeouts, zero
  scene-placement failures across all 30+5 episodes.** The 3 residual failures that
  persisted even at 5000 steps share an exact, reproducible signature — see §3.2.
- **Multi-goal chaining (Sweep B, 10 episodes, incl. same-color X/Y pairs):**
  **9/10 (90%) succeeded**, no crashes, no cross-goal object confusion (same-color
  pairs like "red ball → red cube" always resolved `_find_obj()` to the correct
  distinct object both times). The 1 failure was sub-goal 1 timing out at 0.79m
  (very close) at the 2000-step-per-goal cap — plausibly a budget artifact, not
  re-verified with extended budget (see §4).
- **Crash sweep:** 0 exceptions across 30 (Sweep A) + 5 (extended reruns) + 10
  (Sweep B) + 8 (Sweep C parser) + 2 (Sweep C live) = **55 rollout/parse
  invocations**. Nothing to fix; no sync to the staging repo was needed.

---

## 0. Method

`code/fancy_demo.py`'s terminal loop and Flask `/execute` handler are both thin
wrappers that call `run_fancy_rollout()` / (never) `run_fancy_rollout_multi()`
directly — see §1 for why multi-goal is smoke-only. This sweep drives those same
functions directly (bypassing the CLI/Flask plumbing itself, which
`docs/cam_p3_demo.md`/`docs/showcase.md` already verified end-to-end over real
HTTP with no bugs found) so that many episodes could be swept in one process
without reloading the `Inferencer` (anti-EGL-context-exhaustion) or launching a
browser.

**`render_video=False` throughout.** A quick timing probe showed video mode (BEV
overlay render + MP4 encode every step, as used for the recorded showcase clips in
`docs/cam_p3_demo.md`/`docs/showcase.md`) costs ~470ms/step, while headless mode
(no video, everything else — including the CAM-2 grounding renders that actually
drive the policy — identical) costs **~19–41ms/step**, a ~15-25x speedup. This sweep
does not need footage, only pass/fail + failure-stage diagnostics, so all 55
episodes ran headless.

All scene sampling for Sweeps A/B forces the target `(color, shape)` explicitly
(a local re-implementation of `sample_fancy_scene_long`'s / `sample_fancy_
multi_goal_scene`'s placement logic, parameterized on target combo instead of an
`rng.choice` draw) so full matrix coverage and same-color multi-goal pairs could be
guaranteed by construction rather than left to chance over 30-40 draws. Scripts:
`sc /tmp` scratchpad `dr1_sweep.py`, `rerun_capped.py` (not part of the repo).

---

## 1. Parser envelope — and why it barely matters for the live demo

`_parse_multi_goal_fancy(instruction)` (`code/fancy_demo.py:1058`) is a
regex-based `(color, shape)` extractor. Tested directly (it is unreachable from
any live entry point, so this characterizes it in isolation):

| Instruction | Parsed |
|---|---|
| `find the red ball` | `[red ball]` |
| `Find the Red Ball` | `[red ball]` (case-insensitive) |
| `go to the red ball` | `[red ball]` |
| `walk to the blue cube` | `[blue cube]` |
| `find me a green cone` | `[green cone]` |
| `please find the orange   cube now` | `[orange cube]` (extra whitespace/words fine) |
| `FIND THE PURPLE CYLINDER` | `[purple cylinder]` |
| `find the reddish ball over there` | **`[]`** — adjective between color and noun breaks the regex |
| `find the red ball then find the yellow cube` | `[red ball, yellow cube]` |
| `find the red ball, then find the yellow cube` | `[red ball, yellow cube]` |
| `find the red ball and then find the yellow cube` | `[red ball, yellow cube]` |
| `find the red ball, after that find the yellow cube` | `[red ball, yellow cube]` |
| `Find The Red Ball Then Find The Yellow Cube` | `[red ball, yellow cube]` |
| `find red ball next find yellow cube` | `[red ball, yellow cube]` |
| `go find the red ball then walk to the yellow cube please` | `[red ball, yellow cube]` |
| `find the red ball then find the yellow cube then find the orange cone` | `[red ball, yellow cube, orange cone]` (3-way chain fine) |

**Verdict: the regex itself is solid** within its documented envelope (simple
`[verb phrase] {color} {shape}` clauses joined by `then`/`, then`/`and then`/`after
that`/`next`) — case, whitespace, extra words, and even 3-way chains all handled;
only an inserted adjective between the color and shape words breaks it (empty
list, not a crash — a documented limitation, not a bug).

**But this is moot for the live demo.** `grep -n _parse_multi_goal_fancy
code/fancy_demo.py` shows exactly one hit besides its own `def` — the module
docstring's TL;DR mention. It is never called by `_terminal_loop()`,
`_start_fancy_web_ui()`/`_do_rollout()`, or even by `run_smoke()` (which hand-
builds `goals` from `objects[0]`/`objects[1]` — see `code/fancy_demo.py:2196-2214`
— never from instruction text). The docstring's claim "Reuses: Planner, Executor,
SceneManager, Inferencer... from demo.py" (line 5) is also stale: `grep -n "^from
code.demo\|^import.*demo" code/fancy_demo.py` returns nothing — `demo.py`'s real
`Planner.parse()` (which does resolve free text against the live scene) is never
imported either. `fancy_demo.py`'s own `FancySceneManager` and rollout functions
are fully self-contained and, in single-goal mode, fully instruction-blind.

**Live behavioral confirmation** (`run_fancy_rollout()`, the exact function both
live entry points call, same scene both times — target = purple cone @ 4.50m,
distractors orange ball / yellow cylinder / blue ball):

| Prompt | spotted | final_dist | steps |
|---|---|---|---|
| `find the red ball` (names an object not in the scene) | False | 4.936m | 600 |
| `gibberish zzz not english` (not parseable at all) | False | 4.936m | 600 |

Identical to 3 decimal places — the prompt is provably inert.

**Practical takeaway for the live demo:** the on-screen scene panel does list the
true target (marked `<TARGET`, via `/scene_info`), so a user who reads it and types
a matching phrase gets what looks like correct instruction-following (survivorship
bias). Any other phrasing — including perfectly well-formed instructions for a
plausible but wrong object, which is the realistic "wild usage" case for a
user who hasn't read the scene list carefully — silently and confidently
executes toward the pre-picked target anyway, with the mismatched text sitting
right there on the status banner the whole time.

---

## 2. Object-class × color matrix (what the sampler actually supports as a target)

`RELIABLE_COLORS = ["red", "orange", "yellow", "purple"]` (`code/fancy_demo.py:102`)
gates every target draw in every sampler actually used at runtime
(`sample_fancy_scene_long`, `sample_fancy_multi_goal_scene`,
`FancySceneManager.new_scene()`) — non-reliable colors (blue/green/cyan) can appear
only as **distractors**, never as the scene's `target_index` object, per the
documented HSV-vs-wall collision (`docs/grounding_dist.md`). So the sampler's real
target envelope is **4 colors × 4 shapes = 16 combos**, not the full 7×4=28 palette.
Sweep A's 30-episode plan covers all 16 at least once (every shape ≥7×, every
color ≥6×, both above the task's ≥4×/≥3× bar) — see §3.1 for coverage counts.

---

## 3. Sweep A — single-goal, 30 episodes

### 3.1 Full episode table

Plan: color-major traversal of all 16 reliable `(color, shape)` combos, then 14
more to pad to 30 (coverage: red/orange/yellow=8 each, purple=6; ball/cube=8 each,
cylinder/cone=7 each — all above the ≥3×/≥4× bar). Distance cycled through 7
buckets spanning 3–9m (±0.4m jitter). All targets sampled out-of-initial-FOV
(guaranteed by the sampler's own placement constraint).

| ep | color | shape | dist(m) | bearing° | result @2400-cap | steps | final_dist(m) |
|---|---|---|---|---|---|---|---|
| 0 | red | ball | 3.17 | +117.1 | OK | 545 | 0.468 |
| 1 | red | cube | 4.23 | -106.5 | OK | 1442 | 0.471 |
| 2 | red | cylinder | 4.74 | +56.3 | OK | 643 | 0.456 |
| 3 | red | cone | 5.94 | +136.3 | OK | 908 | 0.465 |
| 4 | orange | ball | 6.66 | -76.8 | OK | 1693 | 0.468 |
| 5 | orange | cube | 7.88 | -155.1 | **capped** → OK @5000 (4796 steps) | 2400 | 1.113 |
| 6 | orange | cylinder | 9.10 | -137.1 | OK | 2318 | 0.483 |
| 7 | orange | cone | 3.18 | -66.8 | OK | 1251 | 0.470 |
| 8 | yellow | ball | 3.96 | +136.9 | OK | 682 | 0.474 |
| 9 | yellow | cube | 5.05 | +57.1 | OK | 675 | 0.462 |
| 10 | yellow | cylinder | 5.96 | -68.9 | OK | 1595 | 0.466 |
| 11 | yellow | cone | 6.84 | -149.6 | **capped** → **still FAIL @5000** | 2400 | 2.494 |
| 12 | purple | ball | 8.21 | +138.6 | OK | 1182 | 0.462 |
| 13 | purple | cube | 8.61 | -143.4 | **capped** → OK @5000 (2252 steps) | 2400 | 0.893 |
| 14 | purple | cylinder | 2.93 | -101.4 | OK | 1272 | 0.467 |
| 15 | purple | cone | 3.90 | +75.1 | OK | 562 | 0.461 |
| 16 | red | ball | 5.38 | -108.5 | OK | 1621 | 0.462 |
| 17 | red | cube | 6.35 | +102.4 | OK | 898 | 0.457 |
| 18 | red | cylinder | 6.95 | -124.7 | OK | 1893 | 0.473 |
| 19 | red | cone | 7.64 | -154.8 | **capped** → **still FAIL @5000** | 2400 | 1.249 |
| 20 | orange | ball | 8.78 | +53.3 | OK | 1074 | 0.466 |
| 21 | orange | cube | 2.69 | +123.3 | OK | 513 | 0.464 |
| 22 | orange | cylinder | 3.90 | -82.8 | OK | 1374 | 0.460 |
| 23 | orange | cone | 5.10 | +158.4 | OK | 869 | 0.462 |
| 24 | yellow | ball | 6.04 | -60.2 | OK | 1594 | 0.469 |
| 25 | yellow | cube | 7.02 | +149.0 | OK | 1068 | 0.465 |
| 26 | yellow | cylinder | 8.06 | +131.9 | OK | 1151 | 0.464 |
| 27 | yellow | cone | 8.64 | -142.8 | **capped** → **still FAIL @5000** | 2400 | 0.932 |
| 28 | purple | ball | 3.05 | +174.6 | OK | 672 | 0.477 |
| 29 | purple | cube | 4.09 | +81.5 | OK | 587 | 0.466 |

**Success: 25/30 @2400-step cap → 27/30 (90%) with the 5 capped episodes given
5000 steps.** Zero falls, zero exceptions, zero true internal scan-timeouts (the
900-step scan give-up never fired — search always located the target), zero
scene-placement failures anywhere in the 3–9m / full-matrix sweep.

### 3.2 The 3 episodes that failed even at 5000 steps — a genuine, reproducible pattern

All 5 originally-capped episodes were the longest-distance ones (7.6–9.1m,
mechanically expected since more distance = more scan+walk steps). Re-running at
5000 steps (≈2× the original cap) resolved 2 as ordinary "just needed more time":
`orange cube @7.88m` (succeeded at 4796 steps) and `purple cube @8.61m` (succeeded
at 2252 steps — comfortably under even the original cap on replay, i.e. it was
oscillating near the finish line, not truly stuck).

The other 3 — **all three are `cone`, all three required the scan's CW-reversal
leg (bearing -142.8° to -154.8°)** — show an identical, reproducible failure
signature:

1. Approach monotonically closes to within **0.59–0.90m** of the target (tantalizingly
   near the `stop_r=0.5m` threshold — this is not a "lost the target" or "wandered
   off" failure; it gets there).
2. Distance then reverses and **climbs back out**, never re-closing.
3. It settles into a **rock-stable plateau** (flat to ±0.03m) at a larger distance
   — 1.46m (red cone, 7.64m), 2.4m (yellow cone, 8.64m), or 4.76m (yellow cone,
   6.84m) — sustained for **2000+ steps with no further change**, no fall
   (`height` stays in the normal 0.74-0.76m gait band throughout).

| color | shape | dist(m) | bearing° | closest approach | final plateau | steps | fell |
|---|---|---|---|---|---|---|---|
| yellow | cone | 6.84 | -149.6 | 0.60m @step2100 | 4.756m | 5000 | No |
| red | cone | 7.64 | -154.8 | 0.88m @step2200 | 1.455m | 5000 | No |
| yellow | cone | 8.64 | -142.8 | 0.59m @step2300 | 2.399m | 5000 | No |

By contrast, `cube`/`cylinder` targets in the **same** distance+bearing regime
(orange cube -155.1°, orange cylinder -137.1°, purple cube -143.4°) all eventually
succeeded. Cones at shorter range or non-reversal bearings also succeed fine
(`ep7` orange cone 3.18m, `ep15` purple cone 3.90m, `ep23` orange cone 5.10m,
`ep3` red cone 5.94m +136.3° — all OK). So this is not "cones are broken" — it's
specifically **cone + long range (≳6.8m) + reversal-side approach**.

**Hypothesis (not confirmed — flagging for a future diagnosis pass, not fixing
here per this task's mandate):** `cone` is the largest-footprint shape in
`code/arena.py`'s `SHAPES` table (size 0.26 vs. ball/cube 0.24, cylinder 0.22).
Closing to within 0.5m of a slightly larger physical object is a plausible trigger
for a body/leg collision that deflects the robot off its approach line before the
stop condition latches — which would produce exactly this "gets close, then
kicked away to a stable new position" signature, and matches the qualitative
pattern already documented for a different scenario in `docs/nx8_stall.md`
(genuine physical-collision stall: position goes flat, no fall, unrecoverable
without external intervention). This reads as a residual failure class distinct
from the ones already named in that doc and in `docs/fa2_residuals.md`
(`didnt-reach`-by-drift, `scan` under-coverage) — call it **"near-target
collision/stall"** — worth a name in the taxonomy even though root-causing it is
out of scope here. Sample size is small (3/3, but only 3 long-range-reversal cone
episodes existed in this sweep); worth widening before treating the shape
correlation as certain.

---

## 4. Sweep B — multi-goal, 10 episodes

4 same-color pairs (one per reliable color, two shapes each) + 6 different-color
pairs, using `run_fancy_rollout_multi()` (the function `run_smoke()`'s multi-goal
episode calls — **not** reachable from the live terminal/web demo, see §1).

| ep | kind | goal 1 | goal 2 | overall | sub1 | sub2 | sub1 final_dist | sub2 final_dist | total steps |
|---|---|---|---|---|---|---|---|---|---|
| 0 | same_color | red ball | red cube | **FAIL** | didnt-reach | success | 0.793m | 0.478m | 2453 |
| 1 | same_color | orange cube | orange cylinder | OK | success | success | 0.484m | 0.463m | 3186 |
| 2 | same_color | yellow cylinder | yellow cone | OK | success | success | 0.471m | 0.480m | 3134 |
| 3 | same_color | purple cone | purple ball | OK | success | success | 0.471m | 0.467m | 1585 |
| 4 | diff_color | red ball | orange cube | OK | success | success | 0.468m | 0.464m | 2234 |
| 5 | diff_color | yellow cylinder | purple cone | OK | success | success | 0.462m | 0.478m | 2172 |
| 6 | diff_color | orange cone | yellow ball | OK | success | success | 0.464m | 0.469m | 1976 |
| 7 | diff_color | purple cube | red cylinder | OK | success | success | 0.463m | 0.469m | 2441 |
| 8 | diff_color | red cone | purple ball | OK | success | success | 0.470m | 0.482m | 1303 |
| 9 | diff_color | yellow cube | orange cylinder | OK | success | success | 0.469m | 0.468m | 2239 |

**9/10 (90%) overall success.** No crashes, no falls anywhere. The one failure
(`ep0`, sub-goal 1 "red ball") closed to **0.79m** — close but outside `stop_r` —
right at the 2000-step-per-sub-goal cap (dist=0.59m at step 1900, back up to 0.79m
by step 2000); this looks like the same "ran out of my sweep's step budget while
still finishing" pattern as 2 of the 5 Sweep-A cap cases (§3.2), not necessarily a
"cone-class" stall (goal 1 was a `ball`, and it wasn't re-run with extended
budget to confirm either way — flagging as **inconclusive** rather than claiming
it resolves).

**No chain re-query bugs found.** The riskiest case per the task brief — same-color
X→Y pairs, where `run_fancy_rollout_multi`'s `_find_obj()` does an exact
`(color, shape)` match with a same-color-only fallback (`code/fancy_demo.py:1120-
1129`) — worked correctly in all 4 same-color episodes: sub-goal 2 always locked
onto its own distinct object, never onto sub-goal 1's already-completed one
(verified by each sub-goal's `final_dist` independently converging to ~0.46-0.48m
against different target coordinates).

---

## 5. Crash sweep result

Zero exceptions across all 55 rollout/parse invocations (30 Sweep A + 5 extended
reruns + 10 Sweep B + 8 Sweep C static-parser probes + 2 Sweep C live-decoupling
runs) — `grep -i "exception\|traceback\|error:"` over every run log returned
nothing. Zero scene-placement failures (the object-placement rejection sampler
never exhausted its budget, even at the 9m edge of the requested 3-9m range,
against the fixed 8m-half-size arena). **No trivial crash was found, so no fix was
made and no sync to `VLA_mujoco_unitree/code/` was needed.**

---

## 6. Overall verdict

| | Result |
|---|---|
| Single-goal success (2400-step budget) | 25/30 (83%) |
| Single-goal success (extended budget where capped) | **27/30 (90%)** |
| Multi-goal success | **9/10 (90%)** |
| Falls | 0 / 45 |
| Exceptions/crashes | 0 / 55 |
| True scan-timeouts (internal 900-step give-up) | 0 / 45 |
| Instruction parser reachable from the live demo | **None** (single-goal: no
  parsing at all; multi-goal: parser exists but is dead code) |
| Parser envelope (of the unreachable `_parse_multi_goal_fancy`) | Handles all
  tested phrasing/capitalization/whitespace/verb variants and 2-3-way chains;
  breaks only on an adjective wedged between color and shape noun (returns `[]`,
  no crash) |

**Bottom line for a live demo:** the underlying rollout (search → lock →
walk → stop, with the CAM-2 camera handoff and AVOID obstacle bias) is reliably
solid — 90% success with zero falls and zero crashes across the full reliable
color×shape matrix at 3-9m, and the one residual failure class found (near-target
collision/stall, apparently cone+long-range+reversal-specific) is narrow and
visually benign (robot stands and idles at a plateau distance, does not fall or
misbehave dramatically). The actual reliability risk for "wild usage" is not
robustness of the walk — it's that **the demo doesn't do what its own text box
implies**: typed instructions are cosmetic in the only mode a live user can
actually trigger (single-goal), and the multi-goal chaining feature visible in the
showcase reel isn't reachable at all from the interactive UI. Neither is a crash;
both are silent, confidence-preserving behavioral mismatches — worth knowing before
a user starts typing.

---

## FIXED (NX-15, 2026-07-10)

The headline finding above is resolved. Both live paths (`_terminal_loop()` and
the Flask `/execute` → `_do_rollout()` handler) now parse the typed instruction
and drive the rollout toward the object it actually names, sourced from the
CURRENT scene's object list — not `scene_cfg['target_index']`. Full design,
behaviors, and live test evidence are in `docs/nx15_live_parse.md`; summary:

- New shared function `resolve_live_instruction(instruction, scene_cfg)` in
  `code/fancy_demo.py` is the ONE parser+resolver both live entry points call.
  Built on `_parse_multi_goal_fancy()`'s clause-splitting regex (kept under its
  original name/signature) generalized with a whole-clause color/shape word
  scan, plus ambiguity handling modeled on `demo.py`'s
  `Planner._resolve_referent()`.
- `_parse_multi_goal_fancy()` — this doc's "only parser, and it's dead code"
  finding — is no longer dead: `resolve_live_instruction()` reuses its
  internals and multi-goal ("find X then find Y") instructions from either live
  entry point now route through `run_fancy_rollout_multi()`, previously
  reachable only from `run_smoke()`.
- Behaviors added: exact match → go; ambiguous (e.g. "the ball" with two balls)
  → score by how many of the clause's OTHER words match each candidate, tie →
  one-line clarification (terminal: reprompt; web: `/execute` returns
  `{"launched": false, "clarify": "..."}`, shown via the same `addLog()` channel
  the UI already used for errors); no match → `{"launched": false, "error":
  "No <X> in this scene; scene has: <list>"}`, no rollout.
- **Live test, the exact scenario this doc used to prove the bug** (§1's
  "Live behavioral confirmation" table): re-run with the fix in place, naming
  an object B while the scene's `target_index` default points at object A
  (e.g. scene = `[red cone <- default, blue ball, red cylinder]`, instruction
  `"find the blue ball"`) — the rollout now measures success against B's
  coordinates and reaches it (`final_dist≈0.47-0.49m`), confirmed across 4
  independent B-not-A live runs (2 web `/execute`, 2 terminal), including two
  same-color-pair scenes (`red cone`/`red cylinder`; `cyan cone`/`cyan cube`)
  where exact (color, shape) matching had to discriminate between same-colored
  siblings. One live multi-goal `/execute` case ("find the red cylinder then
  find the purple ball") also succeeded end-to-end, 2/2 sub-goals, each
  measured against its own named object.
- **Scripted/headless entry points are untouched and confirmed byte-compatible**:
  `run_smoke()` (and by extension the showcase/recording APIs that call
  `run_fancy_rollout()`/`run_fancy_rollout_multi()` with an explicit
  `scene_cfg['target_index']`) were not modified and re-verified working
  (2/2, single + multi-goal) after the change — `scene_cfg['objects']
  [target_index]` remains the default/fallback for any caller that sets it
  explicitly; only the two live entry points now override it per-instruction.
- 3-episode live-path reliability smoke (random scenes, random valid
  instructions, some naming a non-default object, varied phrasing templates):
  **3/3 success, 0 crashes** — comparable to (this doc's) Sweep A's 90%
  baseline, on a 3-episode sample.
- Files synced byte-identical (no git) to
  `VLA_mujoco_unitree/code/fancy_demo.py`.
