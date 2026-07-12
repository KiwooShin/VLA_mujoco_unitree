# NX-15 — Live Instruction-Driven Target Selection for `fancy_demo.py`

**Date:** 2026-07-10
**Agent:** NX-15
**Fixes:** `docs/dr1_demo_reliability.md`'s headline finding — in the live demo
paths, the typed instruction was cosmetic; the rollout always targeted
`scene_cfg['objects'][scene_cfg['target_index']]`, and the file's own
multi-goal parser (`_parse_multi_goal_fancy`) was dead code never called from
any live entry point.

**Requirement:** random scene → user TYPES which object → robot searches →
locates → moves to THAT object.

---

## 1. What changed

`code/fancy_demo.py` gained one new shared function,
`resolve_live_instruction(instruction, scene_cfg)`, plus its internals
(`_split_multi_goal_parts`, `_extract_goal_hint`, `_resolve_goal_to_index`).
It is the single parser+resolver used by **both** live entry points:

- `_terminal_loop()` (terminal REPL mode)
- the Flask `/execute` route → `_do_rollout()` (web UI mode)

No third implementation was written: the clause-splitting regex (`then` /
`and then` / `after that` / `afterwards` / `next`) is the same regex already
used by both `demo.py`'s `Planner.parse()` and the old
`_parse_multi_goal_fancy()`; `_parse_multi_goal_fancy()` itself is kept under
its original name and public contract (`instruction -> [{color, shape,
prompt_part}, ...]`) but is now implemented on top of the new shared
internals instead of its own standalone regex. Ambiguity resolution
(`_resolve_goal_to_index`) is modeled on `demo.py`'s
`Planner._resolve_referent()` (unique match → go; multiple candidates →
clarify).

### Parsing: whole-clause word scan, not adjacent-pair regex

The old `_parse_multi_goal_fancy()` required `{color} {shape}` (or
`{shape} ... {color}`) to appear in a fairly rigid adjacent pattern, which is
why DR-1 documented `"find the reddish ball over there"` as returning `[]`
(adjective wedged between color and shape breaks the regex). NX-15's
`_extract_goal_hint()` instead scans the whole clause for any word that is a
member of the known color set (`red, yellow, blue, green, orange, purple,
cyan` — `code/arena.py`'s full `COLORS`, not just the 4
`RELIABLE_COLORS`, since a user can validly name a non-reliable-color
*distractor* object) or the known shape set (`ball, cube, cylinder, cone` —
`RELIABLE_SHAPES`, reused as-is since shapes have no reliable/unreliable
split). `color`/`shape` are set only when exactly one word of that kind is
found; the full mention-sets are kept for ambiguity scoring. This is
order-independent and handles `"the ball that is red"`,
`"red-colored ball"`, and the DR-1-documented breaking case
(`"the reddish ball"` — `\bred\b` correctly does NOT match inside
`"reddish"`, so only `shape=ball` is extracted, still enough to resolve if
there's a unique ball in the scene) without needing extra regex patterns.

### Resolution against the CURRENT scene

`resolve_live_instruction()` splits the instruction into clauses, extracts a
hint per clause, then resolves each hint against `scene_cfg['objects']`
(never a cached/previous scene — always whatever `scene_manager._scene_cfg`
or `scene_mgr._scene_cfg` currently is). Modes:

| Mode | Trigger | Behavior |
|---|---|---|
| `single` | 1 clause, unique object match | Rollout target = that object's index |
| `multi` | ≥2 clauses ("then"-chained), each resolves uniquely | Routes through `run_fancy_rollout_multi()` (previously dead-reachable only from `run_smoke()`) |
| `clarify` | A clause matches ≥2 objects, tied on attribute score | One-line clarification question, NO rollout |
| `no_match` | A clause parses to a real (color, shape) but no such object exists in the scene | `"No <X> in this scene; scene has: <list>"`, NO rollout |
| `no_parse` | A clause has no recognizable color or shape word at all | `"I didn't understand '<clause>'. Try things like...”`, NO rollout |

Ambiguity scoring (`_resolve_goal_to_index`): when a hint (e.g. shape-only —
`"find the ball"`) matches multiple scene objects, each candidate is scored
by how many of the clause's *other* mentioned color/shape words match its
attributes; a unique top scorer wins without asking, a tie triggers the
one-line clarification. Example (same-color pair `red cone` + `red
cylinder`): `"find the red one"` → tie on the "red" attribute alone (neither
candidate's shape is mentioned) → clarify. `"find the red cylinder"` → exact
match on both attributes → resolves unambiguously to the cylinder.

### Wiring into the two live paths

- **`_do_rollout()` / `/execute`**: parsing+resolution now happens
  **synchronously inside the Flask route handler**, before the rollout thread
  is spawned. `clarify`/`no_match`/`no_parse` return immediately as
  `{"launched": false, "clarify": "..."}` or `{"launched": false, "error":
  "..."}` — the same JSON channel the web UI's `sendInstr()` JS already reads
  (`addLog()` — extended to also branch on `d.clarify`). Only `single`/`multi`
  launch the background rollout thread, which now receives the pre-resolved
  `parsed` dict as an argument instead of re-deriving the target from
  `scene_cfg['target_index']`.
- **`_terminal_loop()`**: calls `resolve_live_instruction()` right after
  reading `input()`; `clarify`/`no_match`/`no_parse` print a `Bot: ...`
  message and loop back to the prompt (no rollout, no `new_scene()`);
  `single`/`multi` proceed to `run_fancy_rollout()` /
  `run_fancy_rollout_multi()` as before.
- In both paths, for `single` mode the target is applied via `resolved_scene
  = dict(scene_cfg); resolved_scene["target_index"] = <resolved idx>` — a
  **copy**, so `run_fancy_rollout()`'s signature and its internal
  `scene_cfg['objects'][scene_cfg['target_index']]` default are **completely
  unchanged**. This is also how `run_fancy_rollout_multi()` already applied
  its own per-sub-goal target override (`sub_scene["target_index"] =
  obj_idx`), so the pattern was already established in this file, not
  invented for this fix.
- The `<TARGET` marker in `_scene_desc()` (web) and `_terminal_loop()`'s scene
  listing was removed — it used to reveal the sampler's pre-picked
  `target_index`, which is exactly the value that used to (wrongly) drive the
  rollout regardless of what was typed. Now that the real target depends on
  the instruction, marking one object as "the" target would be misleading
  again.

### What did NOT change

`run_fancy_rollout()`, `run_fancy_rollout_multi()`, `run_smoke()`, and
`main()`'s argument parsing are byte-unchanged in signature and behavior.
`scene_cfg['objects'][scene_cfg['target_index']]` remains the resolution path
for any caller that doesn't go through `resolve_live_instruction()` — i.e.
every scripted/headless entry point (`--smoke`, showcase/recording scripts
that build `scene_cfg` and call these functions directly with an explicit
`target_index`).

---

## 2. Live test results

All tests below were run against the actual Flask server (`--web`) or the
actual terminal REPL (`--no-render` for speed, matching DR-1's headless
timing method — ~20-25ms/step observed, consistent with DR-1's ~19-41ms/step
finding), using `checkpoint/goto_best.pt`, pure defaults
(`GROUND_NET=1`/`AVOID=1` module defaults, unchanged).

### 2.1 Parser/resolver unit checks (no rollout)

All 9 `resolve_live_instruction()` mode checks passed
(`single`/`multi`/`clarify`/`no_match`/`no_parse`, incl. same-color-pair
ambiguity and a distractor-color target). All 16 rows of DR-1's original
`_parse_multi_goal_fancy` envelope table (§1 of `dr1_demo_reliability.md`)
re-verified — same outputs, including the `"reddish ball"` case which now
additionally carries a resolvable `shape` hint instead of returning nothing.

### 2.2 Live B-not-A tests (the exact scenario DR-1 used to prove the bug)

| # | Path | Scene (`[idx] color shape`, default `target_index` marked) | Instruction | Resolved target | Result |
|---|---|---|---|---|---|
| 1 | web `/execute` | `[0] red cone <- default`, `[1] blue ball`, `[2] red cylinder` | "find the blue ball" | idx 1 | SUCCESS, final_dist=0.478m, steps=638 |
| 2 | web `/execute` | `[0] purple ball <- default`, `[1] cyan cone`, `[2] cyan cube` (same-color pair 1↔2) | "find the cyan cube" | idx 2 | SUCCESS, final_dist=0.485m, steps=1124 |
| 3 | terminal | `[0] red cone <- default`, `[1] blue ball`, `[2] red cylinder` | "find the blue ball" | idx 1 | SUCCESS, final_dist=0.473m, steps=635 |
| 4 | terminal | same scene as #3 | "find the red one" (ambiguous: cone/cylinder both red) | — | `clarify`: *"Multiple matching objects found: red cone (at 4.4m), red cylinder (at 2.8m). Which one? (say the color and the shape)"* — no rollout |

In every launched case the rollout's `final_dist` (~0.46-0.49m, at/under the
0.5m `stop_r` threshold) is measured against the **named** object's
coordinates, which are 2-4m away from the scene's *default* target — a wrong
target would have produced a `final_dist` of several meters (as DR-1's §1
byte-identical-metrics test showed for the pre-fix code), not sub-0.5m.

### 2.3 Web `/execute` clarify + no-match (no rollout launched, scene untouched)

- `"find the red one"` on a `red cone`/`red cylinder` scene →
  `{"launched": false, "clarify": "Multiple matching objects found: red cone
  (at 4.4m), red cylinder (at 2.8m). Which one? ..."}`
- `"find the purple cube"` (not in scene) →
  `{"launched": false, "error": "No purple cube in this scene; scene has: red
  cone, blue ball, red cylinder"}`
- `/scene_info` and `/status` confirmed the scene was untouched by either
  call (no thread spawned, no `new_scene()` triggered).

### 2.4 Multi-goal live test

Web `/execute` with `"find the red cylinder then find the purple ball"` on a
scene `[purple cone <- default, red cylinder, purple ball]`:
`{"launched": true, "mode": "multi", "targets": ["red cylinder", "purple
ball"]}`. Server log:
```
[multi] sub-goal 1/2: 'red cylinder' at dist=4.83m
[multi] sub-goal 1/2 => success  dist=0.476m
[multi] sub-goal 2/2: 'purple ball' at dist=4.60m
[multi] sub-goal 2/2 => success  dist=0.460m
```
2/2 sub-goals succeeded, each against its own named object — neither
defaulted to `purple cone` (the scene's `target_index`).

### 2.5 Terminal scripted-stdin test

Piped stdin (`gibberish nonsense text` / `find the purple cube` / `find the
blue ball` / `new` / `quit`) through `python code/fancy_demo.py --no-render
--device cuda` end to end:
```
Bot: I didn't understand 'gibberish nonsense text'. Try things like 'find the red ball' or 'go to the orange cube'.
Bot: No purple cube in this scene; scene has: red cone, blue ball, red cylinder
Executing: 'find the blue ball' -> target: blue ball
...
Result: SUCCESS  steps=635  dist=0.473m  wall=17.5s
```
followed by a clean `new` (scene cycle) and `quit` (clean exit). A separate
run confirmed the `clarify` branch (§2.2 row 4).

### 2.6 3-episode live-path reliability smoke

Driven through `resolve_live_instruction()` + `run_fancy_rollout()` directly
(DR-1's headless-direct method, `render_video=False`), 3 random scenes,
random valid instructions using 5 varied phrasing templates (`"find the {c}
{s}"`, `"go to the {c} {s}"`, `"walk to the {s} that is {c}"`,
`"locate the {c}-colored {s}"`, etc.), biased 70% toward naming a
**non-default** object:

| ep | instruction | named idx | default idx | result | final_dist | steps |
|---|---|---|---|---|---|---|
| 0 | "go to the purple cylinder" | 0 | 0 | SUCCESS | 0.465m | 1533 |
| 1 | "go to the green cone" | 2 | 0 | SUCCESS | 0.468m | 1404 |
| 2 | "find the red cylinder" | 0 | 0 | SUCCESS | 0.470m | 1489 |

**3/3 success, 0 crashes** — comparable to DR-1's Sweep A 90% single-goal
baseline (small sample; no regressions observed).

### 2.7 Regression check: scripted/headless entry point unaffected

`python code/fancy_demo.py --smoke --out <dir> --device cuda --no-render
--n-smoke 2` (1 single-goal + 1 multi-goal episode, `run_smoke()`'s own
scene sampling and direct `target_index`/goal-list construction, untouched
by this change): **2/2 success**, including the multi-goal sub-goal pair —
confirms `run_fancy_rollout()`/`run_fancy_rollout_multi()` signatures and
`run_smoke()` remain byte-compatible.

---

## 3. Crash sweep

Zero exceptions across all live tests: 9 parser/resolver unit checks + 4
B-not-A live rollouts + 2 clarify/no-match probes + 1 multi-goal live
rollout + 1 terminal scripted-stdin session (3 sub-cases) + 1 terminal
clarify probe + 3 reliability-smoke episodes + 2 headless regression
episodes = **23 live/regression invocations, 0 crashes**.

One **pre-existing, out-of-scope** issue was observed (not introduced by
this change, not touched by this fix): the Flask `/status` route can throw a
transient `TypeError: Object of type ndarray is not JSON serializable` for
the ~3-second window between a rollout finishing and the auto-`new_scene()`
reset, because `_status_state['result']` is set to the raw rollout result
dict verbatim (which contains a `path_trail_out: List[np.ndarray]` field) and
`jsonify()` can't serialize it. This predates NX-15 (the `result` dict
contents and the `/status` handler's `jsonify(dict(_status_state))` were not
modified here) and is orthogonal to instruction parsing; flagging it for a
future pass rather than fixing it under this task's scope.

---

## 4. Files changed / synced

- `code/fancy_demo.py` — all changes described above.
- `code/fancy_demo.py` byte-copied (no git) to
  `VLA_mujoco_unitree/code/fancy_demo.py`; diff confirmed
  empty post-copy.
- `docs/dr1_demo_reliability.md` — "FIXED (NX-15, 2026-07-10)" addendum.
- `docs/nx15_live_parse.md` — this doc.

No separate shared-parser module was created; the single shared function
(`resolve_live_instruction`) and its internals live in `code/fancy_demo.py`
alongside the two call sites that use it, per the task's "one shared
function, not a third implementation" instruction.
