# CAM-1: Wide-FOV Single Camera

**Design-brief finding for the near-field target-loss problem** (target drops below the frame
within ~0.9–1.15 m of the goal — see `docs/grounding_dist.md`, `docs/vision_grounding.md`, and
sibling brief `docs/cam_opt2_multicam.md`). Design/feasibility brief; **no code changed**. All
numbers below are measured empirically against the real `arena.py`/`grounding.py` code (via a
scratch harness that imports the actual `build_arena`, `_set_ego_cam`, `get_ego_intrinsics`,
`ground()` — not a hand re-derivation), run on this machine's GB10 GPU with `MUJOCO_GL=egl`.

## 1. How wide-FOV cameras work, and the trade-off (cited)

A pinhole/rectilinear camera's focal length in pixels is fixed by vertical resolution and FOV:
`f = (H_px/2) / tan(FOVY/2)`. Widening FOVY at constant resolution *shrinks* `f`, so **every
object's pixel footprint shrinks with it** (`pixel_size ≈ f·D/distance`) — this is the
fundamental FOV-vs-angular-resolution trade-off: "narrowing the field of view significantly
increases the angular resolution, with much higher sampling density" and vice versa
([mobile-robot fisheye perception discussion](https://commonlands.com/blogs/technical/wide-angle-lenses-and-fisheye-camera-lens-distortion)).
The only way to widen FOV *without* losing far-object detectability is to grow resolution to
compensate — there is no free lunch.

**Distortion/rectification is a separate axis, and doesn't apply here.** True fisheye
projections (equidistant, equisolid, stereographic) are a different, non-linear pixel-to-angle
mapping used specifically to exceed what a rectilinear lens can reach; the standard
generalization (Kannala-Brandt) is explicitly scoped to wide-angle/fisheye lenses and is
typically fit **up to ~110–115° FOV** in practice ([A Generic Camera Model and Calibration
Method for Conventional, Wide-Angle, and Fish-Eye Lenses](https://www.computer.org/csdl/journal/tp/2006/08/i1335/13rRUyv53Gt);
[MATLAB fisheye/KB calibration](https://www.mathworks.com/help/vision/ref/cameraintrinsicskb.html)).
Our target FOVY (80–90°, see §2) is comfortably inside plain-rectilinear territory — real
depth cameras used on humanoids ship in almost exactly this range (RealSense D435: ~87°×58°
depth FOV; D455: ~86° matched RGB/depth FOV — [Intel D435 specs](https://www.intel.com/content/www/us/en/products/sku/128255/intel-realsense-depth-camera-d435/specifications.html),
[D455 specs](https://www.intel.com/content/www/us/en/products/sku/205847/intel-realsense-depth-camera-d455/specifications.html)).
**Conclusion: no fisheye lens, no distortion model, no rectification step is needed.** MuJoCo's
`mjCAMERA_FREE` renderer is already a plain perspective/pinhole projection (confirmed:
`get_ego_intrinsics()`'s `fx==fy` derivation is exactly the standard pinhole formula), so
`backproject_pixel()`/`cam_to_egocentric()` in `grounding.py` need **zero math changes** for
FOVY=80–90° — only the FOVY *value* fed into them needs to change (see the critical bug in §2).
This mirrors the accepted pattern for near+far complementary coverage on legged/manipulator
robots — e.g. a wide body-mounted camera for global context paired with (or, here, substituting
for) a narrow-purpose close-range view ([FALCON loco-manipulation](https://arxiv.org/pdf/2512.04381)).

## 2. Recommended concrete config — and two critical implementation findings

**Geometry model** (validated against this codebase — same formula as `cam_opt2`):
for a downward-tilted pinhole camera at height `H`, pitch `θ`, half-FOV `φ=FOVY/2`:
`d_near = H/tan(θ+φ)` (bottom-edge ground intersection), and if `θ>φ`, `d_far = H/tan(θ−φ)`;
**if `θ≤φ` the horizon itself is inside the frame and there is no far-edge cutoff at all** —
distant ground points asymptote toward the horizon row but never leave the frame. At the
existing mount (`CAM_HEAD_Z=0.55`, pelvis≈0.79 → `H≈1.34m`) and `θ=32°` (`CAM_PITCH`, unchanged
per this option's brief): solving `θ+φ=76.9°` for `φ` with `θ` fixed gives **FOVY≈90°** as the
FOVY that (a) pushes `d_near` down to ~0.30 m and (b) simultaneously puts the horizon inside the
frame (`θ−φ=−13°<0`), so far-range is bounded only by pixel resolution/detector sensitivity, not
by frame geometry. **This matches the brief's own hint exactly.** FOVY=80° is a valid, cheaper
alternative (`d_near≈0.42m`) if 0.30 m turns out not to be strictly required.

### Finding #1 — `EGO_FOVY=90.0` in `arena.py` is currently a dead constant

`get_ego_intrinsics(w,h,fovy_deg)` (arena.py:264) computes `fx,fy` from whatever `fovy_deg` is
passed — but the *actual rendered* FOVY for a MuJoCo `mjCAMERA_FREE` camera comes from
**`model.vis.global_.fovy`** (a model-wide setting), which `build_arena()` never sets — it stays
at MuJoCo's compiled-in default, **45°**. This exact 90-vs-45 mismatch is what forced the
existing `EGO_FOVY_RENDERED=45.0` hardcode in `grounding.py:50`
(`get_ego_intrinsics_rendered()`) — i.e. the codebase has already been bitten by this once.
**To actually widen FOV, add `spec.visual.global_.fovy = 90.0` in `build_arena()`** (same place
`spec.visual.global_.offwidth/offheight` are already set, arena.py:137-141), and update
`EGO_FOVY_RENDERED`/`get_ego_intrinsics_rendered()` in `grounding.py` **in the same commit** —
ideally collapse these into one shared constant so this class of bug can't recur.

### Finding #2 — the real near-field floor is a camera-rig bug, not FOV (measured, not assumed)

`_set_ego_cam()` (arena.py:236) builds the camera via `cam.lookat = origin + 1.0·forward_dir`,
`cam.distance = 0.001`. Since MuJoCo's free-camera eye is `lookat − distance·forward`, the
*true* rendered eye sits at `origin + 0.999·forward_dir` — i.e. **already ~0.947 m forward and
~0.53 m lower** than the intended head mount (verified analytically — `0.10+cos(32°)=0.948m`,
matching the codebase's own empirically-measured `CAM_ROBOT_FORWARD_OFFSET_M=0.947` — and
verified by direct rendering: injecting a target at world distance `d` and sweeping `d` shows
the *actual* current ego camera (45°,32°) only becomes visible at `d≈1.1m`, not the idealized
`0.92m`). **Widening FOVY to 90° at this same buggy eye position only moves the cutoff to
`d≈0.85–0.9m`** — a 15% improvement, because the eye's own forward displacement, not the FOV
cone, is now the binding constraint. **Reaching the stated 0.30 m target requires also fixing
the eye position** (e.g. `cam.distance = 1.0` instead of `0.001`, which places the eye exactly
at the intended origin). With both changes, direct measurement shows continuous visibility from
**d=0.2 m through 9 m** (see §3). This fix is still pure sim-camera config (no policy retrain),
but it has two consequences that must ship with it:

- **`CAM_ROBOT_FORWARD_OFFSET_M=0.947` must be recalibrated** (to ≈0.10 m, the true offset once
  the rig bug is fixed) or reported `(dist,bearing)` become silently wrong — measured **~1.6 m
  error at a true 6 m distance (>25%)** when I ran `ground()` end-to-end with the fix applied but
  the *old* offset constant left in place. This is a one-constant recalibration, exactly the
  already-proven process behind the original 0.947 m measurement, but it is a hard
  correctness-blocking prerequisite, not an optional nicety.
- **The robot's own feet enter the bottom of frame.** Rendering a standing pose with the eye-fix
  applied shows the feet occupying roughly the **bottom 30–35% of the image** (they are hidden
  today purely as a side-effect of the eye already being pushed forward of them). A centerline
  target still shows through the gap between the feet in a static standing pose, but a blunt
  bottom-crop big enough to guarantee excluding the feet would re-blind exactly the 0.2–1.0 m
  range this option is trying to add. **Recommend depth-based self-body rejection** (reuse the
  foreground/background depth-mask machinery already in `grounding.py:429-546`, built for a
  different purpose — rejecting wall blobs — extended to reject the robot's own very-near depth
  range) instead of enlarging `IMG_MARGIN_BOTTOM` (currently 0.05, arena.py/grounding.py:130).
  `IMG_MARGIN_LEFT/RIGHT` (0.03 each) likely also need widening, since a wider vertical FOV at
  fixed aspect ratio widens horizontal FOV too, bringing more wall into frame and aggravating the
  already-documented wall/cyan-blue HSV collision (`docs/grounding_dist.md`).

### Two concrete options to prototype

| | **A — FOV-only (conservative)** | **B — FOV + eye-position fix (full)** |
|---|---|---|
| Change | `spec.visual.global_.fovy=90°` only | A + `cam.distance: 0.001→1.0` in `_set_ego_cam` |
| Near cutoff | ~0.85–0.9 m (measured) | ~0.2–0.3 m (measured) |
| Meets 0.3 m goal? | No (partial win only) | Yes |
| Self-occlusion risk | None — feet stay out of frame (confirmed) | Real — feet enter bottom ~30-35% of frame (confirmed); needs depth-based masking |
| `CAM_ROBOT_FORWARD_OFFSET_M` | Unchanged, still valid | Must be recalibrated (≈0.10 m) — else dist/bearing wrong by >25% |
| Risk | Low | Moderate — two more moving parts, both well-scoped |

Resolution: recommend replacing the current two-renderer setup (`_ego_rend` 320×240 +
`_gr_rend` 480×360) with **one** renderer at **640×480**, FOVY=90°, same `θ=32°` tilt — cheaper
overall (one render call instead of two) despite the resolution bump. Escalate to 864×648 (or
higher) only if the far-field margin at 640×480 (see §3) proves insufficient in the full
closed-loop demo eval.

## 3. Expected performance (measured, not projected)

**Pixel area at distance** (0.48 m target sphere, `MIN_BLOB_AREA` threshold = 40 px² per
`grounding.py:124`), FOVY=90°, θ=32°, **Option B (eye-fix applied)**:

| d | 0.2m | 0.3m | 0.5m | 1.0m | 2.0m | 4.0m | 6.0m | 7.0m | 8.0m | 9.0m |
|---|---|---|---|---|---|---|---|---|---|---|
| 640×480 | 2806 | 6609 | 11705 | 6225 | 2169 | 713 | 355 | 270 | 216 | **177** |
| 864×648 | 5119 | 12058 | 21290 | 11347 | 3952 | 1297 | 647 | 499 | 390 | **317** |

Both are visible continuously across the entire 0.2–9 m range (no gap) and both stay
comfortably above the 40 px² threshold at 9 m (4.4× and 7.9× margin respectively). **Without the
eye-fix (Option A)**, the same 640×480/90° config is **invisible below ~0.85 m** — geometry alone
cannot get closer than that at the current (buggy) eye position, confirming §2's finding.

**Does it regress current 4–9 m detection?** The existing best-case far-field camera
(`GROUNDING_PITCH=26°`, 480×360, 45° FOVY) measures **577 px² at 9 m** in the same harness.
640×480/90° (Option B) gives 177 px² (31% of that margin, still 4.4× over threshold); 864×648
gives 317 px² (55%, 7.9× over threshold). **Functionally still working, with reduced safety
margin** — recommend re-running the full demo-distribution closed-loop eval (the 68%
demo/goto benchmark referenced in the brief) before committing; if margin proves too thin in
practice, ~960×720–1152×864 fully matches or exceeds the current 577 px² figure (interpolating
the measured area∝H² scaling), at a render cost still well inside real-time (below).

**Real-time?** Isolated render+depth+`ground()` cost, measured directly in this environment
(GB10, EGL, one concurrent training job running — not an idle machine):

| Config | ms/frame (render+HSV+depth, full `ground()` call) |
|---|---|
| current (480×360, 45°, θ=26°) | 4.7 |
| wide90 + eyefix, 640×480 | 6.9 |
| wide90 + eyefix, 864×648 | 8.6 |
| wide90 + eyefix, 1152×864 | 19.1 |

Taken at face value these are trivial against the 100–200 ms budget implied by 5–10 Hz. **But
this codebase's own production benchmark** (`docs/grounding_dist.md`) measured the *current*
480×360 path at **151 ms mean / 185 ms p95** under real deployment conditions — over 30× my
isolated number, almost certainly reflecting GPU/CPU contention from concurrent training/eval
jobs in this shared environment (one was in fact running during my measurement, confirming this
is a live pattern here, just apparently not yet enough to reproduce their full slowdown).
Scaling the *documented* 151 ms baseline by my measured **relative** multipliers (1.49× at
640×480, 1.84× at 864×648, 4.10× at 1152×864) projects to **~225 ms / ~278 ms / ~619 ms** —
i.e. 640×480 is a tolerable stretch of the budget (cadence drops from 6.6 Hz toward ~4.4 Hz,
slightly under the 5 Hz floor), while 864×648+ risks falling meaningfully below 5 Hz under
representative load. **Recommend re-measuring end-to-end grounding cadence in the actual running
system** (not just this isolated harness) before locking in a resolution above 640×480.

**Enables vision-based stopping?** With Option B's continuous 0.2–9 m coverage, the measured
`(dist,bearing)` stays valid essentially to contact, so the controller can switch its stop
decision from dead-reckoning to the live grounding output almost the whole way in. Realistic
framing: expect a residual ~0.15–0.2 m blind zone right at contact (no downward-tilted camera at
finite height can see the ground directly beneath/behind itself), so the practical design is
"vision-guided to ~0.2–0.3 m, then a short fixed final approach" rather than pure vision-stopping
to 0 cm — still a large improvement over today's ~0.9–1.5 m blind window.

## 4. Trade-offs, feasibility verdict, config to prototype first

**Cost.** Cheap: one constant (`spec.visual.global_.fovy`), one resolution bump, one line in
`_set_ego_cam` (Option B), one constant recalibration (`CAM_ROBOT_FORWARD_OFFSET_M`), and
extending existing depth-mask logic to also reject the robot's own near-field depth range. No
MJCF/XML edits (same as the existing two virtual cameras — `mjCAMERA_FREE`, not body-attached).
Net render calls go from 2/cycle → 1/cycle, partially offsetting the per-pixel cost of the
resolution bump.

**Real risk.** Option B's two prerequisites (offset recalibration, self-body masking) are both
individually well-scoped and precedented in this codebase's own history (E6/V2 fixes did
exactly this kind of empirical constant calibration), but skipping either one silently breaks
something non-obvious: skip the recalibration and distance estimates are wrong by >25% with no
visible symptom until closed-loop eval regresses; skip the masking and the robot's own feet can
occasionally spoof or occlude a near-field detection during gait. Neither is a blocker, both are
must-do items, not nice-to-haves.

**Verdict: feasible, moderate cost, and the only option among "widen this one camera" variants
that can plausibly reach the stated 0.30 m target** — but only the full version (B), not FOV
widening alone (A gets to ~0.85 m, a real but partial win). A true fisheye/panoramic lens is
**not warranted**: 90° is comfortably inside standard rectilinear range, needs no distortion
model, and already achieves continuous 0.2–9 m coverage in this geometry — going wider would
only cost angular resolution for no additional benefit at this task.

**Single config to prototype first:** FOVY=90° (`spec.visual.global_.fovy`, set at build time),
θ=32° (`CAM_PITCH`, unchanged), 640×480, single renderer replacing both `_ego_rend`/`_gr_rend`;
`cam.distance: 0.001→1.0` in `_set_ego_cam`; recalibrate `CAM_ROBOT_FORWARD_OFFSET_M` to ≈0.10 m
(verify empirically, don't trust the analytic estimate alone); add depth-based self-body
rejection using the existing FG-mask code path. Validate in this order: (1) re-run
`bench_detection_dist`-style distance sweep to confirm the recalibrated offset, (2) render a few
standing/walking frames to confirm feet are correctly rejected, (3) full demo/classical
closed-loop eval to confirm no regression vs the 68% baseline, (4) escalate resolution only if
step 3's far-field margin is insufficient.
