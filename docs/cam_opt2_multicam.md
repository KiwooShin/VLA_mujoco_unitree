# CAM-2: Multiple Cameras (Head + Proximity/Chin Cam)

**Design-brief finding for the near-field target-loss problem** (target drops below the frame /
into body-occlusion within ~0.9–1.15 m of the goal, currently forcing ~100-step open-loop
dead-reckoning on the last EMA'd goal vector — `HOLD_GOAL_HORIZON`, `docs/grounding_dist.md`,
`docs/vision_grounding.md`). Design/feasibility brief; no code changed.

## 1. Multi-camera coverage/fusion approaches (cited)

**Dedicated near-field/downward cameras are a standard pattern on legged/humanoid platforms,
added *alongside* — not instead of — the main head/front camera.** Bipedal-robot head/neck
patents describe a **chin-mounted downward camera** specifically so the robot "can look
downward... without needing to use cameras positioned in the forehead region" — i.e. the
forehead/eye camera structurally cannot see what a chin camera is for ([Head and neck assembly
of a bipedal robot](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12447628)).
Production humanoids generalize this to full multi-camera arrays (Tesla Optimus: 8 cameras;
some designs mount 4 cameras around the pelvis at a **35° downward pitch** for combined
coverage — [Stereolabs humanoid camera overview](https://www.stereolabs.com/en-si/machine/humanoid-robot)),
and even a well-resourced commercial quadruped (Boston Dynamics Spot, 5 stereo pairs, ~360°
combined FOV) still ships with **documented residual blind spots near the hips/body** —
multi-camera coverage reduces but does not eliminate near-body blind zones; it's an accepted,
standard trade-off, not a workaround.

**Closest functional analog — quadruped foothold cameras.** A hierarchical quadruped
navigation system explicitly diagnoses our exact failure mode: *"the forward-facing camera
has no perception of obstacles around the foothold; locomotion control may fail"* — and fixes
it with a **second, steeply-tilted "foothold" depth camera** dedicated to 0.5 m-ish near-field
terrain, run by a **separate low-level policy at a different hierarchy level**, only combined
with the front-camera's high-level output at the controller ([Hierarchical Vision Navigation
System for Quadruped Robots with Foothold Adaptation Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10256005/),
Sensors 23(11):5194). Critically, **the two cameras are never pixel-fused** — each runs its own
perception independently and the two outputs are reconciled at the decision layer. This is
exactly the shape of the design below: two (soon three) classical HSV+depth detectors, one
selected per cycle, not a joint multi-view model.

**Camera handoff / selection literature** (multi-camera surveillance & tracking) frames
selection as three parts — *trigger, consistent labeling, next-camera choice* — and explicitly
recommends **triggering the switch near a camera's known minimum-object-distance boundary**,
using each camera's own distance-accuracy envelope to decide when to hand off ([Camera handoff
and placement for automated tracking with multiple omnidirectional cameras](https://www.academia.edu/13921719/Camera_handoff_and_placement_for_automated_tracking_systems_with_multiple_omnidirectional_cameras);
[Camera handoff with adaptive resource management](https://www.sciencedirect.com/science/article/abs/pii/S0262885609002431)).
This is precisely a **hysteresis switch keyed on last-known target distance**, not a novel
mechanism we have to invent — §2 below applies it directly.

**Real-hardware constraint that must be carried into the design.** The G1's stock head sensor
is an Intel RealSense D435i, minimum depth range **~0.40 m** (848×480@90fps, 86°×57° depth FOV);
the D455 upgrade option is tuned for long range instead ([OpenELAB D455/D435i for Unitree
G1](https://openelab.io/blogs/learn/intel-realsense-d455-depth-camera-for-unitree-g1-edu-complete-guide),
[RobotShop D435i for G1](https://www.robotshop.com/products/unitree-intel-realsense-d435i-depth-camera-g1-humanoid-robot)).
MuJoCo's depth buffer has no such floor, so **sim will show clean detection down to 0.2–0.3 m
that a stock D435i-class sensor physically cannot deliver** — a real sim-to-real gap flagged in
§3/§4, not something multi-camera placement alone fixes.

## 2. Recommended proximity-cam config + handoff rule

**Geometry model (validated against this codebase, not just theory).** For a downward-tilted
pinhole camera at height *H* above ground, pitch θ, vertical half-FOV φ=FOVY/2:
`d_near = H/tan(θ+φ)` (closer objects fall below the frame), `d_far = H/tan(θ−φ)` (farther
objects fall above it, θ>φ). With `H = CAM_HEAD_Z + pelvis ≈ 0.55+0.74 = 1.29 m`, `FOVY=45°`
(the *rendered* FOVY per `grounding.py: EGO_FOVY_RENDERED`), this formula reproduces the two
existing cameras exactly: θ=32° (ego) → d_near=**0.920 m** (matches the reported cutoff);
θ=26° (`GROUNDING_PITCH`) → d_near=**1.141 m**, d_far=**21.1 m** (matches why 26° was chosen
for the 4–9 m band). Sweeping θ at the **same H=1.29 m, same FOVY=45°** (i.e. same head mount
point, only the tilt differs — literally a third fixed camera at the existing head origin):

| θ (pitch) | d_near (m) | d_far (m) |
|---|---|---|
| 26° (grounding cam) | 1.14 | 21.1 |
| 32° (ego cam) | 0.92 | 7.71 |
| 50° | 0.41 | 2.48 |
| **58° (proposed proximity cam)** | **0.22** | **1.81** |
| 65° | 0.06 | 1.41 |

**θ=58° at the *same* mount point covers 0.22–1.81 m** — inside the 0.3 m target floor with
margin, past the 1.5 m proximity-band ceiling with margin, and overlapping the ego cam's own
0.92 m near-cutoff by **~0.9 m** (a wide, safe hysteresis band). No height or forward-offset
change is required to hit spec — this is the simplest possible addition: reuse `_set_ego_cam`
verbatim with a new constant pitch.

**Proposed constants** (arena.py, mirroring the existing `CAM_PITCH`/`GROUNDING_PITCH` pattern):

| Constant | Value | Rationale |
|---|---|---|
| `PROXIMITY_PITCH` | 58.0° | d_near≈0.22 m, d_far≈1.81 m at existing head mount |
| `PROXIMITY_W, PROXIMITY_H` | 320×240 | same as ego; close targets are large in frame, no resolution premium needed → cheapest of the three renders |
| `PROXIMITY_FOVY` | 45.0° | same as existing two → `get_ego_intrinsics()` reused unmodified |
| `CAM_HEAD_Z`, `CAM_FWD` | unchanged (0.55, 0.10) | reuse existing mount point — only θ differs |

**Extrinsic correction (keeps the goal vector consistent across cameras — the core "no
retraining" mechanism).** `cam_to_egocentric()` already un-pitches by an arbitrary `pitch_deg`
and adds a **constant forward offset** (`CAM_ROBOT_FORWARD_OFFSET_M=0.947`) to correct for the
`cam.distance=0.001`+"lookat 1 m ahead" quirk in `_set_ego_cam` that displaces the *effective*
render origin from the pelvis by `CAM_FWD + cos(θ)·1.0`. For θ=32° this is `0.10+cos32°=0.948`
(matches the empirical 0.947 to 1 mm). **The same formula for θ=58° gives an analytic estimate
of `0.10+cos58°≈0.63 m`** — this should be a new `CAM_PROXIMITY_FORWARD_OFFSET_M` parameter
threaded through `cam_to_egocentric(..., forward_offset_m=...)` (today it's a single hardcoded
constant reused, imprecisely, for *both* existing cameras — a ~5 cm error the system already
tolerates, so a first-pass analytic proximity-cam offset is acceptable to prototype with, then
calibrate empirically exactly as E6 did for the original 0.947 m constant). Because every
camera funnels through the *same* `ground()` → `(dist, cosθ, sinθ)` representation, **the
frozen downstream policy never sees which camera produced the goal** — this is what makes the
option retraining-free.

**Handoff rule (hysteresis / Schmitt trigger — directly the "switch near minimum-object-distance"
pattern from §1):**
```
state: active_cam ∈ {GROUNDING, PROXIMITY}; last_known_dist (EMA'd, already exists)
D_LO, D_HI = 1.2, 1.6   # m — inside the 0.92–1.81 m dual-visible band

each grounding cycle (5–10 Hz):
    render ONLY active_cam; run classical_ground() on it
    if detected:
        last_known_dist = EMA(last_known_dist, new_dist)
        if active_cam == GROUNDING and last_known_dist < D_LO: active_cam = PROXIMITY
        if active_cam == PROXIMITY and last_known_dist > D_HI: active_cam = GROUNDING
    else:
        miss_count += 1
        if miss_count >= 2:               # bounded fallback probe
            render the OTHER camera this cycle only; adopt its result if found
    default at episode start / no prior detection: GROUNDING (targets start 1.5–9 m away)
```
This is a small, additive state machine in `inferencer.py`'s existing classical-grounding
block (mirrors the EMA/hold logic already used for `goal_source='learned'`) — it does not
touch `grounding.py`'s `ground()` signature beyond adding the `forward_offset_m` parameter.

**Demo display.** `fancy_demo.py`'s `compose_sbs_frame(ego, bev)` already does exactly this
kind of side-by-side `cv2.concatenate`; extend to a 3-pane **ego | proximity | BEV**
(`STREAM_W = EGO_W + PROXIMITY_W + BEV_W`), with the status banner (already drawn per-frame)
labeled with which camera is currently active — makes the handoff visible in the demo, not
just numerically correct.

## 3. Expected performance

- **Full-range visibility 0.3–8 m?** In sim: **yes**, by construction — `{proximity:
  0.22–1.81 m} ∪ {ego: 0.92–7.71 m, currently unused for classical detection but available} ∪
  {grounding: 1.14–21 m}` spans continuously from 0.22 m to 20+ m with two comfortable overlap
  bands. On real hardware: bounded below by the D435i-class **~0.40 m minimum depth floor**
  (§1) — treat **0.4–0.5 m**, not 0.3 m, as the realistic hardware floor unless paired with a
  shorter-range depth modality.
- **Retains 4–9 m detection (68% demo goto)?** **Yes, unchanged** — this option is purely
  additive (new camera + new selection branch); the grounding-cam code path, its 26° pitch, and
  its validated 78%-detectable-color / 46.7% overall demo numbers are untouched. The hysteresis
  default (GROUNDING unless `last_known_dist<1.2 m`) means nothing changes for the entire
  existing far-field regime.
- **Real-time with 2 renders?** **Yes in steady state** — only *one* camera is rendered per
  grounding cycle (mutual exclusion via hysteresis), so per-cycle cost is unchanged from today's
  measured **151 ms mean / 185 ms p95 render** at 5–10 Hz cadence (`docs/grounding_dist.md`); a
  320×240 proximity render is, if anything, cheaper than the 480×360 grounding render. Two
  renders in the *same* cycle only happen transiently at handoff/fallback-probe (a handful of
  times per episode) — a bounded ~150 ms one-off against a 600–1400-step episode, not a
  sustained cost. **On the real robot this concern doesn't exist at all**: both RGBD streams
  arrive continuously and for free from onboard sensors (per this project's own eval-protocol notes: "on a real robot...
  render cost is zero"); the "2 renders" question is purely a MuJoCo-sim engineering detail.
- **Enables vision-based stopping?** **Yes** — in sim, down to ~0.22 m, comfortably inside the
  existing 0.5 m stop-radius / `HOLD_STEPS_REQUIRED` logic, replacing most of the current
  ~0.9–1.5 m blind dead-reckoning window with an actual visual fix. On hardware, expect reliable
  vision-stopping down to ~0.4–0.5 m — still cutting the blind final-approach distance by
  roughly half to two-thirds versus today.

## 4. Trade-offs, feasibility verdict, config to prototype

**Compute/engineering cost — low.** One more persistent `mujoco.Renderer` (same
pre-allocate-once pattern already used for `_gr_rend`, which is precisely what fixed the prior
EGL-context-exhaustion bug — precedent already exists), one more entry in the offscreen buffer
`max(W,H)` sizing, one small hysteresis state machine (~20–30 lines), one new forward-offset
constant to calibrate. No architecture change, no MJCF change (all three cameras are virtual
`mjCAMERA_FREE` cams, not body-attached — no XML edits needed, same as today).

**Advantage over dynamic/active tilt (CAM-3).** Because every proximity-cam pitch here is
*fixed*, this option does **not** need to fix the position/pitch-coupling quirk in
`_set_ego_cam` (the render origin drifting with θ) as a hard prerequisite the way a continuous
tilt schedule does — it only needs **one** offset calibration per fixed camera, exactly the
already-solved, low-risk process behind the existing 0.947 m constant. The cost of that
simplicity is a discrete third render call instead of a free `atan2` recompute.

**Real risk — self-occlusion at steep pitch, not modeled identically in sim.** The same
position/pitch quirk (render origin ≈ mount + 1·view-direction) means a *steeper* pitch shifts
the effective optical center further **down and less far forward** (cos58°=0.53 vs cos32°=0.85)
— i.e. closer to the robot's own chest than the shallower existing cameras, which is exactly
where body/arm-swing occlusion becomes plausible on real hardware, and needs an empirical
check (render test frames at 0.3–1.5 m and visually confirm no persistent self-occlusion,
same validation style as E6/V2). Mitigation lever if it appears: bump `CAM_FWD` for this
camera specifically (0.10→0.25–0.35 m) to push the optical center further past the torso.
Also shared with every near-field option: `MIN_DEPTH_M=0.60` in `grounding.py` currently
discards depth <0.6 m and **must be lowered** (e.g. ~0.15–0.2 m) or the near-field fix will
appear to still fail regardless of camera geometry.

**Verdict: high feasibility, low-to-moderate cost, non-regressive by construction** (existing
camera paths untouched) and it directly targets the stated goal without touching the frozen
policy. Recommended as the primary near-field fix candidate — simpler and lower-risk than a
continuously-servoed camera, at the modest cost of one more render call that is already
compute-free on real hardware.

**Single config to prototype:**
`PROXIMITY_PITCH=58.0°`, `PROXIMITY_W,H=320,240`, `PROXIMITY_FOVY=45.0°`, same head mount
(`CAM_HEAD_Z=0.55`, `CAM_FWD=0.10`) as the existing ego camera; `CAM_PROXIMITY_FORWARD_OFFSET_M
≈0.63 m` (analytic first pass, calibrate like the 0.947 m constant); hysteresis thresholds
`D_LO=1.2 m / D_HI=1.6 m` on the existing EMA'd `last_known_dist`; lower `MIN_DEPTH_M` to
~0.15–0.2 m alongside it.
