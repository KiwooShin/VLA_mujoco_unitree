# CAM-3: Active Gaze Control (Dynamic Tilt) / Camera Repositioning

**Design-brief finding for the near-field target-loss problem** (target drops out of frame within ~0.9 m of the goal, blocking vision-based stopping). Design/feasibility brief; no code changed.

## 1. Background: active vision & camera-mount geometry

**Active vision / gaze control.** Classical active-vision literature distinguishes *saccades* (fast re-orientation to acquire a new fixation point) from *smooth pursuit* (continuous tracking that keeps a target foveated as it or the observer moves) — both implemented on pan-tilt heads that servo the optical axis toward a target so it stays centered in a narrow, high-resolution FOV rather than relying on a wide static FOV (Cannata & Maggiali; Yaguchi et al., *Pan-tilt camera control for vision tracking*; INRIA *Gaze control of an active vision system in dynamic scenes*). Humanoid platforms with head+neck gaze DOF (e.g. iCub: 3-DOF eyes + 3-DOF neck) use exactly this "look at the target, not at the horizon" strategy during reach/approach, including switching to a narrower, more steeply-downward-pointed view as the hand/body nears the object ("look-down-while-reaching"). This is the biological/robotic precedent for our proposal: **re-aim the optical axis at the target's *current* estimated position instead of holding a fixed pitch.**

**Camera-mount geometry (height/tilt vs. blind zone).** For a downward-tilted pinhole camera at height *H* above the ground, pitch θ below horizontal, vertical half-FOV φ=FOVY/2, the ground-plane visibility band is bounded by two rays:
- **Near edge** (bottom of frame): `d_near(θ) = H / tan(θ+φ)` — objects closer than this fall out the bottom.
- **Far edge** (top of frame, only exists if θ>φ): `d_far(θ) = H / tan(θ-φ)` — objects farther than this fall out the top (∞ if θ≤φ).

This matches the general mobile-robot/LiDAR-mounting result that *raising sensor height widens the blind zone and increases far reach, while increasing tilt angle shrinks the blind zone and shrinks far reach* (e.g. a 40°-tilted LiDAR example reducing blind spot from 3 m→0.21 m at the cost of range). **The two knobs trade off in opposite directions — a single fixed (H,θ) cannot maximize both near and far coverage simultaneously.** Servo/tracking literature also flags a real cost of *fast* pan-tilt motion: rolling-shutter "jello" distortion and motion blur under rapid re-aiming degrade detection accuracy, and mechanical servo error adds jitter — relevant to how fast we're allowed to slew pitch.

## 2. Recommended design for this system

**Verified against our own geometry.** With `H=CAM_HEAD_Z+pelvis=0.55+0.74=1.29 m`, `FOVY=45°` (φ=22.5°, the *actual* rendered FOVY per `grounding.py`'s `EGO_FOVY_RENDERED`, not the stale `arena.EGO_FOVY=90`), plugging θ=32° into `d_near` gives **0.920 m** — an exact match to the reported "~0.92 m bottom-edge cutoff," confirming the model. The same formula at θ=32° gives `d_far ≈ 7.71 m`, and at θ=26° (current `GROUNDING_PITCH`) gives `d_near≈1.14 m`, `d_far≈21 m` — consistent with why 26° was chosen for the 4–9 m demo band (huge far margin, only slight near-cutoff shift). Sweeping θ shows the fundamental limit:

| θ (pitch) | d_near (m) | d_far (m) |
|---|---|---|
| 20° | 1.41 | ∞ (θ<φ) |
| 26° (current grounding cam) | 1.14 | 21.1 |
| 32° (current "ego" mount ref) | 0.92 | 7.71 |
| 40° | 0.67 | 4.09 |
| 50° | 0.41 | 2.48 |
| 65° | 0.056 | 1.41 |

No single row covers 0.3–8 m: to see 0.3 m you need θ≈65°, but that caps far vision at 1.4 m and would *destroy* the validated 4–9 m detection. **A fixed mount cannot satisfy the spec — this is the quantitative case for dynamic tilt.**

**Controller.** Drive pitch from the grounding module's own last-known distance estimate `D_hat` (already held/EMA'd across the 5 Hz grounding cadence — no new sensing needed):

```
pitch_deg = clip( degrees(atan2(H_cam, max(D_hat - F_cam, eps))), PITCH_MIN=20°, PITCH_MAX=65° )
```

This is the "look at the target's base" centering law, clamped to a realizable band. Solving the clamp boundaries against the same formula shows the schedule is *unclamped* (perfectly centers the target, β=0) for `D∈[0.70, 3.64] m`, and saturates at 65°/20° outside that. At the extremes: D=0.3 m → target sits ~86% down the frame (14% margin to the bottom edge — same order as the existing 91%-down tolerance already used for 9 m targets); D=8 m → target sits ~26% down from the top (comfortable margin, since 20°<φ means no far cutoff exists at all). **Net: the clamped schedule covers the full 0.3–8 m span with margin at both ends**, using exactly the [20°,65°] range the task brief proposed.

**Cheap to recompute per frame.** Intrinsics (fx,fy,cx,cy) depend only on (W,H,FOVY) — pitch never touches them, so nothing to recompute there. What *does* need to vary per frame is the extrinsic `pitch_deg` fed to `arena._set_ego_cam()` (already a parameter) and to `grounding.cam_to_egocentric(..., pitch_deg=...)` (already a parameter) — this is an O(1) `atan2`+`clip`, negligible at 5–10 Hz.

**Fixed-mount alternative (if a static mount is preferred).** The table above proves no single pitch works, but a **forward-offset increase** (`CAM_FWD` 0.10→~0.35–0.45 m, e.g. a small forehead boom) is a "free" win layered on any pitch choice: it shifts `D_near = d_near(θ)+F` closer to the robot origin by a constant, without any of the multiplicative θ trade-off, and directly reduces the plausible body/limb occlusion cone in front of the chest on real hardware. Recommendation if dynamic tilt is rejected: keep the existing two fixed cameras (32°/26°) and simply extend `CAM_FWD`, accepting a near limit around ~0.6–0.7 m (not 0.3 m) — better than status quo but short of the target range.

## 3. Expected performance

- **Visible 0.3→8 m, centered:** Yes over most of the range (perfectly centered 0.70–3.64 m); safely in-frame with 14–26%+ margin at the 0.3 m and 8 m extremes. Enables the near-field detection the task needs.
- **Retains 4–9 m detection:** Yes — for D>3.65 m the schedule clamps at 20°, which is *shallower* (more far-permissive, `d_far=∞`) than the currently-validated 26°, so the 4–9 m band (87% detectable-color rate observed) is preserved or slightly improved, not regressed.
- **Real-time:** Yes, trivially — one `atan2`+clip per grounding cycle (5–10 Hz); no measurable added latency to the 50 Hz control loop.
- **Enables vision-based stopping:** Yes, this is the main payoff — a valid (dist,bearing) estimate should now exist all the way to the ~0.3 m stopping band, where currently it goes blind.
- **View-stability concerns:** Two real risks, both fixable pre-prototype:
  1. **Position/pitch coupling bug in `arena._set_ego_cam`.** The free `MjvCamera` is positioned via `cam.distance=0.001` + a `lookat` point 1.0 (unit-vector) m ahead of the intended mount — this makes the *effective* render-camera position silently drift with pitch (this is exactly why `grounding.py` needed the empirical `CAM_ROBOT_FORWARD_OFFSET_M=0.947` hack, measured only for θ=32°). A distance-dependent pitch would need this fixed first (set `cam.distance=1.0` to match the unit lookat vector, decoupling position from pitch entirely) — a one-line, low-risk correctness fix, but a hard prerequisite for a *continuous* schedule (the current single-constant offset hack does not generalize across pitches).
  2. **`MIN_DEPTH_M=0.60` floor.** At D≈0.3 m the true camera-to-target depth is likely <0.6 m and would be silently discarded by the existing near-depth noise filter — this must be lowered (e.g. to ~0.15–0.2 m) alongside the tilt change, or the near-field fix will appear to still fail. Convenient overlap: real D435-class RGBD sensors also bottom out near ~0.2–0.3 m, so this isn't purely a software choice.
  3. HSV hue thresholds are ~view-angle-invariant (diffuse/matte materials, no specular model); shape-based circularity for the ball is angle-invariant, but cube/cone silhouette scoring was tuned mainly around 20–32° and may need re-validation across the full 20–65° sweep.

## 4. Trade-offs, HW realism, feasibility verdict

**HW realism — this is the load-bearing caveat.** The stock Unitree G1 has **no independent head/neck pan-tilt joint**; the D435i/camera is rigidly fixed inside the head. The nearest actuator is the 3-DOF **waist** (yaw ±155°, pitch ±45°, roll ±30°), which moves the *entire* torso+arms, not just the camera. So on real hardware, "dynamic tilt" is not a camera-only config change — it is either (a) an aftermarket pan-tilt neck mount (hardware modification, outside stock config), or (b) repurposing waist pitch, which perturbs the gait/balance the distilled, WBC-free joint policy was tuned around — likely requiring retraining, which the task explicitly rules out. **This means the SIM-side benefit (fixing the classical grounding module used for training/eval) is fully "legal" and cheap, but literal on-robot deployment of continuous dynamic tilt is NOT currently feasible without either a hardware add-on or violating the no-retrain/WBC-free constraints.** Separately, since the sim's grounding camera is a free virtual camera (not attached to a body/joint in `g1_gear_wbc.xml`, which has no head joint at all), there's no MJCF change needed to prototype this in sim — it's pure software.

There is also a real self-occlusion risk on hardware that the current sim doesn't model: because of the position/pitch coupling bug above, the sim's grounding camera effectively sits ~1 m *ahead* of the robot body, so the body never occludes it in-frame (confirmed in `grounding.py`'s own comments) — but a *true*, correctly-mounted head camera steepening its gaze toward the robot's own feet at close range is exactly the geometry where swinging legs/arms would start to clip the view on real hardware. This can't be validated in the current sim without also fixing the position/pitch coupling and re-rendering with the real mount point.

**Verdict: feasible and recommended for sim prototyping now** (cheap, no retraining, closes a proven quantitative gap the current fixed-pitch scheme structurally cannot close); **not yet hardware-deployable as-is** — flag as sim-only / research-prototype pending either a neck-mount hardware decision or a waist-pitch co-design (with re-validated gait) as future work.

**Single config to prototype:** replace the constant `GROUNDING_PITCH=26.0` with
`GROUNDING_PITCH = clip(degrees(atan2(1.29, D_hat - 0.10)), 20.0, 65.0)`,
computed once per grounding cycle from the held/EMA'd `D_hat`, fed into both `_set_ego_cam(..., pitch_deg=GROUNDING_PITCH)` and `cam_to_egocentric(..., pitch_deg=GROUNDING_PITCH)` — after (i) fixing `cam.distance` to `1.0` in `_set_ego_cam` so the mount point stops drifting with pitch, and (ii) lowering `MIN_DEPTH_M` to ~0.15–0.2 m.
