# CAM-4: LiDAR / Extra Depth-Sensor Fusion for Close-Range Approach

**Design brief CAM-4.** Design/feasibility brief — no code changes. Evaluates adding a LiDAR
(or a second/wide depth sensor) to solve the documented close-range problem: the ego camera
(1.29 m height, 32° down-tilt, ~45° FOV) puts the target below the frame bottom and into
body-occlusion by ~0.9 m, after which the deployed system (`code/inferencer.py`) already
falls back to `HOLD_GOAL_HORIZON=100`-step **open-loop dead reckoning** on the last EMA'd
goal vector (`_last_known_goal`, `docs/grounding_dist.md`, `docs/vision_grounding.md`).

## 1. LiDAR-camera fusion for colored objects, and the color-ID wall

LiDAR (and most depth-only sensors) return **geometry, not color/texture** — this is the
first thing every camera-LiDAR fusion survey states as the motivating limitation: LiDAR's
monochromatic point cloud "makes it challenging... to recognize objects, limiting its
usefulness in applications requiring detailed semantic information," which is why fusion
work exists at all (colorization/"painting" surveys, PMC12610118; OmniColor, arXiv:2404.04693).
The standard fix pattern in the literature is never "identify with LiDAR" — it's **identify
with the camera, then hand the identity to a geometric track**:

- **PointPainting** (Vora et al.) projects LiDAR points into a camera-only segmentation/detection
  output and "paints" each point with the resulting class score before handing the enriched
  cloud to a LiDAR-only tracker — camera does semantics, LiDAR does geometry, sequentially fused.
- **Image–point-cloud instance matching** (MDPI *Sensors* 26(2):718, 2026) projects 2D instance
  masks into LiDAR space via calibrated extrinsics to carve out an **object-specific point
  cluster**, then runs Euclidean clustering + a Kalman-filter tracker with residual-gated data
  association on that cluster to survive occlusion.
- **Person-following robots** (the closest analog to "approach a specific known target through
  the last few meters") fuse a 2D laser range finder with a camera precisely because "when a
  person wears clothing that obscures leg visibility to the laser, the camera can maintain
  tracking using color," and conversely the laser keeps a leg-cluster track alive when the
  camera loses the person to occlusion/FOV (Springer 10.1007/s00521-022-07765-6; MDPI
  *Sensors* 25(6):1754, human-following with 3D LiDAR + point-cloud projection).

**Takeaway for a color-instruction task:** LiDAR can never answer "is this the red ball or the
blue one" — it can only continue tracking a cluster that a camera has already labeled. Any
LiDAR design here is strictly a **post-identification geometric tracker**, never a grounding
replacement.

## 2. Concrete pipeline for this system, if pursued

The repo's architecture already separates "goal source" from the joint policy
(`goal_source ∈ {gt, classical, learned}` in `inferencer.py`), so a LiDAR track would slot in
as a 4th goal source with zero policy changes:

1. **Camera phase (existing, unchanged):** `classical_ground()` in `code/grounding.py` runs
   HSV+depth at 5–10 Hz, producing `(dist, cosθ, sinθ, confidence)` in camera frame. Continue
   until `confidence` drops / `not_visible=True` (target ~0.9 m, exits frame/gets occluded).
2. **Seed the geometric track:** back-project the last valid color-blob centroid (already done
   via depth median in `grounding.py`) into robot-frame `(x, y, z)`. This is the LiDAR track's
   initial state — identical in spirit to PointPainting's mask→cluster handoff, just without a
   neural segmenter (classical HSV mask is the "instance mask").
3. **Scan the LiDAR/depth fan:** cast a ray fan from the sensor origin each control tick.
   MuJoCo has no native point-cloud/LiDAR sensor type; the two supported ways to get one are
   (a) declare a grid of `<rangefinder>` sensors via the `<replicate>` MJCF tag (one issue
   explicitly requests this pattern for LiDAR, google-deepmind/mujoco#1654), or (b) call
   `mj_multiRay()` directly from Python once per tick with a programmatically built ray-origin/
   direction array — cheaper to iterate and what most MuJoCo RL-sim LiDAR rigs actually do
   (google-deepmind/mujoco#1439). For this task, (b) is preferable: a forward fan of
   ~30×20 rays spanning ±60° azimuth × −20°…+65° elevation from a pelvis-mounted site is enough
   (no need to pay for a full 360° scan — the target is always roughly ahead once tracking has
   been seeded).
4. **Cluster + associate:** bucket adjacent same-range rays into clusters (identical to the
   classic laser-based leg-clustering used in person-following: adjacent beams within a small
   range-jump threshold = one cluster). Pick the cluster nearest the *predicted* target position
   (previous position, since a stationary target + known robot velocity gives a trivial
   constant-position/constant-velocity gate) — this is the same nearest-neighbor gating used in
   the cited human-tracking-with-3D-LiDAR papers, just without their pedestrian-motion model
   since targets here are static.
5. **Emit the goal:** convert the tracked cluster centroid to `(dist, cosθ, sinθ)` and write it
   into the *same* `cached_goal_vec` / EMA slot the classical path already uses — the policy
   never sees the difference. Fall back to today's `_last_known_goal` hold if the cluster is
   lost for > some horizon (identical hold-goal safety net, just with a tighter horizon since the
   geometric track should re-lock almost every tick a valid cluster exists).

## 3. Expected performance and cost

**Real-time:** cheap in simulation. Analytic ray-mesh intersection (`mj_ray`/`mj_multiRay`) costs
microseconds per ray — negligible next to the 151 ms mean OpenGL grounding-render already
measured in `docs/grounding_dist.md`. Hundreds of rays/tick is not a real-time risk here; the
system's real bottleneck is (and remains) RGB/depth *rendering*, not ray casting.

**Detection quality at close range:** genuinely better than what the pipeline currently trusts.
`grounding.py` hard-floors valid depth at `MIN_DEPTH_M=0.60` ("sensor noise / very close floor");
a real scanning LiDAR such as the Livox Mid-360 — notably the exact accessory Unitree already
sells for the G1 (`G1-MID360L`) — has a blind zone of only **0.1–0.2 m**, 360°×(−7°…52°) FOV,
200k pts/s, 265 g, ~6.5 W. So a LiDAR would out-range the existing depth pipeline in the
0.1–0.6 m band specifically. But note this 0.6 m floor in the current code is a **software
choice**, not a hardware limit of the existing RGB-D depth channel — it can likely be relaxed
for free.

**Expected SR impact:** modest, and *not* aimed at the documented dominant failure modes.
This project's own failure taxonomy (`docs/grounding_dist.md`) attributes
demo-distance misses to (1) cyan/blue-vs-wall HSV collision at 4–9 m, (2) same-color-distractor
mis-selection, (3) out-of-scan-range initial bearing — all **far-range identification**
problems, not close-range stopping precision. Reported final-approach behavior is already
0 falls with `stop_r` 0.4–0.6 m absorbing residual drift over the ~1.1 m of blind walk the
100-step hold covers. A geometric close-range tracker would tighten last-mile stopping
precision and rescue the rare "overshoot/circles" cases (e.g. `grounding_dist.md` ep10), but
it **cannot** fix distractor mis-selection once *both* the target and a same-color distractor
are already out of camera FOV — geometry alone still can't tell them apart, matching §1's
citations exactly. Rough expectation: **+low single digits to ~10 pp** on scenes where the
target is cleanly isolated near the approach path; **~0 pp** on the distractor/wall-collision
failure modes, which are the majority of the residual gap.

## 4. Honest verdict

**Not worth it for this task**, for three compounding reasons:

1. **It doesn't touch the actual bottleneck.** This project's own diagnosis loop already
   isolated the dominant demo-distance failures as far-range color/HSV problems (wall
   collision, distractors, scan coverage) — a close-range geometric tracker is orthogonal to
   all three.
2. **It duplicates a cheaper existing capability.** The onboard RGB-D camera already renders a
   depth channel every grounding tick; the "loses target at 0.9 m" problem is really a
   **camera-FOV/pitch/occlusion** problem (which a wider-FOV camera, a second downward camera,
   or active tilt — the sibling CAM options — fix at the *source*, and while still carrying
   color) plus an overly conservative `MIN_DEPTH_M` in software. Both are deploy-side,
   zero-new-hardware fixes consistent with the project's "no retraining, deploy-side only"
   constraint; a LiDAR is not.
3. **Integration cost is disproportionate to the payoff.** Even though MuJoCo ray-casting
   itself is computationally free, adding a LiDAR still means: a new MJCF ray-fan (sim) or a
   real Livox Mid-360 with extrinsic calibration to the RealSense camera, a driver, time-sync,
   and a clustering/data-association module (real engineering, not a config change) — for a
   sensor whose entire contribution is "hold the last color-verified position slightly more
   robustly for the last ~0.5–0.9 m," a job the existing EMA + hold-goal mechanism already does
   with 0 recorded falls.

**Where LiDAR *would* earn its keep:** if this were headed to the real G1 outdoors or in
lighting conditions where HSV becomes unreliable across the board (not just at long range),
360° coverage and lighting-invariance become genuinely valuable, and Unitree conveniently
already sells the exact accessory (`G1-MID360L`, 265 g, ~6.5 W). But that is a different
problem (illumination/lighting robustness) than the one stated here (color-instruction final
approach), and should be scoped as such if ever revisited.

**Recommendation:** skip LiDAR. If last-mile stopping precision is later shown to be a real
measured failure mode (not yet demonstrated in the eval logs), prefer implementing the same
"visual-ID → geometric nearest-cluster track" *pattern* on the existing RGB-D depth channel
(cheap, sim-only, no new sensor) before considering new hardware.

---
### Sources
- LiDAR monochromatic limitation / colorization surveys: [LiDAR Point Cloud Colourisation Using Multi-Camera Fusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC12610118/); [OmniColor (arXiv:2404.04693)](https://arxiv.org/pdf/2404.04693)
- PointPainting: [Point Cloud Painting for 3D Object Detection (PubMed 38400401)](https://pubmed.ncbi.nlm.nih.gov/38400401/); [PointPainting: Sequential Fusion for 3D Object Detection](https://www.researchgate.net/publication/337485096_PointPainting_Sequential_Fusion_for_3D_Object_Detection)
- Image–point-cloud instance matching / clustering + Kalman tracking under occlusion: [Robot Object Detection and Tracking Based on Image–Point Cloud Instance Matching, MDPI Sensors 26(2):718](https://www.mdpi.com/1424-8220/26/2/718)
- Person/human-following camera+LRF fusion, color-vs-geometry handoff: [Detecting and tracking using 2D laser range finders and deep learning](https://link.springer.com/article/10.1007/s00521-022-07765-6); [Robust Human Tracking Using a 3D LiDAR and Point Cloud Projection, MDPI Sensors 25(6):1754](https://www.mdpi.com/1424-8220/25/6/1754); [Person-Following Algorithm Based on Laser Range Finder and Monocular Camera Data Fusion](https://www.researchgate.net/publication/344614584_Person-Following_Algorithm_Based_on_Laser_Range_Finder_and_Monocular_Camera_Data_Fusion_for_a_Wheeled_Autonomous_Mobile_Robot)
- MuJoCo ray casting / LiDAR simulation: [mj_ray / mj_multiRay, MuJoCo API functions](https://mujoco.readthedocs.io/en/latest/APIreference/APIfunctions.html); [Using the replicate tag on a rangefinder sensor to simulate a LIDAR, google-deepmind/mujoco#1654](https://github.com/google-deepmind/mujoco/issues/1654); [How can I implement a rangefinder sensor?, google-deepmind/mujoco#1439](https://github.com/google-deepmind/mujoco/issues/1439)
- Livox Mid-360 specs (blind zone, FOV, weight, power): [Livox Mid-360 Specs](https://www.livoxtech.com/mid-360/specs)
- Unitree G1 + Mid-360 accessory (weight/cost context): [Unitree G1 Mid360 LiDAR (G1-MID360L) Specs](https://www.robotsinternational.com/Unitree-G1-MID360L-G1-Mid360-Lidar.htm); [Unitree G1 specs](https://www.unitree.com/g1/)

### Repo facts used
- Camera geometry: `EGO_FOVY=90` (native), grounding path uses `FOVY=45°`, `CAM_PITCH=32°`, `GROUNDING_PITCH=26°` — `code/arena.py`
- Depth floor: `MIN_DEPTH_M=0.60` — `code/grounding.py`
- Hold-goal dead-reckoning: `HOLD_GOAL_HORIZON=100`, `_last_known_goal`, `_goal_ema` (α=0.4) — `code/inferencer.py`
- Documented failure taxonomy (wall-HSV collision, distractors, scan range): `docs/grounding_dist.md`
