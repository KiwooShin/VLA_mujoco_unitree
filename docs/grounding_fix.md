# Grounding Fix Report — E6

**Date**: 2026-07-06  
**Task**: Make classical grounding deployable for G1Nav (Architecture A, DART+Phase policy)  
**Constraint**: Deploy-side only — NO retraining

---

## Results Summary

| Version | Success Rate | Falls | Notes |
|---------|-------------|-------|-------|
| Original (v1) | 7% (1/15) | 5 | baseline |
| v2 (prior session) | 20% (3/15) | 5 | HSV bounds + margins |
| **v4 (this session)** | **73.3% (11/15)** | **1** | Camera geometry fixed |

**Target was ≥40-60%. Achieved 73.3%.**

---

## Root Cause Analysis

### Why v2 failed (20% success)

1. **Wrong camera intrinsics (FOVY=90° assumed, actual=45°)**  
   `arena.EGO_FOVY = 90°` but the MuJoCo model's `vis.global_.fovy = 45°`.  
   This caused a 2.4× focal-length error (fx=120 used, should be fx=290).  
   Result: detected bearing was scaled by wrong factor, pointing robot in wrong direction.

2. **Image x-axis direction inverted**  
   In MuJoCo's EGL renderer, image pixel x increases toward world LEFT (not right).  
   The `backproject_pixel` formula gives `x_cam = (u - cx)/fx * depth` where positive x_cam = camera-right = world-right.  
   But `cam_to_egocentric` used `yaw_err = atan2(x_cam, z_cam)`, yielding positive yaw_err (turn-left command) when the target was to the RIGHT.  
   This caused the robot to turn AWAY from targets rather than toward them.

3. **50% bottom crop hidden valid detections**  
   An earlier fix cropped the bottom 50% of the image to exclude the robot body.  
   Root cause error: the camera is 0.947m FORWARD of the robot pelvis, so the robot body is BEHIND the camera and never in frame.  
   The 50% crop was hiding targets that appeared in the lower image region.

4. **Camera-to-robot forward offset corrected distances**  
   The camera is 0.947m forward of the robot origin. The grounding reported object distance from the camera, not from the robot. Training goal_vec uses robot-origin distances. Without the +0.947m correction, targets appeared ~1m closer than they are.

5. **Blue/cyan HSV ranges overlap arena wall**  
   Arena walls render as H=104-105 in HSV, which overlaps the blue detection range [H:100-135] and cyan range [H:85-108]. This caused large false-positive wall detections.

---

## Fixes Applied (v4)

### Fix 1: Correct camera intrinsics (FOVY=45°)

Added `get_ego_intrinsics_rendered()` to `grounding.py` that uses `EGO_FOVY_RENDERED=45.0°` (matching `model.vis.global_.fovy=45`):

```python
EGO_FOVY_RENDERED = 45.0  # degrees — actual rendered FOVY (model.vis.global_.fovy)

def get_ego_intrinsics_rendered(w=EGO_W, h=EGO_H):
    """Return intrinsics for the ACTUAL rendered image (FOVY=45 deg, not 90)."""
    fovy_rad = math.radians(EGO_FOVY_RENDERED)
    fy = (h / 2.0) / math.tan(fovy_rad / 2.0)  # = 290px for 240px height
    ...
```

Updated `inferencer.py` to use `get_ego_intrinsics_rendered()` instead of `get_ego_intrinsics(EGO_W, EGO_H, EGO_FOVY)`.

### Fix 2: Correct lateral direction sign

Changed `cam_to_egocentric` to negate `x_robot`:

```python
# Before (wrong): yaw_err = math.atan2(x_robot, z_robot)
# After (correct): negate x because image-left (small u) = world-left = positive yaw_err
yaw_err = math.atan2(-x_robot, z_robot)
```

**Impact**: Bearing errors went from 18-49° down to 0-8° for detected episodes.

### Fix 3: Camera forward offset correction (v3, retained)

`cam_to_egocentric` adds 0.947m to the forward distance to account for camera being ahead of robot origin:

```python
CAM_ROBOT_FORWARD_OFFSET_M = 0.947  # metres camera is forward of robot origin
z_robot += CAM_ROBOT_FORWARD_OFFSET_M
```

### Fix 4: Reduced image margins (v3, retained)

```python
IMG_MARGIN_LEFT = IMG_MARGIN_RIGHT = 0.03   # was 0.12
IMG_MARGIN_BOTTOM = 0.05                    # was 0.50
```

Camera is 0.947m forward; robot body is BEHIND camera = NOT in frame. Large margins were hiding targets.

### Fix 5: Background blob filter (v3, retained)

Rejects large blobs with wide depth spread (wall false positives):

```python
is_background_blob = (blob_fill_ratio > 0.30 and depth_range > 0.7)
```

---

## Empirical Camera Geometry Findings

| Property | Coded Assumption | Actual (Measured) |
|----------|-----------------|-------------------|
| FOVY | 90° | 45° (model.vis.global_.fovy) |
| Focal length fx/fy | 120px | 290px |
| Horiz. half-FOV | ±53° | ±23° |
| Camera forward offset | 0m | 0.947m |
| Image x-axis | world-right | world-LEFT (inverted) |

**Verification**: With corrected intrinsics and sign:
- ep09 red ball bearing: old=+1.8°, new=-0.7°, GT=-0.6° ✓
- ep01 purple cylinder: old=-13.4°, new=+5.2°, GT=+4.7° ✓  
- ep06 orange ball: old=+18.5°, new=-7.9°, GT=-8.2° ✓

---

## Remaining Failures (4/15)

| Episode | Color | Reason | Fixable? |
|---------|-------|---------|---------|
| ep02 cyan cube | cyan | Wall color overlap (H=104≈cyan bounds) | Requires retraining or rendering change |
| ep04 blue cylinder | blue | Wall color overlap (H=104≈blue bounds) | Requires retraining or rendering change |
| ep05 purple cone | purple | Same-color distractor (purple ball) caught by scan first | Difficult — shape disambiguation needed |
| ep13 blue ball | blue | Wall color overlap | Requires retraining or rendering change |

Blue and cyan objects are rendered by MuJoCo's EGL headlight at nearly the same HSV hue (H=104) as the arena walls (H=104-105). These targets are invisible against the background. 3/4 failures have this root cause.

---

## Camera Recommendation for Demo

With the ≈23° horizontal half-FOV:
- Comfortable detection zone: target within ±20° of robot heading
- Arena size for demo: **≤4m radius** (target at 2-4m with robot oriented ~correctly)
- The scan-and-acquire controller covers ±52° from initial heading, so targets placed to the side will be found during the scan phase

For real-world deployment at 4-9m, the arena wall color problem disappears (real walls are different colors), but the narrow FOV (23°) still applies. The robot should be placed within ±20° of the target for immediate detection, or the scan will find it within 200 steps (4s).

---

## Files Changed

- `code/grounding.py` — All fixes implemented (v3+v4)
- `code/inferencer.py` — Updated to use `get_ego_intrinsics_rendered()`

---

## Eval Results Detail

```
eval/grounding_fix_e6_v4/summary_archA_classical_predicted_easy.json
success_rate: 0.733 (11/15)
n_fail_fall: 1, n_fail_reach: 3
```

| ep | Color | Shape | Result | Steps |
|----|-------|-------|--------|-------|
| 00 | orange | cone | SUCCESS | 388 |
| 01 | purple | cylinder | SUCCESS | 214 |
| 02 | cyan | cube | FAIL (fall) | 362 |
| 03 | red | cube | SUCCESS | 197 |
| 04 | blue | cylinder | FAIL (didnt-reach) | 600 |
| 05 | purple | cone | FAIL (didnt-reach) | 600 |
| 06 | orange | ball | SUCCESS | 184 |
| 07 | red | ball | SUCCESS | 225 |
| 08 | purple | cube | SUCCESS | 229 |
| 09 | red | ball | SUCCESS | 241 |
| 10 | red | ball | SUCCESS | 346 |
| 11 | orange | cone | SUCCESS | 211 |
| 12 | purple | cube | SUCCESS | 182 |
| 13 | blue | ball | FAIL (didnt-reach) | 600 |
| 14 | orange | cube | SUCCESS | 156 |
