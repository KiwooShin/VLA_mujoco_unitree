# Experiment Ledger — Index

This folder is the project's experiment ledger: a chronological, per-mechanism
record of what was tried, adopted, or rejected while building the G1Nav VLA
stack, plus the closed-loop evidence behind each call. Every code comment in
`code/*.py` of the form `docs/<name>.md` cites one of the files below —
they are the primary source for *why* the shipped system looks the way it
does, not just *what* it does. Verdicts follow one convention throughout:
**ADOPT** (shipped, gated by a no-regression check), **REJECT** (tried,
falsified, and reverted), **diagnosis/measure** (an eval or analysis pass
that didn't itself change behavior), or **CLOSE** (an issue root-caused and
consciously left unfixed, with the reasoning recorded).

Numbers throughout are seed-999, n=15 closed-loop success rates on the
`easy`/`demo`/`search`/`maneuver` presets unless stated otherwise. See the
top-level `README.md` for the current headline results.

## 1. Locomotion & stability

- `architecture_decision.md` — ADR-001: the modular-VLA architecture decision (grounding → egocentric goal → velocity → distilled joint policy) and why an end-to-end alternative was passed over.
- `bakeoff.md` — First closed-loop bake-off across 3 grounding conditions: 0/15 everywhere, isolating the real bottleneck to locomotion stability, not grounding.
- `gaitfix.md` — Root-causes gait collapse to mean-regression on absolute joint targets; residual/normalized action targets revive oscillation (ADOPT).
- `dart_phase.md` — DART recovery data + gait-phase input: 0% → 100% easy/ground-truth-goal, zero falls (ADOPT).
- `demo_dart.md` — Demo-distribution DART (all robot start-yaws): fixes the yaw covariate-shift gap, 7% → 80% on demo/ground-truth (ADOPT).
- `finalize.md` — Assembles the first WBC-free deployable checkpoint set and baseline showcase videos.
- `maneuver.md` — Adds the turn-after-landmark skill: 80% (12/15) on its own procedural eval (ADOPT).
- `repro_maneuver.md` — Re-gates the deployed maneuver checkpoint under the later camera stack and reproduces its training pipeline from scratch end-to-end (measure/CONFIRM: camera-independent, no regression).
- `rot_dart.md` — Rotation-recovery fine-tune: fixes search-skill falls but regresses demo/easy on the shared model (REJECT — first demonstration that shared-model retrains can regress unrelated skills).
- `vel_proprio.md` — Proprioception-fed velocity head: regresses goto to 0% (REJECT — second demonstration of the shared-model-regression risk).

## 2. Vision / grounding

- `vision_grounding.md` — First learned in-model grounding head: bearing MAE far worse than the classical baseline (REJECT — motivates staying classical for now).
- `grounding_dist.md` — Dedicated grounding camera + goal smoothing, then depth-foreground and wall-stripe filters: demo/classical 6.7% → 46.7%, easy 80% → 93% (ADOPT).
- `grounding_fix.md` — Fixes two deploy-side camera bugs (FOV-Y, x-sign) blocking classical grounding from deploying at all: easy/classical 7% → 73% (ADOPT).

## 3. Camera-visibility option space

- `cam_opt1_widefov.md` — Design brief: single wide-FOV camera option for near-field visibility.
- `cam_opt2_multicam.md` — Design brief: dedicated second camera with a handoff — becomes the adopted CAM-2 design.
- `cam_opt3_activetilt.md` — Design brief: active pan-tilt camera — feasible but a fixed dual-camera setup covers the same range more simply; not built.
- `cam_opt4_lidar.md` — Design brief: adding a LiDAR sensor — would help close-range precision but not the arena's dominant far-range identification failures; not built.
- `cam_p0.md` — Phase 0: fixes two camera-geometry bugs (eye-position drift, hardcoded pitch offset): easy 93→100%, demo 60→66.7% (ADOPT).
- `cam_p1.md` — Phase 1: adds the CAM-2 proximity camera + Schmitt-trigger handoff: visible-to-stop range 0.7m → 0.256m, zero regression (ADOPT).
- `cam_p2.md` — Phase 2: A/B tests a single wide-FOV camera against CAM-2 — loses on every axis and is twice as slow (REJECT, reported with full honest numbers).
- `cam_p3_demo.md` — Finds and fixes a plausibility-gate deadlock in the interactive demo's camera-switching logic (ADOPT, demo-file fix).
- `cam_p4_gate.md` — Ports the CX-3 deadlock fix into the gated evaluation path: re-gate matches the champion numbers exactly (ADOPT).

## 4. Lock-management & failure analysis

- `fa1_failures.md` — Deep-dive on the 5 demo + 3 search residual failures: corrects the record on how many are genuinely "wall-hue" vs. other mechanisms.
- `fa2_residuals.md` — Re-diagnoses the 2 residual demo failures (ep2/ep4) under the fully-adopted default stack: both are pure scan-coverage misses, not occlusion (feeds directly into the NX-10 scan fix).
- `rs1_lock_mgmt.md` — Design brief proposing 5 lock-management hardening mechanisms (M1–M5+), ranked by which residual episode each targets.
- `nx1_scan.md` — Bounded bidirectional scan for the search skill: 80% → 93.3%, falls 3→0 (ADOPT).
- `nx2_final.md` / `nx2_impl.md` / `nx2_iso.md` — Lock-management hygiene (area-quality floor + innovation gate): zero regressions, zero fixes on their own, but safe to keep (ADOPT as hygiene).
- `nx3_size_gate.md` — Physical-size plausibility gate for lock selection: breaks two passing episodes (REJECT).
- `nx4_depth_split.md` — Depth-guided blob splitting and re-selection: merged blobs turn out to be a depth continuum, not splittable (REJECT).
- `nx5_coherence.md` — Odometry-coherence watchdog: falsified before gating — the incoherence signal is anti-correlated with the actual failure (REJECT). Synthesizes why 5 different classical lock-management axes all fail for the same underlying reason (bearing-correct-but-depth-wrong detections), motivating the pivot to a learned detector.

## 5. Learned detector (GROUND_NET)

- `nx6_data.md` — Dataset generation for the learned heatmap/CenterNet detector training.
- `nx6_train_heatmap.md` — Trains the heatmap-variant detector (the one ultimately adopted).
- `nx6_train_centernet.md` — Trains a CenterNet-style alternative for comparison; the heatmap variant wins.
- `nx6_judge.md` — Offline judging/selection methodology between the trained detector variants.
- `nx6_final.md` — Integrates the learned heatmap detector: demo 66.7% → 80%, but breaks one previously-passing episode (REJECT net, pending the fix in §6).
- `nx7_adoption.md` — Attempts a hysteresis/coast-rescan fix for the one broken episode: doesn't fix it, but re-diagnoses it as a locomotion stall rather than a grounding gap (REJECT, diagnosis).
- `nx14_detector_v2.md` — Detector v2: hard-negative + far-range oversampling, full 60-epoch convergence. Beats v1 on every offline axis with zero closed-loop regressions across 5 gate lines (ADOPT). Also documents and resolves an operational checkpoint-overwrite incident transparently.

## 6. Obstacle avoidance

- `nx8_stall.md` — STALL_BREAK watchdog attempt for the broken episode: doesn't fix it, but geometrically proves the episode is a physical collision with a scene object (REJECT, diagnosis).
- `nx9_avoid.md` — Local obstacle avoidance (depth-corridor repulsion): fixes the collision episode *and* an unrelated search-skill collision in one pass — clears the adoption bar for both avoidance and the learned detector together: demo 66.7% → 86.7%, search 93.3% → 100% (ADOPT).
- `nx11_ep4.md` — Closes out the one remaining demo failure: finds, fixes, and reverts a related self-body bug (it doesn't flip the outcome); the failure is root-caused as a policy-level balance limit and consciously left as an open item (CLOSE).
- `nx13_avoid_hygiene.md` — Re-evaluates NX-11's reverted self-body fix on its own merits: zero regressions across 5 gate lines (ADOPT).

## 7. Scan / search refinement

- `nx10_scan_fix.md` — Fixes the initial scan's realized-yaw coverage bug (commanded ±90° only physically realized ~±62°): demo 86.7% → 93.3% (ADOPT).
- `nx12_turn_dwell.md` — Finer-grained scan-dwell fix attempt for a fresh-seed rotation instability: delays but never eliminates the fall, and costs step margin elsewhere (CLOSE/revert).

## 8. Generalization & validation

- `gen1_multiseed.md` — Full adopted stack on 2 fresh scene seeds / 90 episodes: every adopted mechanism generalizes cleanly; the one real number movement traces to an already-documented open risk (measure).
- `robustness.md` — Multi-seed robustness pass (4 seeds) on the pre-camera-era stack, establishing the original stable-numbers baseline band.

## 9. Demo reliability & showcase

- `fs1_first_scene.md` — Fixes the interactive demo's very first camera draw (a one-time visual glitch on cold start).
- `dr1_demo_reliability.md` — Wild-usage reliability sweep on the interactive demo: 90% success with zero falls/crashes across the color×shape matrix, but finds the typed instruction is cosmetic in the reachable demo path (a real UX/logic gap, fixed next).
- `nx15_live_parse.md` — Fixes the issue DR-1 found: wires the interactive demo's live instruction parsing to actually select the target instead of using a fixed scene-config index.
- `nx16_cone_stall.md` — Diagnoses and fixes a near-target stall specific to cone-shaped targets under the learned detector (confidence decay at close range).
- `showcase.md` — Builds and verifies the showcase reel and web UI end-to-end (all endpoints live-checked).
- `vr1_rehearsal.md` — Final verification rehearsal: follows the public README top-to-bottom as a fresh user would, confirming the documented setup and reproduction steps actually work.

---

A handful of internal planning notes and process logs from development are
not included here, since they carry no evidence value beyond what's already
captured in the docs above.

## Demo presentation era (added after initial curation)
- [vf1_showpiece.md](vf1_showpiece.md) — renderer upgrade (heatmap overlay, HUD, cards) with behavior-invariance check.
- [vf3_bev_fixes.md](vf3_bev_fixes.md) — BEV projection-convention bug diagnosis and pixel-level verification.
- [vf5_cam_objects.md](vf5_cam_objects.md) — lower BEV camera + 7-object scene samplers; first-scene re-curation.
