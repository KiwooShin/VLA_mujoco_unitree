# Maneuver Skill — Camera-Gate Re-Check + Reproduction Retrain (CX-4)

**Date:** 2026-07-09
**Agent:** CX-4
**Scope:** (1) Re-gate the DEPLOYED `checkpoint/maneuver_best.pt` under the current
camera code (CAM-2 adopted; `code/arena.py` / `code/grounding.py` / `code/inferencer.py`
changed by docs/cam_p0.md, cam_p1.md, cam_p2.md) — the camera experiment's no-regression
gates covered easy/demo/search but never maneuver. (2) Reproduce the maneuver fine-tune
stage from the REPRO2 goto pipeline (an earlier reproduction log verified goto from scratch but
never the maneuver stage).

**Protocol (both parts):** `code/eval_maneuver.py`, seed=999, n=15, hybrid-vel
(TF vel during TURN_PHASE only — the released protocol, docs/maneuver.md §11),
`--render-n 0`, device=cpu. Baseline = **73.3% (11/15)** = `eval/stable_maneuver/`
(S10 "stable numbers", 2026-07-06; multi-seed context: 66.7%
mean [46.7–73.3], "NOTABLY VARIABLE", `docs/robustness.md`).

---

## TL;DR

| Question | Answer |
|---|---|
| Did the camera changes regress maneuver? | **NO** — no camera-attributable regression. New-code runs: 66.7 / 66.7 / 73.3%; old-code (released) runs: 73.3 (historical) / 80.0 / 60.0%. Overlapping bands; per-arm means differ by 1 episode in 45. Physics proven **bit-identical** old-vs-new (see §2c). CAM-2 gate stays CLOSED. |
| Did the maneuver fine-tune reproduce? | **YES** — `runs/REPRO_maneuver_A/epoch_0003.pt` = **73.3% (11/15)**, confirmed in two independent eval runs (bit-identical trajectories), exactly matching the released deployed number. Behavioral peak at epoch 3 (released run peaked at epoch 2), selected by closed-loop success per protocol. |

---

## 1. Setup notes / corrections

- **Deployed checkpoint identity:** `checkpoint/maneuver_best.pt` = `runs/maneuver_A/epoch_0002.pt`
  (pinned per docs/finalize.md), proprio_dim=62, task=maneuver.
- **README flag drift (same class found for goto in an earlier reproduction log):** README's
  maneuver commands use `--maneuver-data/--loco-data/--loco-ckpt/--cuda/--ckpt/--n-episodes`;
  the actual argparse is `--data <repo1> <repo2 ...>` (nargs+), `--resume-ckpt`,
  `--device cuda`, `--checkpoint`, `--n`.
- **Base-checkpoint provenance:** README documents fine-tuning from
  `runs/demo_dart_A/epoch_0003.pt`, but the actual historical run (logs/maneuver_train.log)
  resumed from `runs/demo_dart_A/epoch_0010.pt`. Both sit in the "ep3–10 all ≈80% demo/GT"
  band (E7/docs/demo_dart.md). This reproduction followed the README/dispatch:
  resumed from `runs/REPRO2_demo_dart_A/epoch_0003.pt` (the REPRO2 campaign's
  behavioral-best goto checkpoint, 80% demo/GT per eval/R2_e03).
- **`--hybrid-vel` is now default-on** in eval_maneuver.py (flag kept as deprecated no-op);
  passed explicitly anyway.
- CX-5 was editing `code/inferencer.py` concurrently — verified **irrelevant**:
  `eval_maneuver.py`'s import chain never touches inferencer.py or grounding.py, and
  inferencer.py's mtime was unchanged across the re-gate run anyway.

---

## 2. Part 1 — Re-gate of DEPLOYED maneuver_best.pt under current camera code

### 2a. Results (all runs, deployed checkpoint, seed 999, n=15, hybrid-vel)

| Run | Camera code | Threads | Success | Falls | Notes |
|---|---|---|---|---|---|
| `eval/stable_maneuver` (S10 baseline, 2026-07-06) | released (pre-camera) | (historical) | **11/15 = 73.3%** | 1 | the gate baseline |
| `eval/REPRO_maneuver_regate_deployed` | **current (CAM-2)** | default | 10/15 = 66.7% | 2 | run 1 |
| `eval/REPRO_maneuver_regate_deployed_run2` | **current (CAM-2)** | default | 10/15 = 66.7% | 1 | run 2 — different per-episode pattern than run 1 |
| A/B: scratchpad `ab_oldcode_n15` | released (zip overlay) | default | 12/15 = 80.0% | 1 | old code, same CPU/env as run 1/2 |
| A/B: scratchpad `ab_newcode_1t` | **current (CAM-2)** | OMP=1 | **11/15 = 73.3%** | 1 | equals baseline exactly |
| A/B: scratchpad `ab_oldcode_1t` | released (zip overlay) | OMP=1 | 9/15 = 60.0% | 2 | old code's low tail |

Old-code arm: {73.3, 80.0, 60.0} → mean 71.1%. New-code arm: {66.7, 66.7, 73.3} → mean
68.9%. Difference = 1 episode in 45. The direction even *reverses* between the
multithreaded pair (old 80.0 > new 66.7) and the single-threaded pair (new 73.3 > old 60.0).

### 2b. Per-episode structure of the variance

Episode-level comparison across all six runs shows a fixed fingerprint:

- **Solid successes in every run:** eps 2,4,5,6,7,8,10,12 (8 episodes; ep3 succeeded in
  5/6 runs — its one miss was under **old** code).
- **Consistent failures in every run:** ep9 (+46..+51° — turns before the landmark or
  stops just past threshold), ep14 (no_landmark, −81..−88°).
- **Borderline coin-flips at the 25° success threshold:** ep0 (23.0/26.2/23.8/22.2/23.5/25.5°)
  and ep1 (25.6/22.2/23.0/26.7/25.2/22.8°) — flip in BOTH directions in BOTH code arms,
  including between two runs of the *same* code.
- **Marginal-stability episodes:** ep11 falls in 4/6 runs (both arms; survived once per arm),
  ep13 falls in 3/6 runs (twice new-code, once old-code single-thread — and *succeeded*
  under new-code single-thread). Neither is code-correlated.

The runs are not trajectory-reproducible under load: two runs of the *identical* new code
gave different per-episode outcomes (run 1: ep9 no_landmark, ep11 fall; run 2: ep9/ep11
wrong_heading at +25.4/+26.9°, no ep11 fall). Root cause of the jitter is float-level
nondeterminism in the torch CPU policy loop under varying machine load (training + other
agents' evals ran concurrently), amplified by 1400-step closed-loop chaos — the same class
of run-to-run variance already documented for this skill (docs/robustness.md: seed-999
73.3% but 46.7–73.3 across seeds; V5's EGL nondeterminism finding).

### 2c. Camera-attributability: ruled out structurally AND empirically

1. **No causal path.** With `--render-n 0`, eval_maneuver never constructs an
   `ArenaRenderer` (vision runs on a cached zero-image TinyViT embedding; goal/vel are
   GT-privileged). The only camera-experiment surface in the eval's path is `build_arena()`.
2. **The arena diff is physics-inert.** Full diff vs the released zip: new constants,
   `CAMERA_MODE` toggle (no-op at the default `cam2`), offscreen buffer sizing (computes
   to the same 640×480), `spec.visual.global_.fovy` override only in `widefov` mode, and
   `cam.distance` 0.001→1.0 (positions the *render* camera eye only). No body/joint/geom
   change; the ego/grounding/proximity cameras are runtime `mjCAMERA_FREE` objects, not
   model elements.
3. **Bit-identical physics (measured).** For the ep13 scene (seed 999), old and new arena
   code compile to identical models (nq=36, nv=35, nbody=45, ncam=0, ngeom=109) and a
   200-step `mj_step` rollout gives the **same MD5 over (qpos, qvel)**:
   `37f9c7d9fd6a3150ba27c27a626c8e4d` both arms.
4. `eval_maneuver.py` itself is byte-identical to the released zip version.

### 2d. Verdict (Part 1)

**No regression attributable to the camera changes. The CAM-2 adoption gate does not
reopen.** The honest statement of the deployed maneuver number under the current code is
**66.7–73.3% observed across three runs (10–11/15)** against a baseline whose own
identical-code re-runs span 60.0–80.0% — i.e. the baseline 73.3% is the center of a ±1–2
episode noise band, and the new-code runs sit inside it.

---

## 3. Part 2 — Reproduction retrain of the maneuver fine-tune

### 3a. Training

Matched the released hyperparameters (logs/maneuver_train.log — actual history, not the
README's drifted flags): mixed data `dataset/maneuver` + `dataset/dart_combined_v2`
(571 eps = 143 maneuver + 428 locomotion; 346,320 train / 52,531 val frames),
**epochs=10, batch=128, lr=5e-5**, swing_weight=2.0, GRU proprio expansion 57→62-d
(old weights preserved, new cols orthogonal-init), device=cuda.

```bash
PYTHONPATH=. MUJOCO_GL=egl python code/train_maneuver.py \
    --data dataset/maneuver dataset/dart_combined_v2 \
    --resume-ckpt runs/REPRO2_demo_dart_A/epoch_0003.pt \
    --out runs/REPRO_maneuver_A \
    --epochs 10 --batch 128 --lr 5e-5 --device cuda
```

Log: `logs/REPRO_maneuver_train.log`. Val-loss decreased monotonically
(0.0766 → 0.0588 by epoch 10) — as in the released run, **val loss does not track
behavioral success** (see sweep below), so selection is by closed-loop eval, never
val-loss (README warning; docs/maneuver.md §11).

### 3b. Closed-loop epoch sweep (seed 999, n=15, hybrid-vel)

| Epoch | val_act | Success | Falls | Notes |
|---|---|---|---|---|
| 1 | 0.0766 | 5/15 = 33.3% | 1 | |
| 2 | 0.0752 | 7/15 = 46.7% | 2 | (released run peaked here at 80/73.3%) |
| **3** | 0.0730 | **11/15 = 73.3%** | 1 | **behavioral peak — deployment-equivalent** |
| 3 (re-run) | — | **11/15 = 73.3%** | 1 | bit-identical per-episode confirmation |
| 4 | 0.0697 | 9/15 = 60.0% | 0 | |
| 5 | 0.0676 | 6/15 = 40.0% | 0 | |
| 6 | 0.0660 | 10/15 = 66.7% | 0 | |
| 7 | 0.0652 | 6/15 = 40.0% | 1 | |
| 8 | 0.0629 | 7/15 = 46.7% | 0 | |
| 9 | 0.0623 | 8/15 = 53.3% | 0 | |
| 10 | 0.0622 | 7/15 = 46.7% | 1 | val-loss minimum — would be "model_best" by val-loss, 26.6pp below the behavioral peak |

The sweep reproduces the released run's characteristic shape — a sharp early behavioral
peak followed by decline while val-loss keeps improving (released: e1 0% → e2 80% →
e3 66.7% → e4/e5 50%; repro: e1 33% → e3 73.3% → e5 40%). The peak lands at epoch 3
instead of epoch 2, which is expected variation given (a) the different base checkpoint
(REPRO2 epoch_0003 vs historical demo_dart_A epoch_0010) and (b) the documented
variability of this skill; the *selection protocol* (sweep epochs, pick by closed-loop)
is what reproduces, and it recovers the released number exactly.

Per-episode fingerprint of REPRO epoch 3 (both runs identical): fails ep3 (no_landmark
−89.7°), ep9 (no_landmark +51.0°), ep11 (wrong_heading +37.8°), ep13 (fall) — and
*succeeds* on ep14, which the deployed model fails in every run. Different model, different
failure set, same 73.3% headline: consistent with reproduction-within-noise of a
NOTABLY-VARIABLE skill, not with a copy of the original weights.

### 3c. Verdict (Part 2)

**REPRODUCED (within noise).** Best epoch = `runs/REPRO_maneuver_A/epoch_0003.pt` at
**73.3% (11/15)**, doubly confirmed, equal to the released deployed 73.3% and inside the
released multi-seed band (66.7% [46.7–73.3]). The full pipeline
(scratch goto → demo-DART fine-tune → maneuver fine-tune) is now verified end-to-end.

---

## 4. Artifacts

| Path | What |
|---|---|
| `runs/REPRO_maneuver_A/` | Reproduction fine-tune checkpoints (10 epochs) + curves.json |
| `logs/REPRO_maneuver_train.log` | Training log |
| `eval/REPRO_maneuver_regate_deployed/`, `..._run2/` | Deployed-ckpt re-gate runs (new camera code) |
| `eval/REPRO_maneuver_sweep/e01..e10/`, `e03_run2/` | Reproduction epoch sweep |
| `logs/REPRO_maneuver_regate_*.log`, `logs/REPRO_maneuver_sweep_*.log` | Eval logs |
| `logs/REPRO_maneuver_ab_*.log` | Old-vs-new camera-code A/B eval logs (old code from released zip, run via PYTHONPATH overlay; outputs in session scratchpad `ab_*/`) |
