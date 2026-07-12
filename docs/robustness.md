# Multi-Seed Robustness Evaluation

**Date:** 2026-07-07
**Experiment:** EVAL2 (robustness eval)
**Method:** 3 additional held-out seeds (1234, 2025, 77) + original (999), n=15 each.
**Fix applied:** EGL per-episode renderer reset (gc.collect + cuda.synchronize between episodes; already in eval_closedloop.py from S10 fix).
**Checkpoints:** checkpoint/goto_best.pt (easy/demo/search), checkpoint/maneuver_best.pt (maneuver).
**All WBC-free:** keyframe init + student scan. --device cuda.

---

## Raw Per-Seed Results

### Goto Easy/Classical

| seed | n_success | n | rate |
|------|-----------|---|------|
| 999 (baseline) | 14 | 15 | 93.3% |
| 1234 | 13 | 15 | 86.7% |
| 2025 | 14 | 15 | 93.3% |
| 77 | 15 | 15 | 100.0% |

**MEAN: 93.3% | RANGE: [86.7%, 100.0%]**

### Goto Demo/Classical

| seed | n_success | n | rate |
|------|-----------|---|------|
| 999 (baseline) | 9 | 15 | 60.0% |
| 1234 | 9 | 15 | 60.0% |
| 2025 | 12 | 15 | 80.0% |
| 77 | 11 | 15 | 73.3% |

**MEAN: 68.3% | RANGE: [60.0%, 80.0%]**

### Search Out-of-FOV

| seed | n_success | n | spot_rate | reach_rate |
|------|-----------|---|-----------|------------|
| 999 (baseline) | 12 | 15 | 93.3% | 80.0% |
| 1234 | 11 | 15 | 93.3% | 73.3% |
| 2025 | 13 | 15 | 93.3% | 86.7% |
| 77 | 9 | 15 | 86.7% | 60.0% |

**MEAN: 75.0% | RANGE: [60.0%, 86.7%]** (SPOT-rate: 86.7–93.3%, stable)

### Maneuver

| seed | n_success | n | rate | n_fall |
|------|-----------|---|------|--------|
| 999 (baseline) | 11 | 15 | 73.3% | 0 |
| 1234 | 11 | 15 | 73.3% | 0 |
| 2025 | 11 | 15 | 73.3% | 0 |
| 77 | 7 | 15 | 46.7% | 0 |

**MEAN: 66.7% | RANGE: [46.7%, 73.3%]**

---

## Summary Table

| Skill | Seed 999 (reported) | Mean ± Range | Verdict |
|-------|--------------------|--------------|---------| 
| Goto easy/classical | 93.3% | 93.3% [86.7–100%] | SOLID |
| Goto demo/classical | 60.0% | 68.3% [60.0–80.0%] | CONSERVATIVE |
| Search out-of-FOV | 80.0% | 75.0% [60.0–86.7%] | SLIGHTLY LUCKY |
| Maneuver | 73.3% | 66.7% [46.7–73.3%] | NOTABLY VARIABLE |

---

## Analysis

### Goto Easy/Classical (93.3% mean — SOLID)
The single-seed number (93.3%) matches the mean exactly and sits in the center of the range. 
The failure mode is consistent: cyan/blue HSV wall collision at close range, identical across 
seeds. This is the most robust skill — the 86.7%–100% band reflects scene color sampling 
variance (seed 77 drew no blue/cyan scenes; seed 1234 drew 2 extra cyan).

### Goto Demo/Classical (68.3% mean — ORIGINAL WAS CONSERVATIVE)
The reported 60.0% was slightly pessimistic: multi-seed mean is 68.3%, with a range of 
[60.0%, 80.0%]. The 60.0% figure holds for seeds 999 and 1234 (which happened to draw 
~6–7 cyan/blue targets). Seed 2025 (12/15=80%) and seed 77 (11/15=73.3%) drew fewer 
cyan/blue scenes. The core finding (demo-distance goto works reliably for non-cyan/blue 
targets ~78–87%) is stable; variance is almost entirely explained by how many cyan/blue 
objects the scene sampler draws. If 7/15 scenes are cyan/blue → ~60%; if 3–4 → ~73–80%.

### Search Out-of-FOV (75.0% mean — SEED 999 SLIGHTLY LUCKY)
The single-seed 80.0% is above the multi-seed mean of 75.0%. Seed 77 pulls the mean 
down to 60.0% — 6 failures, driven by a higher proportion of long-scan episodes (bearing 
60–90° from initial heading requires nearly full 360° rotation). SPOT-rate is more stable 
(86.7–93.3%) — the real variance is in the REACH phase after spotting, where long-scan 
covariate shift causes falls. The search skill remains above 60% even in the worst seed.

### Maneuver (66.7% mean — MOST VARIABLE, SEED 77 NOTABLY LOW)
This is the most variable skill. Seeds 999/1234/2025 all give exactly 73.3% (11/15). 
Seed 77 drops to 46.7% (7/15) with 4 no_landmark failures: the robot turns before 
reaching the landmark. Inspection of seed 77's scenes shows the same scene distribution 
(landmark distances 3.4–5.4m), so this is not a scene-difficulty artifact — it reflects 
genuine behavioral variance of the current maneuver model (which uses privileged GT vel 
teacher-forcing during TURN_PHASE and has no proprio in the velocity head, making it 
sensitive to specific scene layouts and initial conditions). 

The 46.7% is not a systematic floor — seeds 1234 and 2025 confirm 73.3% is reproducible. 
However, reporting 73.3% as the single-number should be accompanied by a ±range note.

---

## Recommended Reported Numbers

For the paper/report, use the multi-seed mean with the range in brackets:

| Skill | Reported (4-seed) |
|-------|------------------|
| Goto easy/classical | **93.3%** [86.7–100%] |
| Goto demo/classical | **68.3%** [60.0–80.0%] |
| Search out-of-FOV | **75.0%** [60.0–86.7%] |
| Maneuver | **66.7%** [46.7–73.3%] |

Alternatively, since seed 999 was the original held-out seed, report it as the headline 
with ± range from robustness sweep:

| Skill | Seed 999 (headline) | ± range |
|-------|--------------------|---------| 
| Goto easy/classical | 93.3% | ±6.7pp |
| Goto demo/classical | 60.0% | +20 / 0 pp |
| Search out-of-FOV | 80.0% | +6.7 / −20 pp |
| Maneuver | 73.3% | 0 / −26.6 pp |

---

## Eval Infrastructure Notes

- `eval_closedloop.py`: added `--seed` argument (default=999) to support held-out seed sweep.
- `eval_search.py`: added `--seed` argument (default=999).
- `eval_maneuver.py`: already supported `--seed` (unchanged).
- EGL fix: per-episode `gc.collect()` + `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` 
  already present in eval_closedloop.py from S10. Confirmed working (5/5 reproducibility 
  previously verified via verify_egl_repro.py). Search and maneuver eval use single 
  renderer objects per episode (ArenaRenderer init/close per episode).
- All evals ran at `--device cuda` with `--no-render` (no video output, grounding renders 
  still execute for classical conditions).
- Wall time per seed: easy≈2min, demo≈6.5min, search≈7.5min, maneuver≈0.5min.

---

## Files

| Path | Content |
|------|---------|
| `eval/robustness/seed_1234/{easy,demo,search,maneuver}/` | Seed 1234 results |
| `eval/robustness/seed_2025/{easy,demo,search,maneuver}/` | Seed 2025 results |
| `eval/robustness/seed_77/{easy,demo,search,maneuver}/` | Seed 77 results |
| `eval/robustness/seed_*/easy/summary_archA_classical_predicted_easy.json` | Per-seed goto easy |
| `eval/robustness/seed_*/demo/summary_archA_classical_predicted_demo.json` | Per-seed goto demo |
| `eval/robustness/seed_*/search/summary.json` | Per-seed search |
| `eval/robustness/seed_*/maneuver/summary.json` | Per-seed maneuver |
