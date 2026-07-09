# VLA_mujoco_unitree — Unitree G1 Humanoid VLA in MuJoCo

A small **Vision-Language-Action** policy for a **Unitree G1 humanoid** navigating an object-filled arena from **onboard RGBD + sensor history + a free-form English instruction**. Output: **15-dim lower-body joint targets at 50 Hz**, running in real time in **MuJoCo** (physics only).

The only pretrained weights reused are the **GR00T-N1.6 language model** (frozen, encoded once per episode and cached). Everything else — the policy, the classical perception, the velocity controller — is trained/built from scratch on synthetic teacher rollouts plus **DART** recovery data. The locomotion teacher is a Unitree whole-body-control (WBC) walk policy used **only at training time**; deployment is 100% WBC-free (an offline standing keyframe initializes the robot, then the student policy drives every step).

**Skills:** `goto` (navigate to a named object) · `search` (rotate to find an out-of-FOV target, then approach) · `maneuver` (turn L/R after passing a landmark). Plus an interactive demo with an **ego-camera | 3D-diagonal-BEV** view and multi-goal instructions ("find X then find Y").

![G1Nav interactive demo — "find the orange cube": scan → locate → walk → reach](assets/demo.gif)

<sub>Live demo (`code/fancy_demo.py`): the instruction names an object outside the robot's initial field of view; the G1 scans in place until it spots the cube, then walks to it. **Left:** onboard ego camera showing the **active** camera — the head camera at range, with an automatic handoff to a steeper **proximity camera** for the final approach, so the target stays in frame all the way to the stop. **Right:** 3D-diagonal follow-cam with path trail, target ring, FOV cone, and a `SEARCHING → LOCATED → MOVING → REACHED` status banner. Real-time, physics-only, WBC-free.</sub>

## Results (closed-loop, seed 999, n=15, WBC-free deploy)

| Task | Condition | Success |
|------|-----------|---------|
| Goto | easy / classical grounding | **100%** |
| Goto | demo-distance (4–9 m) / classical | **66.7%** |
| Goto | demo / GT goal (locomotion ceiling) | **80.0%** |
| Search | out-of-FOV / classical | **80%** (spot-rate 93%) |
| Maneuver | turn after passing a landmark | **73.3%** |

Numbers are with the **two-camera perception system** (head camera + proximity camera with hysteresis handoff, below): vs. the single-camera baseline it lifts easy 93.3→100% and demo 60→66.7% with no search regression, and keeps the target detected down to **0.26 m** (single head camera goes blind below ~0.7 m — before the stop radius).

Policy inference: **3.4 ms/step** (~6× headroom at 50 Hz). 0 falls in the goto/maneuver conditions. EGL-deterministic per seed.

> **Reproducibility note.** These headline numbers are from the released training run. A from-scratch retrain via the two-stage pipeline below reproduces the **GT-goal (pure-locomotion) metrics exactly** — easy/GT **100%**, demo/GT **80%** — which is the load-bearing result (and fixing the curriculum was essential: training `phase_A` on the *combined* set instead of easy-only gives 0% demo). The **classical-grounding** numbers show real run-to-run variance across training draws (grounding-noise robustness is a high-variance property of the fit; a multi-seed sweep spans ~87–100% on easy/classical, and a fresh retrain we verified landed ~73%). Select checkpoints by **closed-loop success, not val-loss**.

---

## Hardware / GPU

| Requirement | Specification |
|-------------|---------------|
| GPU | Developed/tested on NVIDIA GB10 (Grace-Blackwell, sm_121); any modern CUDA GPU should work |
| VRAM | ~7 GB (GR00T-N1.6 LM embedding in bf16); the student policy is tiny (7.9 M params, CPU-eligible) |
| RAM | 16 GB+ for training |
| OS | Linux (headless); `MUJOCO_GL=egl` for offscreen render |
| CUDA | 12.8 (torch 2.7.1+cu128) |

---

## Prerequisites (obtain separately — not included in this repo)

1. **GR00T-N1.6 checkpoint** → `checkpoints/GR00T-N1.6-3B/` — HuggingFace `nvidia/GR00T-N1.6-3B` (~6.2 GB, not gated).
2. **GR00T-WholeBodyControl** (the WBC walk ONNX teacher + the G1 MuJoCo model) → under `third_party/`, from NVIDIA's Isaac-GR00T repo (`n1.6.1-release`). The code uses:
   - `third_party/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Walk.onnx` (teacher)
   - `.../robots/g1/g1_gear_wbc.xml` (G1 MuJoCo model)
3. **Python 3.10 environment** with `requirements.txt`.

> **Run from the repo root** and export `PYTHONPATH` so the local `code` package isn't shadowed by another sourced environment (e.g. ROS): `export PYTHONPATH=.:$PYTHONPATH`. Always set `MUJOCO_GL=egl` for headless rendering (fallback: `xvfb-run -a env MUJOCO_GL=glfw ...`). Some scripts contain machine-specific paths from the dev box — adjust for your environment.

---

## Environment Setup

```bash
# 1. Conda environment (Python 3.10 for GR00T + flash-attn compatibility)
conda create -n g1nav -c conda-forge python=3.10 git-lfs pip -y
conda activate g1nav

# 2. Clone GR00T at the N1.6 release tag (N1.7 main breaks compatibility) and install editable
git clone --branch n1.6.1-release https://github.com/isaac-sim/Isaac-GR00T.git third_party/Isaac-GR00T
pip install -e third_party/Isaac-GR00T --extra-index-url https://download.pytorch.org/whl/cu128

# 3. torch (CUDA 12.8) + flash-attn
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.7.4.post1                 # x86_64; on aarch64 use the prebuilt wheel in the GR00T repo

# 4. Remaining dependencies
pip install -r requirements.txt

# 5. GR00T-N1.6-3B checkpoint (~6.2 GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/GR00T-N1.6-3B', local_dir='checkpoints/GR00T-N1.6-3B')"

# 6. Verify GPU + MuJoCo + GR00T + WBC ONNX load
export PYTHONPATH=.:$PYTHONPATH
MUJOCO_GL=egl python code/check_env.py
```

---

## Dataset Generation (deterministic — seed-reproducible)

```bash
export PYTHONPATH=.:$PYTHONPATH

# Clean easy rollouts (seed 0, 80 episodes)
MUJOCO_GL=egl python code/gen_dataset.py --difficulty easy --seed 0 --num-episodes 80 --out dataset/easy_seed0

# Add the gait-phase column to the clean episodes
python code/gen_dart_dataset.py add-phase --in-dir dataset/easy_seed0 --out-dir dataset/clean_with_phase

# DART easy (seed 42, 200 eps — render-free, fast)
MUJOCO_GL=egl python code/gen_dart_dataset.py generate --difficulty easy --seed 42 --num-episodes 200 --noise 0.07 --out dataset/dart_easy

# DART demo (seed 200, 200 eps; covers all robot start-yaw orientations)
MUJOCO_GL=egl python code/gen_dart_dataset.py generate --difficulty demo --seed 200 --num-episodes 200 --noise 0.07 --maxsteps 1400 --out dataset/dart_demo

# Combine clean + DART into the training set. `combine` merges one clean-dir with one dart-dir;
# the released model trains on dart_combined_v2 = clean_with_phase + dart_easy + dart_demo
# (476 eps / 180,696 frames). Merge the two DART dirs first (or run combine per pair) as needed.
python code/gen_dart_dataset.py combine --clean-dir dataset/clean_with_phase --dart-dir dataset/dart_easy --out dataset/dart_combined_v2

# Maneuver dataset (seed 100; ~159 usable eps after fall-filtering)
MUJOCO_GL=egl python code/gen_maneuver_dataset.py generate --seed 100 --num-episodes 200 --noise 0.07 --maxsteps 1400 --out dataset/maneuver

# GR00T-LM language-embedding cache (2048-d; used by the data loader / language conditioning)
MUJOCO_GL=egl python code/groot_lang.py --ckpt checkpoints/GR00T-N1.6-3B --out dataset/lang_cache.pkl

# Offline standing keyframe (for the WBC-free deploy init)
MUJOCO_GL=egl python code/gen_stand_keyframe.py
```

---

## Training

```bash
export PYTHONPATH=.:$PYTHONPATH

# Stage 1 — Goto policy on the EASY-only set (dart_combined, ~280 eps, yaw=0). This is the
# two-stage curriculum: learn easy locomotion FIRST, then add demo-distance data in stage 2.
# (Training directly on the combined set does NOT reproduce the demo numbers.) 25 epochs (~1.1 h on GB10).
MUJOCO_GL=egl python code/train_dart_phase.py --arch A --data dataset/dart_combined --out runs/dart_phase_A \
    --epochs 25 --batch 64 --lr 3e-4 --swing-weight 2.0 --device cuda

# Stage 2 — Fine-tune on the COMBINED set (dart_combined_v2 = easy + demo DART; fixes yaw covariate shift). 20 epochs.
MUJOCO_GL=egl python code/train_dart_phase.py --arch A --data dataset/dart_combined_v2 \
    --resume-ckpt runs/dart_phase_A/model_best.pt --out runs/demo_dart_A \
    --epochs 20 --batch 64 --lr 1e-4 --reset-epoch --swing-weight 2.0 --device cuda

# Maneuver policy — fine-tune from the goto checkpoint, on maneuver data MIXED with the
# goto set (dart_combined_v2) so the goto skill isn't forgotten
MUJOCO_GL=egl python code/train_maneuver.py --arch A --data dataset/maneuver dataset/dart_combined_v2 \
    --resume-ckpt runs/demo_dart_A/epoch_0003.pt --out runs/maneuver_A \
    --epochs 10 --batch 128 --lr 5e-5 --device cuda
```

> **Select checkpoints by closed-loop success, not offline val-loss** (the two diverge — in our maneuver runs the val-loss minimum is ~27pp worse in closed loop than the behavioral peak, which lands in the first ~3 epochs). For goto use `runs/demo_dart_A/epoch_0003.pt`; for maneuver, sweep the early epochs with `eval_maneuver.py` and take the closed-loop best (epoch 2 in the released run; a from-scratch reproduction peaked at epoch 3, same 73.3%). Copy the chosen ones to `checkpoint/goto_best.pt` and `checkpoint/maneuver_best.pt` (the demo/eval scripts load those by default).

---

## Evaluation (closed-loop, seed 999)

`eval_closedloop.py` uses `--checkpoint`, `--n`, `--goal-source {classical,gt}`, `--difficulty {easy,demo}`, `--seed`, `--device`. (No GR00T is loaded at eval time — the language embedding is zeroed; navigation is driven by the grounding goal.)

```bash
export PYTHONPATH=.:$PYTHONPATH

# Goto — easy / classical grounding (~100%)
MUJOCO_GL=egl python code/eval_closedloop.py --checkpoint checkpoint/goto_best.pt --arch A \
    --difficulty easy --goal-source classical --n 15 --seed 999 --device cuda --out eval/easy_classical

# Goto — demo-distance / classical grounding (~67%)
MUJOCO_GL=egl python code/eval_closedloop.py --checkpoint checkpoint/goto_best.pt --arch A \
    --difficulty demo --goal-source classical --n 15 --seed 999 --device cuda --out eval/demo_classical

# Goto — demo / GT goal (~80%); grounding unused, so --no-render is fine
MUJOCO_GL=egl python code/eval_closedloop.py --checkpoint checkpoint/goto_best.pt --arch A \
    --difficulty demo --goal-source gt --n 15 --seed 999 --device cuda --no-render --out eval/demo_gt

# Search — out-of-FOV target (~80%); reuses the goto checkpoint (no search-specific training)
MUJOCO_GL=egl python code/eval_search.py --checkpoint checkpoint/goto_best.pt --n 15 --seed 999 --device cuda --out eval/search

# Maneuver (~73%)
MUJOCO_GL=egl python code/eval_maneuver.py --checkpoint checkpoint/maneuver_best.pt --n 15 --seed 999 --hybrid-vel --device cuda --out eval/maneuver
```

---

## Interactive Demo

`demo.py` and `fancy_demo.py` automatically load `checkpoint/goto_best.pt` and `checkpoint/maneuver_best.pt` (no checkpoint flags needed).

```bash
export PYTHONPATH=.:$PYTHONPATH

# Terminal REPL — type instructions, watch the robot execute (multi-goal + clarification Q&A)
MUJOCO_GL=egl python code/demo.py --difficulty easy --device cuda

# Web UI (Flask MJPEG stream on port 5000)
MUJOCO_GL=egl python code/demo.py --web --difficulty easy --device cuda
```

### Fancy demo — ego | 3D-diagonal BEV, live web UI, long-distance search + multi-goal

`code/fancy_demo.py` shows the **ego camera | elevated 3D-diagonal BEV follow-cam** side-by-side, with overlays: path trail · target ring + crosshair · FOV cone · status banner (`SEARCHING → LOCATED → MOVING → REACHED`) · multi-goal progress dots.

```bash
# Live web UI (port 5001): open http://localhost:5001, type a prompt, watch the live stream
MUJOCO_GL=egl python code/fancy_demo.py --web --device cuda

# Headless showcase render (5 long-distance search + 1 multi-goal, saves MP4s + reel)
MUJOCO_GL=egl python code/fancy_demo.py --smoke --n-smoke 6 --device cuda --out eval/fancy_demo
```

Example prompts: `find the red ball` · `go to the orange cone` · `find the purple ball then find the yellow cube` · `turn left after passing the blue cube`.
Use **red / orange / yellow / purple** objects for the most reliable grounding (cyan/blue can collide with the wall color in HSV — a documented limitation).

To view the web UI from your laptop over SSH: `ssh -L 5001:localhost:5001 <user>@<host>`, run the command above on the host, then open `http://localhost:5001` locally.

---

## Repository Layout

```
VLA_mujoco_unitree/
├── README.md
├── requirements.txt
├── .gitignore
├── assets/demo.gif             # the README demo clip
└── code/                       # source (28 files)
    ├── teacher.py              # WBC teacher wrapper (training-only)
    ├── arena.py  scene.py  steer.py  maneuver_scene.py  maneuver_expert.py
    ├── gen_dataset.py  gen_dart_dataset.py  gen_maneuver_dataset.py  gen_stand_keyframe.py  groot_lang.py
    ├── dataset.py  dataset_phase.py  dataset_maneuver.py
    ├── small_vla.py  train_dart_phase.py  train_gaitfix.py  train_maneuver.py  action_stats.py
    ├── grounding.py            # classical HSV + depth grounding
    ├── inferencer.py           # closed-loop deploy (3-rate pipeline, WBC-free)
    ├── eval_closedloop.py  eval_search.py  eval_maneuver.py
    ├── demo.py  fancy_demo.py
    └── check_env.py
```

Created at runtime and **gitignored**: `dataset/`, `runs/`, `checkpoint/`, `eval/`, `videos/`.
External, **not committed**: `checkpoints/` (GR00T-N1.6), `third_party/` (GR00T + WBC).

---

## Method (one paragraph)

Modular VLA: `language → cached GR00T-LM embedding` · `RGBD → classical HSV+depth grounding → egocentric goal (dist, bearing)` · `goal → velocity command` · `velocity + proprio history → distilled 15-DoF joint targets`. The joint policy is distilled from the WBC walk teacher via behavior cloning, and stabilized over long horizons with **residual/normalized action targets + DART recovery data + a gait-phase input** — which is what takes naive BC from 0% to 100% on the easy task. Search is a student-driven fixed-CCW scan gated on target visibility; maneuver adds a landmark-pass trigger + heading goal. Real time comes from a 3-rate split: language once per episode, grounding at 5–10 Hz, the action head at 50 Hz.

**Perception — two-camera handoff.** A single pitched head camera goes blind below ~0.7 m (the target exits the FOV bottom edge before the stop radius). Grounding therefore runs on the **active** one of two head-mounted cameras: the head camera at range, and a steeper **proximity camera** (58° pitch) for the final approach, switched by a hysteresis (Schmitt) trigger on the smoothed target distance (in ≤1.2 m, out ≥1.6 m) with depth-based rejection of the robot's own body in frame. Both cameras feed the same `(dist, bearing)` goal, so the policy needs **no retraining**, and only the active camera is rendered each cycle, so steady-state compute is unchanged. This keeps the target detected down to **0.26 m** — through every skill's stop radius. (A wide-FOV single-camera alternative was A/B-tested and rejected: it loses far-range detection, has a shallower close-range floor, and is ~2× slower.)
