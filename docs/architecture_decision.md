# ADR-001 — Architecture Decision (v2, "GroundedNav")

Date: 2026-07-05. Decider: project owner, after Phase-R research (R1–R4).

## Decision
Adopt a **modular VLA** (Architecture A). One policy, interpretable internal heads; the bottom head's output = **15-dim G1 lower-body joint targets** (satisfies the project's task spec). Intermediates (egocentric goal, velocity) are teacher-forced in training, predicted at deploy.

## Why (evidence)
- R4: modular >> end-to-end for sim2real / small-data / real-time ObjectNav (Gervet'23 90% vs 23%; VLFM > 77k-demo PIRLNav).
- R1: GR00T's manipulation prior does not transfer to open-space locomotion → end-to-end fine-tune is data-hungry & risky; R1 itself recommends classical-CV grounding + aux pose head.
- R2: pretrained detectors weight-illegal; classical HSV+depth is near-perfect here; from-scratch OWL-head on frozen SigLIP2 is the legal learned upgrade.
- R3: out-of-FOV search = classical occupancy+frontier+RGBD odometry persistent goal.
- 50 Hz on one GPU forces the rate split (grounding low-Hz, control 50 Hz).

## Module spine
1. **Language (GR00T reuse, frozen):** instruction → GR00T-N1.6 LM embedding, encoded **once/episode, cached** (2048-d).
2. **Grounding (legal):** ego RGB(D) + target(color,shape) → **egocentric goal (dist, cosθ, sinθ)** in robot frame.
   - v2.0 primary: **classical HSV color + shape + depth back-projection** (0 weights). Runs 5–10 Hz, cached, confidence-gated.
   - v2.1 upgrade (parallel track): **from-scratch OWL-ViT-style head on frozen GR00T SigLIP2 patch tokens**, trained on free GT boxes.
3. **Velocity commander (from-scratch MLP):** goal[+episode/FSM state] → (vx,vy,ωz). 50 Hz.
4. **Distilled student (from-scratch, THE output):** (velocity cmd + proprio history) → **15 joint targets**. Distilled from **WBC teacher** (BC + **DAgger + DART**). 50 Hz. **Action chunking H≈16 + temporal ensembling** for gait smoothness (R1/ACT).
5. **Search memory (Phase 2):** occupancy + Yamauchi frontier + RGBD odometry → persistent world-frame goal when off-camera; re-anchor on re-detection.

## Training protocol
- Teacher-force GT (dist,bearing) & vel in offline training; feed **predicted** intermediates during DAgger rollouts (kill exposure bias). Directed chain (no feedback into grounding).
- Losses: grounding smooth-L1 + velocity smooth-L1 + action smooth-L1 + done BCE [+ target_visible BCE for search].
- **Overfit-gate first**; **select/stop on closed-loop success (seed 999)**, never offline loss.
- **Stability / long-horizon training method = OPEN, evidence-selected** (not fixed to DAgger/DART). Candidate levers, chosen per the per-experiment loop + R5 research: {BC baseline, DAgger, DART, **RL fine-tuning of the BC policy** (KL-to-teacher PPO / DAPG / residual RL — trains on the policy's own state dist, directly rewards reach+upright), offline RL (IQL/CQL/TD3+BC), **action chunking + temporal ensembling (ACT)**, diffusion/flow action head, domain randomization / RSI / DeepMimic-style imitation reward}. RL from scratch in MuJoCo satisfies the pretrained-weights constraint (no external pretrained weights). Bottleneck to beat: covariate-shift falls over ~1400-step demo-preset episodes.

## Bake-off (Phase 1, easy goto) — how the decision gets validated
- **A** = modular (grounding→vel→joints, above).
- **C** = pure BC student, goal-text-conditioned, grounding implicit through action loss (control/lower bound).
- Decide on held-out closed-loop success + real-time wall-clock. Optional **B** (hierarchical latent on GR00T) only if data proves cheap.

## Canonical data schema (contract for S3 data + S4 model)
Per timestep, LeRobot layout:
- `ego_rgb` (H×W×3, uint8), `ego_depth` (H×W×1, float) — onboard camera. Input res for policy ≈ 128².
- `proprio`: 15 leg/waist (q,qd)=30 + base IMU (quat4+angvel3+linacc3=10) + prev_action(15) → ~55-d (S3 finalizes exact dims; S4 reads dim from data).
- `lang_emb`: 2048-d cached GR00T-LM embedding (per episode).
- Labels: `action`=15 teacher joint targets · `goal`=(dist,cosθ,sinθ) egocentric (privileged, per-frame) · `vel_cmd`=(vx,vy,ωz) teacher command · `done` · [`target_visible` for search] · `task_description`.
- Histories: proprio history (K≈6–20); 2–4 recent RGB frames optional.
- 3rd-person camera rendered for `videos/` only (never a policy input). Object poses privileged (labels only).

## Legality check ✓
output=joints; only GR00T params reused (LM; optional SigLIP2 for grounding upgrade); WBC teacher-only (not in deploy loop); classical CV=no weights; commander/student from scratch; physics-only.

## Difficulty / demo target (main-agent decision 2026-07-05)
Deliverable = a **demo-ready video that showcases the system convincingly**, so the shipped task is HARD, not the floor. Arena/scene generator is **parametric** (`--difficulty`):
- **easy** — 4 m arena, ~3 objects, target in FOV, dist ~1.5–2.5 m, STOP_R 0.6, horizon ~600. Use ONLY for bring-up, overfit gate, A-vs-C bake-off.
- **demo** (the deliverable) — **10–12 m arena, 5–7 objects spread far apart (min pairwise spacing ~2.5 m), robot at edge, target dist 4–9 m, target often OUT of initial FOV (→ turn/search), STOP_R 0.4, long horizon ~1400 steps (~28 s), varied lighting.**
Showcase behaviors on demo preset: goto (long-distance, distractors) · maneuver (turn after passing landmark) · search (off-camera target) · multi-goal sequence · live interactive REPL. All final `videos/` render on **demo**. Ramp easy→demo via DART+DAgger (long-horizon stability is the crux).
