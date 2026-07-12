# NX-6 — Learned Grounding, CenterNet Variant (TRAIN)

**Date:** 2026-07-09
**Agent:** NX-6 (TRAIN), consumes `dataset/det_v1` + `dataset/det_failcases`
(`docs/nx6_data.md`).
**Goal:** train a small (<5M param), from-scratch, query-conditioned
RGBD detector that replaces `code/grounding.py`'s HSV pipeline — predicting,
per (class, color) query, presence + pixel centroid + metric distance, from
which (dist, bearing) is recoverable via camera intrinsics. Select by VAL
recall@(bearing<2deg, dist<0.5m) at precision>=0.9, then grade honestly
against `dataset/det_failcases` — the actual point of this exercise: does it
reject the classical grounder's wall-stripe false locks (demo ep0/2/4/5) and
separate the cyan-ball/cyan-cube twin (ep12)?

**Bottom line:** yes on both counts. On the 118-frame failcase set the
selected checkpoint **never once claims presence when the true target isn't
actually visible (103/103 correct rejections)**, keeps **perfect precision**
on the frames where it does claim presence, and correctly discriminates the
ep12 twin (queries for "cyan ball" and "cyan cube" in the same frame each
land on their own object, not each other's). Val/test recall under the
stated bearing/dist criterion sits at ~0.68-0.70 with precision~0.90-0.91.
Latency is 1.4-1.8ms/frame on the GB10 GPU (0.55M params) — no binding
constraint on the 5-10Hz budget. One honest miss: the distance head has a
residual **systematic over-estimate bias (mean +0.29 to +0.32m)**, most
pronounced at 3-5m range; a targeted fix (`--dist_w 3.0`, run B) narrowed the
error spread but didn't remove the bias — flagged below with a concrete
follow-up.

---

## 1. Architecture (`code/model_centernet.py`)

Query-conditioned CenterNet-style detector, **trained from scratch** (project
constraint: only GR00T-LM may hold pretrained params):

```
RGBD (4,120,160) --stem(s=2)--> (32,60,80) --stage1(s=2)--> (64,30,40)  [output stride 4]
                                                    |
                                          FiLM(query) -> ReLU
                                                    |
                              FiLMResBlock(64->64) -> FiLMResBlock(64->96) -> FiLMResBlock(96->96)
                                                    |
                                            head_trunk (3x3 conv)
                                          /            |            \
                                heatmap(1x1->1)   offset(1x1->2)   dist(1x1->1, softplus+0.05)
```

- **Query conditioning**: `class_id` (4) / `color_id` (7) embeddings (dim 24
  each) -> concat -> 2-layer MLP -> FiLM (per-channel scale/shift, near-
  identity init) applied at 4 points in the trunk. One-hot/embedding
  conditioning, **not text**, per the project constraint.
- **Params: 547,388 (0.55M)** — `count_params()` in the model file; **11% of
  the 5M budget**, chosen generously (an earlier 0.2M sizing also worked, but
  the extra width was cheap given the huge param headroom and the latency
  numbers below still leave >>10x margin).
- **Canonical input resolution 160x120**: both deploy cameras (grounding
  480x360, proximity 320x240) are downsampled by an **exact integer factor**
  (3x and 2x respectively) to this one shared resolution — both cameras are
  in-distribution for a single trunk, per the brief. This is also why it's
  a clean choice: `code.grounding.get_ego_intrinsics_rendered` gives both
  cameras the **same FOVY (45deg) and aspect ratio (4:3)**, differing only in
  pixel count and mount pitch — so 160x120 intrinsics are identical for both
  cameras, and only the pitch (26deg grounding / 58deg proximity) needs to be
  threaded through at decode time.
- **Output stride 4** -> heatmap grid 40x30.

### Decode (`code/centernet_utils.py`, `code/model_centernet.decode_peak`)

1. Heatmap peak (argmax of `sigmoid(logits)`) -> presence confidence.
2. Sub-pixel center = peak cell + predicted offset, in canonical pixels.
3. **Distance** = the distance head's own value at the peak cell (metres,
   `softplus(x)+0.05` for positivity) — a **learned** quantity, not a raw
   depth-channel read (see rationale below).
4. **Bearing** = geometric: read a median depth in a 3x3 window around the
   predicted center from the **input depth channel** (the same measurement
   source the classical grounder already uses), then
   `arena.backproject_pixel` -> `grounding.cam_to_egocentric(...,
   use_corrected_unpitch=True)`. Falls back to the distance head's own value
   if the depth read is invalid (<=0.05m, e.g. under depth-dropout aug).

**Why a learned distance head at all, if depth is a direct input?** Depth is
noisy/absent near-field (the reason `code/grounding.py` needed its own
depth-outlier handling for the proximity camera, `docs/cam_p1.md`) — a head
trained end-to-end on the *analytic* GT distance (not raw z-depth) can fuse
appearance (object size/perspective) with a possibly-corrupted depth channel,
and was explicitly trained under **depth-channel dropout** (25% of train
samples zero the whole depth channel) to be robust to exactly that failure
mode. Bearing, by contrast, is a much lower-dimensional, more purely
geometric quantity once you have *a* depth reading — deriving it via the
validated project pipeline reuses machinery that's already been checked
against analytic ground truth (next section) rather than asking the network
to learn trigonometry from scratch.

**Decode formula convention — why "always corrected", not the production
per-camera legacy toggle:** `code.grounding.ground()` intentionally keeps the
26deg grounding camera on an older, slightly-biased un-pitch formula in
production so as not to shift the distribution a specific deployed policy
checkpoint was tuned against (`docs/cam_p1.md`). Empirically (val split,
unoccluded, `area_px>=200`, using the labels' own analytic GT bearing/dist as
truth):

| convention | grounding p95 bearing err | proximity p95 bearing err |
|---|---|---|
| naive (ignore pitch entirely) | 1.6deg | 18.4deg |
| pitch-corrected, **no depth/offset** (direction-only) | 6.0deg | 5.4deg |
| production legacy toggle (grounding=legacy, proximity=corrected) | 5.6deg | 3.3deg |
| **always-corrected both cams + real depth (this detector's choice)** | **1.2deg (both cams pooled, area>=200)** | |

The always-corrected formula (matching `docs/nx6_data.md` §5's own dataset
label-geometry self-check, which reports p95=2.67deg over a slightly broader
filter) is the only convention that leaves headroom under the <2deg
bearing-recall gate once the network's own localization error stacks on top
— confirmed by reproducing the dataset's own check (738 filtered val
detections): **p95 bearing err = 1.2deg, p95 dist err = 0.09m** with GT
centroid+depth. Ceiling achievable at canonical (160x120, downsampled)
resolution with a *perfect* detector: **95.4%** of positives satisfy
bearing<2deg AND dist<0.5m simultaneously — i.e. even flawless localization
can't reach 100% under this criterion; ~4.6% is baked into the geometry/
discretization at this resolution, not something training can fix.

---

## 2. Data pipeline (`code/dataset_det.py`)

Builds query-conditioned samples on the fly from `dataset/det_v1`'s per-frame
object lists (`docs/nx6_data.md`):

- **Query sampling** per frame draw: 50% positive (query = a visible object's
  own class+color), 30% **hard negative** (a visible object's color paired
  with a *different* class, or its class paired with a *different* color —
  forces the FiLM conditioning to gate on the *joint* (class,color), not
  either factor alone — this is what makes the ep12 discrimination possible
  at all), 20% random negative (includes the dataset's own ~24% zero-object
  frames as free hard negatives). Presence label is always re-derived by
  looking up whether the drawn (class,color) actually matches a visible
  object — sampling strategy only biases *which* queries get seen more, never
  mislabels one.
- **Preprocessing**: both cameras' images are resized **once** at dataset
  construction (not per-`__getitem__`) to a training "canvas" 15% larger than
  the canonical 160x120, so per-step augmentation is just cheap array slicing.
- **Train-only augmentation**: random crop canvas->canonical (small
  translation jitter, "small crops" per brief), horizontal flip (p=0.5,
  centroid coordinate flipped too — trivially valid since bearing is derived
  from pixel position, no explicit left/right label to invalidate),
  brightness/contrast jitter + Gaussian pixel noise on RGB, **depth-channel
  dropout** (p=0.25, zeroes the whole depth channel — forces the model not to
  collapse if depth is bad near-field, per the brief).
- Val/test/failcases: **no augmentation**, canonical-resolution images built
  directly, and a **fixed seeded query list** (every visible object as a
  positive + ~2 hard/random negatives per frame) so metrics are comparable
  checkpoint-to-checkpoint.

## 3. Loss (`code/train_centernet.py`)

Standard CenterNet modified-focal loss on the heatmap (pos/neg pixel
weighting, `alpha=2, beta=4`) + masked L1 on the sub-pixel offset (only at
the true center cell) + masked Smooth-L1 (Huber, beta=1) on the distance head
(same masking). `total = hm_loss + off_loss + dist_w * dist_loss` — see
run A vs B below for `dist_w`.

## 4. Training runs

AdamW, lr=3e-4 cosine-annealed to 0, batch=128, 30 epochs, weight decay
1e-4, grad-clip 5.0, ~24s/epoch on the GB10 (8,840 train frames, workers=4).
Model selection each epoch: run the fixed val query set (3,284 samples),
sweep threshold in `[0.05, 0.95]`, pick `argmax(recall_strict)` subject to
`precision >= 0.9` where `recall_strict` = fraction of true-positive queries
detected present AND within bearing<2deg AND dist<0.5m of GT (the exact
brief-specified criterion) — checkpoint saved whenever this improves.

**Run A** (`dist_w=1.0`, `runs/nx6_centernet_A/`): converged smoothly
(hm/off/dist train losses: 12.5/0.79/0.92 at step 0 -> 0.23/0.10/0.02 by
epoch 27). Best at epoch 26.

**Distance calibration check (post-hoc)** on run A revealed a **systematic
positive bias**: mean signed error `pred - gt` = **+0.316m** across all 1,110
val positives (median +0.33m), worst in the 3-5m band (+0.55m mean, only
52.9% within the 0.5m tolerance there vs 89-93% at 0-2m). This is a real
finding, not an eval bug (verified the standalone inference wrapper
reproduces the training dataset's own predictions bit-for-bit on the same
frames before concluding this).

**Run B** (`dist_w=3.0`, `runs/nx6_centernet_B/`, same everything else):
one targeted retrain to see if boosting the distance loss's share of the
shared-trunk gradient would fix the calibration. Result: **mean bias only
marginally reduced (+0.316m -> +0.292m)**, but the **error spread tightened
substantially** (fraction within 0.5m: 79.9% -> 87.7% pooled over all val
positives) — net effect, better recall_strict on val/test despite the bias
not being solved:

| | val precision | val recall_strict | test precision | test recall_strict |
|---|---|---|---|---|
| A (dist_w=1.0) | 0.904 | 0.680 | 0.909 | 0.682 |
| **B (dist_w=3.0), selected** | 0.902 | **0.695** | 0.893 | 0.688 |

Per the stated selection rule (max val recall_strict s.t. precision>=0.9),
**run B is the selected checkpoint** (`runs/nx6_centernet_B/best.pt`,
epoch 23, thr=0.45). Both runs are kept for comparison.

**Val threshold sweep (run B, full table)** — the selected thr=0.45 is
indeed the precision>=0.9 row with the highest recall_strict:

| thr | precision | recall_strict | recall_presence |
|---|---|---|---|
| 0.05 | 0.587 | 0.846 | 0.986 |
| 0.30 | 0.813 | 0.810 | 0.926 |
| 0.40 | 0.875 | 0.742 | 0.840 |
| **0.45** | **0.902** | **0.695** | 0.777 |
| 0.50 | 0.929 | 0.644 | 0.715 |
| 0.65 | 0.986 | 0.358 | 0.384 |

(Full sweep in-repo via `select_threshold()`; note recall_strict is
*non-monotonic* in threshold at the low end — below ~0.15 it's flat/slightly
declining because it's bounded by localization accuracy, not just presence
detection, so lowering the threshold further only adds false positives, not
more accurate true positives.)

**Distance bias is not fully closed** — flagged honestly as the main
follow-up (see §7).

## 5. Failure-case acid test (`dataset/det_failcases`, run B / selected ckpt)

Query = each episode's own instructed `(target_color, target_shape)`
(`frames.parquet`), one sample per frame. Ground truth presence = whether
that (class,color) has a labeled row for that frame (i.e. whether the true
target is **actually visible** per MuJoCo segmentation at that pose — the
classical grounder failed several of these precisely by confidently
reporting a detection when the true target was NOT visible, or by locking
onto the wrong object).

```
n_direct=118 (103 gt-absent "reject" frames, 15 gt-present frames)
precision                 = 1.000   (0/0 false claims — every "present" call was correct)
recall_presence           = 0.533   (8/15 correctly flagged present when visible)
recall_strict             = 0.533   (8/15 present AND bearing<2deg AND dist<0.5m)
reject_rate_on_not_visible = 1.000  (103/103 — NEVER a confident false lock)
```

**`reject_rate_on_not_visible=1.000` is the headline number** — this is
exactly the classical grounder's documented failure mode
(`docs/nx6_data.md` §6: ep0 "marginal/flickering blob at wrong depth", ep2
"confident false-lock, monotonically diverging", ep4 "total detection miss ...
false hue-overlap", ep5 "same stall signature as ep0"). The learned detector
never reproduces that: on every single frame where the true target had left
the segmented view, it correctly said "not visible" rather than confidently
reporting a wrong distance.

**Run-to-run note on recall (be honest about small-n noise):** run A scored
recall_presence=0.733 (11/15) / recall_strict=0.667 (10/15) on this same set
— *higher* than run B's 8/15, despite run B winning on the (1,110-sample) val
set. Diffing the two runs frame-by-frame: the difference is **entirely** 3
borderline `demo_ep5` frames (60-62, the "stall" episode with a marginal/
flickering blob even in the GT-labeled frames) whose presence confidence sits
right at the 0.45 threshold (A: 0.57/0.49/0.54; B: 0.41/0.35/0.36) — when
computed anyway, their bearing/dist *would* still have been accurate in both
runs. This reads as threshold-boundary sensitivity on a handful of inherently
marginal frames, not a fundamental capability regression — but with n=15
positive failcase frames total, this is a genuine limitation of how much this
acid test alone can distinguish the two checkpoints; `reject_rate=1.000` and
`precision=1.000` (the safety-critical properties) are identical and perfect
in both.

### Twin discrimination (ep12 cyan-ball vs. cyan-cube — the point of this variant)

For every failcase frame with a co-visible distractor (different class or
color from the instructed target, e.g. a cyan ball next to the instructed
cyan cube), additionally query the **distractor's own** (class,color) and
check the prediction lands on *its* location, not the target's:

```
n_distractor_queries        = 62
distractor_present_rate     = 0.871  (correctly says "present" -- correct, it IS visible)
distractor_localization_rate = 0.823  (present AND localized to ITS OWN position, bearing<2deg/dist<0.5m)
distractor_hijack_rate      = 0.016  (1/62 -- prediction landed closer to the TARGET than the queried distractor)
```

The critical frames — `demo_ep12` frame_uid 79/80, where the doc'd twin-
hijack happens (`docs/nx6_data.md` §6: classical grounder locks onto the
cyan ball at dist~1.0m while the true cyan-cube target is at dist~6.18m, same
frame, both visible):

| query | ep12 frame | bearing_err | dist_err | accurate | hijacked |
|---|---|---|---|---|---|
| cyan cube (instructed target) | 79 | 0.09deg | 0.11m | True | — |
| cyan ball (distractor) | 79 | 0.09deg | 0.28m | True | False |
| cyan cube (instructed target) | 80 | (n/a, target left frame by step 200 per doc) | | | |
| cyan ball (distractor) | 80 | 0.08deg | 0.26m | True | False |

Querying "cyan cube" and "cyan ball" on the *same frame* land on two
different, correct locations — the class+color joint conditioning (via the
hard-negative training signal in §2) does what it's meant to. Only 1/62
distractor queries anywhere in the failcase set showed a hijack, and it was
not in ep12.

## 6. Latency

Single-frame (batch=1) forward, 200 iters after 20-iter warmup:

| device | latency |
|---|---|
| GB10 GPU (CUDA) | **1.4-1.8ms** |
| CPU (same weights) | ~47-50ms |

547K params leaves >>10x margin under the 5M budget; 1.8ms leaves >>10x
margin under a 100-200ms (5-10Hz) cycle budget even including the decode's
Python-side depth-window read and backprojection (not separately timed, but
those are O(1) numpy ops, negligible next to the CPU number above which
already dwarfs them).

## 7. Checkpoint + inference wrapper

- **Selected**: `runs/nx6_centernet_B/best.pt` (epoch 23, thr=0.45) — see
  `runs/nx6_centernet_B/README.md` for the full checkpoint contents and API.
- Comparison run: `runs/nx6_centernet_A/best.pt` (epoch 26, thr=0.45,
  `dist_w=1.0`).
- Standalone inference wrapper: `code/nx6_infer_centernet.py`
  (`load_model()`, `detect()`) — mirrors `code.grounding.ground()`'s call
  contract (dist, bearing, confidence, not_visible) for a drop-in toggle,
  matching `docs/nx6_data.md` §8's suggested next step.
- Training/eval driver: `code/train_centernet.py` (train + val-select +
  failcase acid-test + latency bench + checkpoint save, single command).
- Model: `code/model_centernet.py`. Dataset: `code/dataset_det.py`. Shared
  geometry/heatmap utils: `code/centernet_utils.py`.

## 8. Honest limitations / follow-ups

1. **Distance bias not fully closed.** +0.29m mean signed error remains
   after the `dist_w=3.0` retry, worst at 3-5m range. Next things to try:
   (a) a post-hoc linear (or per-range-bucket) calibration fit on val and
   folded into the checkpoint/wrapper (cheap, principled — val is exactly
   what calibration is for), (b) feeding `cam_type`/pitch as an explicit
   input rather than only using it at decode time, (c) longer training with
   a distance-head warmup phase before the heatmap loss dominates the shared
   trunk's early gradients.
2. **Failcase n=15 positive frames is small.** `reject_rate=1.0` and
   `precision=1.0` are the numbers that matter most for this variant's
   purpose and are unambiguous (103/103 and all-positive respectively), but
   the recall split (8/15 vs 11/15 across A/B) has wide binomial noise at
   this sample size — don't over-read the single-digit swing.
3. **Val/test recall_strict (~0.69) vs. the ~0.95 geometric ceiling (§1)**
   means there's real headroom in localization/distance accuracy left on the
   table beyond the calibration fix above — more epochs, a slightly larger
   trunk (still well under the 5M budget), or class-balanced query sampling
   (right now hard-negatives are drawn uniformly over the wrong-class/wrong-
   color complement, not stratified by how visually confusable that
   particular substitution is) are the natural next levers.
4. Not yet gated end-to-end (swapped into `code/grounding.ground()`'s role
   behind a toggle and re-run through the full easy/demo/search closed-loop
   eval) — per `docs/nx6_data.md` §8 this is the necessary next step before
   ADOPT, out of scope for this TRAIN task.
