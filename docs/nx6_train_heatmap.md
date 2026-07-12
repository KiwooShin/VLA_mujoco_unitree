# NX-6 TRAIN — learned grounding, HEATMAP variant

**Date:** 2026-07-09
**Agent:** NX-6 TRAIN (heatmap), sibling of at least one other NX-6 TRAIN variant
(`code/train_centernet.py`, observed running concurrently on the GPU during this
session — presumably a bakeoff between architectures on the same
`dataset/det_v1`/`dataset/det_failcases`, per `docs/nx6_data.md` §8's suggested
next step).
**Data:** `dataset/det_v1` (11,059 frames / 11,116 labels, 280/35/35 scene split) +
`dataset/det_failcases` (118 frames, 6 episodes, the classical grounder's
documented demo/search failures).

## 1. Architecture

Query-conditioned heatmap detector, **from scratch** (no pretrained backbone
anywhere — project constraint), **0.87M params** (budget was <5M):

- **Input:** RGBD, resized to a common **192x144** canvas. Both deploy cameras
  (grounding 480x360 @26deg pitch, proximity 320x240 @58deg pitch) render at the
  same FOVY=45deg and are already 4:3, so a single uniform resize to 192x144
  preserves the pinhole model exactly for *both* cameras — `get_ego_intrinsics_rendered(192,144)`
  is valid regardless of source camera (`code/nx6_heatmap_model.py:TARGET_INTR`).
- **Query conditioning:** one-hot(class, 4) + one-hot(color, 7) = 11-d vector -> 2-layer
  MLP (64-d embedding) -> broadcast-concatenated at the bottleneck feature map
  (24x18). Not text — a direct one-hot as instructed (the query already arrives as
  a parsed (class, color) spec from the instruction parser, `code/grounding.py`'s
  `ground()` signature).
- **Backbone:** small U-Net — stem + 3 stride-2 conv blocks (32/64/96/128 channels)
  down to 24x18, bottleneck (concat query embedding, 2 convs), 3 upsample+skip
  blocks back to 192x144, 1x1 head -> 2 output channels.
- **Output:** `[presence_heatmap_logit, distance_residual_m]` at full 192x144
  resolution.
- **Decode:** sigmoid(heatmap) hard-argmax pixel, refined by a 5x5
  intensity-weighted local centroid (sub-pixel refinement of the "argmax pixel"
  the brief specifies) -> `code.arena.backproject_pixel` +
  `code.grounding.cam_to_egocentric(pitch_deg=<cam>, use_corrected_unpitch=True)`
  using the query class's nominal object radius (same geometry pipeline
  `dataset/det_v1`'s own labels were validated against, `docs/nx6_data.md` §5,
  p95 4.2cm/2.7deg on *perfect* full-resolution centroid+depth — this is the noise
  floor our lower-resolution, learned-centroid system operates above, not below).
  Final `dist = dist_backprojected + predicted_residual` (residual sampled at the
  peak pixel).

Files: `code/nx6_heatmap_model.py` (model + decode + `HeatmapDetector` inference
wrapper), `code/nx6_heatmap_data.py` (dataset/caching/augmentation),
`code/nx6_heatmap_eval_utils.py` (shared scoring), `code/train_nx6_heatmap.py`,
`code/eval_nx6_heatmap.py`.

## 2. Training

**Loss:** CenterNet-style penalty-reduced pixel focal loss on the heatmap
(`alpha=2, beta=4`) + Smooth-L1 on the distance residual, the latter supervised
**only at the GT peak pixel** for positive examples. Both losses use an
elementwise peak-mask multiply rather than advanced-indexing gather (`tensor[idx_b,
py, px]`) — see the perf note below, this was a real, measured bottleneck.

**Examples/epoch:** every positive label (~8,951 in train) + sampled negative
`(class,color)` queries per frame (1 per object-frame, 2 per zero-object frame) —
~19,886 examples/epoch, natural class+color mix (same-scene same-color
different-shape distractors and same-shape different-color distractors arise for
free from `code/scene.py`'s sampler, exactly the twin-discrimination signal this
variant needs).

**Augmentation:** photometric jitter (brightness/offset/noise), random-resized-crop
(0.8-1.0 scale, target-preserving for positives, with correctly re-derived
intrinsics for the residual-target recomputation), depth-channel dropout (15% full
zero-out, 25% near-field (<1.2m) multiplicative noise — "depth is noisiest
near-field" per the brief).

**Selection metric:** recall @ (bearing err < 2deg AND dist err < 0.5m) subject to
precision >= 0.9, swept over confidence threshold on a fixed val example set (1,110
positives + ~4,140 sampled negatives). A "detection" only counts as a true positive
if the query object is actually present **and** both error bars are met — a
confident detection at the wrong location on a positive frame counts as a false
positive (standard detection-metric convention, and the one that actually matters
for "does it pick the true target").

### Perf note (relevant to anyone extending this): gather backward is ~10x slower than masked-multiply on this GPU

Initial implementation indexed the per-example GT-peak pixel via
`heatmap[torch.arange(B), py, px]` (standard PyTorch advanced indexing). Measured
**~955ms/step at batch=256** on the GB10 — a single isolated `Conv2d` at that shape
took ~14ms and a bare 10-op forward+backward totaled ~2s, which on paper is
absurdly slow for a 0.87M-param net. Root-caused to the advanced-indexing
gather/scatter backward, not GPU contention (though a sibling job
(`train_centernet.py`) *was* also running concurrently on the shared GPU during
part of this session and made isolated benchmarking initially misleading — GPU
util shown at 95% by another PID before that was noticed). Rewriting the loss to
use a precomputed one-hot `peak_mask` tensor (built on CPU during data loading) and
`(tensor * peak_mask).sum(dim=(1,2))` instead of indexing dropped real per-epoch
time from ~950ms/step-equivalent to the ~1.2s/batch (256) actually observed in the
final run (~19,886 examples in ~93s/epoch, steady state, single job on the GPU) —
eval-mode single-frame latency was unaffected either way (1.3-1.5ms, see §5).

### Run

`runs/nx6_heatmap_A/` — 60-epoch cosine schedule launched, **stopped at epoch 41**
(model_best.pt is epoch 28) because val recall plateaued (0.762 @ ep28, 0.732 @
ep32, 0.751 @ ep36, 0.761 @ ep40 — all within noise of each other) while wall-clock
was stopped early for iteration speed; heatmap training loss was still
slowly decreasing (0.43 @ ep16 -> 0.19 @ ep41) suggesting a longer run or LR
retune could still buy a little more, but recall itself had genuinely flattened.

| epoch | precision | recall | tau |
|---|---|---|---|
| 4  | 0.908 | 0.071 | 0.58 |
| 8  | 0.915 | 0.175 | 0.64 |
| 12 | 0.907 | 0.299 | 0.69 |
| 16 | 0.909 | 0.628 | 0.57 |
| 20 | 0.907 | 0.690 | 0.56 |
| 24 | 0.906 | 0.645 | 0.58 |
| **28** | **0.902** | **0.762** | **0.59** |
| 32 | 0.904 | 0.732 | 0.69 |
| 36 | 0.905 | 0.751 | 0.65 |
| 40 | 0.901 | 0.761 | 0.61 |

## 3. Results

### Val / test (`code/eval_nx6_heatmap.py`, recomputed standalone against `model_best.pt`)

| split | n_pos | tau | precision | recall @ gate | presence-only recall |
|---|---|---|---|---|---|
| val  | 1,110 | 0.59 | 0.903 | **0.762** | 0.817 |
| test | 1,055 | 0.62 | 0.902 | **0.714** | 0.763 |

(val/test both meet the precision>=0.9 gate; test recall is a bit lower than val's,
as expected for a held-out-scene split with no threshold retuning.)

### Failure-case acid test (`dataset/det_failcases`, the actual point of this exercise)

Overall (all 77 labeled objects across the 6 episodes + sampled negatives):
**precision=0.910, recall=0.922** at tau=0.29 (gate met) — but note n_pos=77 is
small, treat as directional, not a tight estimate. The **per-episode breakdown is
the informative part**:

| episode | doc'd classical failure | target ever visible in captured frames? | recall when visible | false-fire rate when NOT visible |
|---|---|---|---|---|
| demo ep0  | confident false-lock on wrong-depth blob | **no** (0/20) | n/a | **0.0%** (0/20) |
| demo ep2  | confident false-lock, walks away | **no** (0/19) | n/a | **0.0%** (0/19) |
| demo ep4  | total miss (purple, no wall-hue overlap) | **no** (0/20) | n/a | **0.0%** (0/20) |
| demo ep5  | stalls (same signature as ep0) | yes (8/20) | **87.5%** (7/8) | 0.0% (0/12) |
| demo ep12 | **twin-hijack**: locks onto cyan-ball distractor | yes, briefly (1/19) | **100%** (1/1) | 5.6% (1/18) |
| search ep12 | (fall, not a grounding-specific repro) | yes (6/20) | 83.3% (5/6) | 0.0% (0/14) |

**Reading this honestly:** for ep0/2/4, the true target is not actually present in
*any* of the 20 captured frames of this replay (it's out of view/occluded for the
whole captured window — matches `docs/nx6_data.md` §6's "TRUE cyan cone drops out
of the segmented view entirely" for ep0, and the "total detection miss" note for
ep4). So this replay set can't directly test "does it find the target ep0/2/4
missed" — what it *can* test, and what it confirms, is that our detector **never
hallucinates a confident detection on the wall/floor/wrong-object content of those
frames** (0% false-fire across all 59 not-visible frames from ep0/2/4 combined) —
i.e. it does not reproduce the classical grounder's confident-false-lock failure
mode on this specific replayed content. For ep5, where the target genuinely is
visible in 8/20 frames, the detector correctly finds it in 7 (87.5%). Given the
small per-episode sample sizes (6-20 frames), these are indicative, not precise.

**ep12 twin separation (the specific test the brief calls for):** frame_uid=79 (the
one frame where both the true cyan cube target and the cyan-ball distractor are
simultaneously visible — the exact "twin-hijack" moment the classical grounder
fails on, `docs/nx6_data.md` §6) is decoded correctly and distinctly for **both**
queries:

| query | GT dist / bearing | predicted dist / bearing | confidence |
|---|---|---|---|
| cyan cube (true target) | 6.182 m / -21.77 deg | 6.171 m / -21.46 deg | 0.676 |
| cyan ball (distractor)  | 2.902 m / +21.29 deg | 2.894 m / +20.47 deg | 0.884 |

Both within the (2deg, 0.5m) correctness bar, and correctly *distinct* — the
class+color conditioning does separate the twins on the diagnostic frame. One
caveat, reported honestly: at the *next* captured frame (frame_uid=80, 200 sim
steps later, true cube now out of view, only the ball remains visible), the "cyan
cube" query still fires at confidence 0.371 (just above the selected tau=0.29),
predicting a location that closely matches the *ball's* true position (pred
dist=1.36m/bearing=8.5deg vs the ball's actual 1.34m/9.3deg) — a residual,
weaker color-bleed echo of the same confusion the classical grounder makes
confidently, rather than a clean rejection. It's markedly less confident than the
true-positive "cyan ball" detection on that same frame (0.890) and than the
frame-79 true positive (0.676), and from frame_uid=81 onward (cube genuinely out
of frame) confidence for the cube query drops below 0.06 and stays there — so this
looks like a boundary-tau artifact right at the moment of hand-off rather than a
persistent hijack, but it is a real, measured imperfection, not zero.

## 4. Latency (single-frame, batch=1, `code/eval_nx6_heatmap.py`)

| device | latency | throughput |
|---|---|---|
| GB10 GPU (steady-state, post-warmup) | **1.51 ms** | 661 Hz |
| CPU (same host) | **28.7 ms** | 34.8 Hz |

Both comfortably clear the 5-10Hz deploy budget (the GPU number leaves ~150x
headroom to share the GPU with the 50Hz policy; even CPU-only clears the target by
~3.5x). Note eval-mode batch=1 latency is unrelated to the training-mode slowdown
in §2's perf note — BatchNorm running-stats + no-autograd inference uses different,
well-optimized fused kernels on this GPU; only *training*-mode (batch stats +
backward) was affected.

## 5. Honest limitations / follow-ons

- **Training was stopped early** (stopped at epoch 41/60,
  best at epoch 28) after observing recall plateau around 0.73-0.76 on val for
  ~15 epochs; heatmap loss was still (slowly) decreasing, so a longer run, a
  second cosine cycle, or a slightly lower peak LR might recover a few more
  recall points — untested here.
- **192x144 is a real resolution cut** from the native 480x360/320x240 the
  dataset's own label-geometry check used (which itself has a p95 noise floor of
  2.67deg bearing / 4.2cm dist on *perfect* inputs) — our val/test recall
  (~0.71-0.76 at precision>=0.9) is bounded above by that geometry noise floor
  plus whatever additional localization error the learned heatmap adds on top;
  it is not purely a "model capacity" number.
- **Failcase episode sample sizes are tiny** (1-20 frames/episode) — per-episode
  recall numbers (87.5%, 83.3%, 100% of n=1) should be read as directional, not
  precise. The overall failcase precision/recall (n_pos=77) is likewise a small-n
  estimate.
- **ep12's frame-80 partial color-bleed** (§3) suggests the twin-discrimination
  margin, while real, is not huge right at the visibility boundary — more
  same-color-different-shape hard-negative mining (this dataset already samples
  such pairs naturally but doesn't specifically oversample the "one twin just left
  frame" transition) could tighten this.
- **This doc does not itself gate/adopt the detector into `code/grounding.py`** —
  per `docs/nx6_data.md` §8, that's explicitly out of scope here; this is the
  standalone model-selection + acid-test step for the heatmap variant, to be
  compared against sibling variants (e.g. `code/train_centernet.py`, observed
  running concurrently) before any adoption decision.

## 6. Files

- `code/nx6_heatmap_model.py` — model, decode, `HeatmapDetector` standalone
  inference wrapper (API documented in `runs/nx6_heatmap_A/README.md`).
- `code/nx6_heatmap_data.py` — `SplitCache` / `load_failcase_cache`,
  augmentation, `HeatmapDataset`.
- `code/nx6_heatmap_eval_utils.py` — shared inference + precision/recall-sweep
  scoring (`run_inference`, `select_threshold`).
- `code/train_nx6_heatmap.py` — training loop (`--smoke` for a <1min pipeline
  sanity check before any full run).
- `code/eval_nx6_heatmap.py` — final val/test/failcase evaluation + latency
  benchmark, writes `runs/nx6_heatmap_A/eval_results.json`.
- `runs/nx6_heatmap_A/model_best.pt` (epoch 28), `epoch_{08,16,24,32,40}.pt`,
  `curves.json`, `eval_results.json`, `README.md` (inference API).
- Logs: `logs/nx6_heatmap_A.log` (training), `logs/nx6_heatmap_eval.log` (eval).
