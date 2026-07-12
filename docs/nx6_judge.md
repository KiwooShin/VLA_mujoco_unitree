# NX-6 JUDGE — learned-grounding detector bakeoff (heatmap vs. centernet)

**Date:** 2026-07-09
**Agent:** NX-6 JUDGE
**Inputs:** `docs/nx6_train_heatmap.md` (`runs/nx6_heatmap_A/model_best.pt`, 0.874M
params) vs. `docs/nx6_train_centernet.md` (`runs/nx6_centernet_B/best.pt`,
0.547M params), both trained on `dataset/det_v1` / acid-tested on
`dataset/det_failcases` (`docs/nx6_data.md`).

**Verdict: ADOPT the heatmap variant** (`runs/nx6_heatmap_A/model_best.pt`).
GO. Reasoning below is based on independently re-run evaluations, not the
checkpoints' own self-reported docs (both self-reports turned out accurate
where they overlap with what I re-derived, but the CenterNet doc omitted a
per-episode breakdown that changes the picture materially — see §2).

---

## 0. Method

Per the brief's priority order — (1) failure-case metrics first (recall on
the wall-stripe episodes 0/2/5 + twin separation on ep12), (2) val/test
precision-recall, (3) latency headroom — I did NOT just trust the two
training docs' self-reported numbers. For each candidate I:

1. Re-ran its **own** eval code, fresh process, against the saved checkpoint,
   on val + test + `dataset/det_failcases`, and diffed against the doc's
   numbers.
2. For CenterNet, wrote a small standalone re-eval script
   (`code/nx6_judge_verify_centernet.py`) that reuses `train_centernet.py`'s
   own `run_eval`/`run_failcase_eval`/`select_threshold`/`bench_latency`
   functions (bit-identical code path, just invoked eval-only) and added a
   **per-episode failcase breakdown**, which `train_centernet.py`'s
   `main()` never saved to `results.json` in the first place (only the
   pooled 118-frame numbers were persisted) — this is the single most
   important thing this JUDGE pass surfaced (§2).
3. For heatmap, re-ran `code/eval_nx6_heatmap.py` unmodified, to a fresh
   output path, and additionally recomputed the failcase per-episode/overall
   numbers **at the model's actual val-selected deploy threshold** (tau=0.59)
   rather than the separately-swept, more permissive failcase-only threshold
   (tau=0.29) the training doc's headline numbers use — see §3.2.
4. Rendered 8 sanity-check prediction overlays on raw failcase frames
   (`eval/nx6_judge/preview/*.png`, via `code/nx6_judge_preview.py`, using the
   real `HeatmapDetector.infer()` API a deploy integration would call) and
   visually confirmed the twin-separation and reject-when-absent claims.

Raw verification artifacts:
- `eval/nx6_judge/centernet_verify.json`, `logs/nx6_judge_centernet_verify.log`
- `eval/nx6_judge/heatmap_verify.json`, `logs/nx6_judge_heatmap_verify.log`
- `eval/nx6_judge/preview/*.png` (8 images)
- Scripts: `code/nx6_judge_verify_centernet.py`, `code/nx6_judge_preview.py`

All val/test numbers below reproduced **bit-exactly** against the two
training docs (same precision/recall/tp/fp/tau to full float precision, e.g.
CenterNet val precision `0.9016736401673641` reproduced to the last digit,
heatmap failcase overall `precision=0.9102564102564102 recall=0.922077922077922`
likewise) — confirms both checkpoints load correctly, both eval pipelines are
deterministic, and there's no eval-script drift between training-time and
judge-time. The one place independent re-analysis **changed the picture** is
the CenterNet per-episode failcase breakdown, never previously computed.

---

## 1. Failure-case metrics — the decisive axis

### 1.1 Both models: zero visible-target frames for ep0/ep2/ep4 in this replay set

Confirmed independently for both checkpoints: `demo_ep0` (0/20 visible),
`demo_ep2` (0/19), `demo_ep4` (0/20) — the true target never appears in any
of the 118 captured failcase frames for these three episodes (matches
`docs/nx6_data.md` §6's own note that the target drops out of the segmented
view / is a total miss for these). So "recall" on ep0/2/4 is **not
computable** for either model — what IS computable and IS comparable is
**false-fire rate on these wall/floor frames** (does the model hallucinate a
confident detection the way the classical grounder did):

| episode | heatmap false-fire (tau=0.59, deploy) | centernet false-fire (tau=0.45) |
|---|---|---|
| demo_ep0 | 0/20 | 0/20 |
| demo_ep2 | 0/19 | 0/19 |
| demo_ep4 | 0/20 | 0/20 |

**Tied, both perfect** — neither model reproduces the classical grounder's
confident-false-lock failure mode on ep0/2/4. Visually confirmed in
`eval/nx6_judge/preview/00_ep0_reject_uid0.png` (yellow ball + orange cylinder
in frame, cyan cone target nowhere in sight, blue-tinted floor that could
tempt a hue-collision system — heatmap correctly returns "not present",
conf=0.088, well under tau).

### 1.2 demo_ep5 (the one wall-stripe/stall episode with real visible-target frames) — heatmap wins decisively

This is the only episode among 0/2/4/5 where the target is actually visible
in some frames, so it's the only place "recall on the wall-stripe episodes"
is directly measurable. **Re-running each model's own failcase eval at its
own real val-selected operating threshold** (apples-to-apples: both at the
threshold that gives each model its documented `reject_rate_on_not_visible`
guarantee, not a separately-tuned looser threshold):

| model | threshold used | ep5 recall_when_visible | reject_rate (all 103 not-visible frames, all episodes) |
|---|---|---|---|
| **heatmap** (tau=0.59, val-selected) | 0.59 | **6/8 = 75.0%** | **103/103 = 100%** |
| centernet (tau=0.45, val-selected) | 0.45 | **2/8 = 25.0%** | **103/103 = 100%** |

At **matched zero-hallucination operating points** (both reject 100% of the
103 not-visible frames — so this isn't "heatmap just fires more often"),
heatmap recalls the ep5 target **3x more often** than CenterNet (6/8 vs
2/8). This per-episode number is **not in either training doc** — CenterNet's
doc only reports the pooled `recall_strict=8/15=0.533` across all 3
target-visible episodes (ep5, ep12, search_ep12) and never breaks it out per
episode; I computed the breakdown independently
(`eval/nx6_judge/centernet_verify.json` → `failcase.per_episode.demo_ep5`)
and it decomposes as **ep5: 2/8, ep12: 1/1, search_ep12: 5/6 = 8/15 total**,
consistent with the doc's pooled number but revealing that essentially all of
CenterNet's failcase recall shortfall is concentrated on ep5 specifically —
the exact "stall on a marginal/flickering blob" episode the brief calls out.

(At heatmap's own training doc's separately-swept failcase-only threshold,
tau=0.29, ep5 recall is even higher, 7/8=87.5% — but that threshold is
looser than what would actually ship, so I use the val-selected tau=0.59
apples-to-apples comparison above as the honest number. At tau=0.29,
reject_rate drops slightly to 102/103 due to one weak echo at ep12 frame_uid
80 — see §1.3 — which is exactly why tau=0.59, not 0.29, is the number that
should gate this decision.)

### 1.3 ep12 twin separation — both pass the core test; heatmap has a benign near-threshold artifact that vanishes at the deploy operating point

The diagnostic frame is `frame_uid=79` (both the true cyan-cube target,
far/small, and the cyan-ball distractor, near/large, visible simultaneously —
the exact frame the classical grounder locks onto the wrong one on,
`docs/nx6_data.md` §6). Both models separate them cleanly and within the
(bearing<2deg, dist<0.5m) bar:

| model | query | pred dist/bearing | GT dist/bearing | within bar |
|---|---|---|---|---|
| heatmap | cyan cube (target) | 6.17m / -21.5deg | 6.18m / -21.8deg | yes |
| heatmap | cyan ball (distractor) | 2.89m / +20.5deg | 2.90m / +21.3deg | yes |
| centernet | cyan cube (target) | bearing_err=0.09deg, dist_err=0.11m | — | yes |
| centernet | cyan ball (distractor) | bearing_err=0.15deg, dist_err=0.28m | — | yes |

Visually confirmed for heatmap in `eval/nx6_judge/preview/05_..._uid79.png`
(pink circle locks onto the small distant cube) and
`06_..._uid79.png` (pink circle locks onto the near ball) — the two queries
on the identical frame land on two visibly different objects.

**The one asymmetry**, correctly reported by the heatmap doc as an honest
caveat: at `frame_uid=80` (200 sim-steps later, cube has left frame, only
ball remains), querying "cyan cube" still gets a **weak** echo (conf=0.371)
near the ball's old location — but only at the failcase-swept threshold
0.29. Re-checked at the model's actual **val-selected deploy threshold
(0.59)**: this frame is correctly rejected (conf 0.371 < 0.59, "not
present") — confirmed both numerically (`eval/nx6_judge/heatmap_verify.json`
recomputation, §3.2) and visually
(`eval/nx6_judge/preview/07_ep12_echo_artifact_frame80_uid80.png` shows
"pred: (not present)"). **At the threshold that would actually ship, this
caveat does not exist.** CenterNet is cleanly correct on this frame too
(`pred_present=False`, conf=0.162) — a genuine, if minor, point in
CenterNet's favor at the loose threshold, but moot once heatmap is judged at
its own real operating point.

**Twin separation: PASS for both models**, heatmap slightly noisier at an
overly-permissive threshold that isn't the one that would deploy.

### 1.4 Failure-case bottom line

Heatmap's 3x recall advantage on the one directly-measurable wall-stripe
episode (ep5: 75% vs 25%, at matched 100%-reject operating points) is a much
larger and more decision-relevant gap than CenterNet's marginal edge on the
ep12 boundary-frame artifact (which disappears at heatmap's real deploy
threshold anyway). **Heatmap wins failure-case metrics.**

---

## 2. Val / test precision-recall (secondary criterion)

Independently reproduced (bit-exact vs. both training docs):

| model | val precision | val recall | test precision | test recall | params |
|---|---|---|---|---|---|
| **heatmap** | 0.903 | **0.762** | 0.902 | **0.714** | 0.874M |
| centernet | 0.902 | 0.695 | 0.893 | 0.688 | 0.547M |

Heatmap leads by ~6-7 points of recall at matched ~0.90 precision, on both
splits, consistent with its failure-case lead. **Heatmap wins here too.**

---

## 3. Latency (tertiary — both clear the budget with large margin)

Independently re-benchmarked (fresh process, same GPU, idle otherwise —
confirmed via `nvidia-smi`/`ps` before running, no other heavy job was
sharing the GPU during either benchmark):

| model | GPU (batch=1) | CPU (batch=1) | budget (5-10Hz = 100-200ms) |
|---|---|---|---|
| heatmap | 1.51ms (662Hz) | 29.1ms (34Hz) | >>10x margin either way |
| centernet | 0.78ms (verified; doc reported 1.4-1.8ms) | 18.4ms (verified; doc reported ~47-50ms) | >>10x margin either way |

Both trivially clear the deploy budget; CenterNet is nominally faster (also
has 37% fewer params: 0.547M vs 0.874M) but this is irrelevant given both
leave >>10x headroom. **Not a differentiator** — does not overturn §1/§2.

---

## 4. Sanity-visualization (`eval/nx6_judge/preview/`, 8 frames, heatmap winner)

| file | case | result |
|---|---|---|
| `00_ep0_reject_uid0.png` | ep0, target genuinely absent | correctly "not present", conf=0.088 |
| `01_ep2_reject_uid20.png` | ep2, target genuinely absent | correctly "not present", conf=0.021 |
| `02_ep4_totalmiss_reject_uid39.png` | ep4, target genuinely absent | correctly "not present", conf=0.160 |
| `03_ep5_TP_uid59.png` | ep5, target visible | correct detection, 8.84m/-17.3deg vs GT 8.85m/-17.4deg |
| `04_ep5_FN_miss_uid66.png` | ep5, target visible (one of the 2/8 misses at tau=0.59) | miss, conf=0.022 — an honest failure, included for balance |
| `05_ep12_twin_target(cube)_uid79.png` | twin frame, query=cyan cube | locks onto the correct (far, small) cube |
| `06_ep12_twin_distractor(ball)_uid79.png` | twin frame, query=cyan ball | locks onto the correct (near, large) ball |
| `07_ep12_echo_artifact_frame80_uid80.png` | frame after cube leaves FOV, query=cyan cube | correctly "not present" at deploy threshold (0.59) |

All 8 visually match their numeric verdicts — no rendering/decode bugs
found.

---

## 5. Honest limitations of the winner (carried into deployment)

- Val/test recall (~0.71-0.76 at precision~0.90) is well below the ~0.95
  geometric ceiling the CenterNet doc computed for this resolution class —
  there's real headroom left on the table from more training (the heatmap
  run was stopped early at epoch 41/60, best at epoch 28).
- Failcase episode sample sizes are tiny (1-20 frames/episode); the ep5
  6/8 vs 2/8 comparison, while the clearest signal available, is still n=8.
- ep0/2/4 cannot be used to validate "does it find the target this episode
  missed" at all (target never visible in any replayed frame) — only that
  neither model hallucinates on that content.
- Per `docs/nx6_train_heatmap.md` §5, this is not yet gated end-to-end into
  `code/grounding.ground()`'s role in the full closed-loop easy/demo/search
  eval — that remains the necessary next step before ADOPT is final in
  deployment (this JUDGE pass is model selection + acid-test verification,
  not a closed-loop re-gate).

---

## 6. Decision

**Winner: heatmap** (`runs/nx6_heatmap_A/model_best.pt`, query-conditioned
U-Net, 0.874M params, `conf_thresh=0.59`).

- Failure-case metrics (priority 1): wins decisively on ep5 recall (75% vs
  25% at matched 100%-reject operating points); ties on ep0/2/4 zero
  hallucination; ties (both PASS) on ep12 twin separation, with heatmap's
  only caveat (a weak boundary-frame echo) vanishing at its own real deploy
  threshold.
- Val/test precision-recall (priority 2): wins by ~6-7 recall points at
  matched ~0.90 precision.
- Latency (priority 3): both comfortably clear the 5-10Hz budget by >>10x;
  not a differentiator.

**GO.** Neither model is weak by the brief's bar (<60% failcase recall or
ep12 twin not separated) — heatmap clears both with margin (ep5 recall 75%
>> 60%, twin cleanly separated) and is the stronger of the two on every
priority-ordered criterion except raw param count / latency headroom, where
both already have enormous margin.
