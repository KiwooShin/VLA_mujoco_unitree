"""
code.train.nx6_heatmap_eval — NX-6 TRAIN (heatmap variant): final evaluation.

1. VAL / TEST: recall @ (bearing<2deg, dist<0.5m) subject to precision>=0.9
   (same metric train_nx6_heatmap.py uses for model selection, recomputed here
   standalone against the saved checkpoint for an honest final number).
2. FAILCASES (dataset/det_failcases): the acid test.
   (a) overall precision/recall over all labeled objects + sampled negatives.
   (b) per-episode "instructed-target" query — the deploy-realistic case: query =
       the episode's actual (color,shape) instruction on every frame of that
       episode; does the model fire (confidently, correctly located) exactly when
       the true target is actually visible, and stay silent when it's not (the
       wall/floor-hue-collision false-lock the classical grounder suffers)?
   (c) ep12 twin-hijack: dual query (cyan cube vs cyan ball) on demo_ep12 frames —
       does conditioning actually separate the two?
3. Latency: single-frame (batch=1) GPU eval-mode inference, warm + steady-state;
   CPU latency too.

Usage: python code/eval_nx6_heatmap.py --ckpt runs/nx6_heatmap_A/model_best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("MUJOCO_GL", "egl")

from code.nx6_heatmap_data import SplitCache, load_failcase_cache, build_example_index
from code.nx6_heatmap_eval_utils import (InferenceResult, run_inference, select_threshold,
                                         presence_only_pr, _angle_diff_deg)
from code.nx6_heatmap_model import (TinyHeatmapUNet, N_CLASS, N_COLOR, TARGET_W, TARGET_H,
                                    CLASS_NAMES, COLOR_NAMES)


def load_model(ckpt_path: str, device: str) -> tuple[TinyHeatmapUNet, dict]:
    """Loads a TinyHeatmapUNet model and its checkpoint dict.

    Args:
        ckpt_path: Path to the model checkpoint file.
        device: Torch device string to move the model to.

    Returns:
        A tuple of (model in eval mode on device, raw checkpoint dict).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = TinyHeatmapUNet(**ckpt.get("model_cfg", {}))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def eval_split(
    model: TinyHeatmapUNet,
    data_root: str,
    split: str,
    device: str,
    neg_per_object_frame: int = 3,
    neg_per_empty_frame: int = 6,
    seed: int = 999,
) -> tuple[dict, InferenceResult]:
    """Evaluates the model on one dataset split (e.g. 'val' or 'test').

    Args:
        model: The TinyHeatmapUNet model to evaluate.
        data_root: Root directory of the dataset (e.g. dataset/det_v1).
        split: Split name to load via SplitCache (e.g. 'val', 'test').
        device: Torch device string to run inference on.
        neg_per_object_frame: Negative examples sampled per labeled-object frame.
        neg_per_empty_frame: Negative examples sampled per empty frame.
        seed: RNG seed for example-index sampling.

    Returns:
        A tuple (summary, res): a dict of split/precision/recall/etc. metrics,
        and the raw InferenceResult.
    """
    cache = SplitCache(data_root, split)
    rng = np.random.default_rng(seed)
    examples = build_example_index(cache, rng, neg_per_object_frame=neg_per_object_frame,
                                   neg_per_empty_frame=neg_per_empty_frame)
    res = run_inference(model, cache, examples, device, batch_size=256, num_workers=2)
    best, curve = select_threshold(res, min_precision=0.9, bearing_tol=2.0, dist_tol=0.5)
    presence = presence_only_pr(res, tau=best["tau"])
    return dict(split=split, n_examples=len(examples), n_pos=best["n_pos"], **{
        k: best[k] for k in ("tau", "precision", "recall", "tp", "fp", "n_detected", "met_precision_gate")
    }, presence_precision=presence["precision"], presence_recall=presence["recall"]), res


def build_instructed_target_examples(
    cache: SplitCache,
) -> list[tuple[int, int, int, dict | None]]:
    """One example per failcase frame: query = the episode's instructed (color,shape).
    has_target/dist_gt/bearing_gt come from the is_instructed_target=True label row if
    present in that frame (object actually visible), else 'not present'.

    Args:
        cache: SplitCache holding the failcases split's frames/labels.

    Returns:
        A list of (row_i, class_id, color_id, label_or_none) tuples, one per
        frame in ``cache``.
    """
    examples = []
    for i in range(len(cache.frames)):
        cname = str(cache.target_color[i])
        sname = str(cache.target_shape[i])
        class_id = CLASS_NAMES.index(sname)
        color_id = COLOR_NAMES.index(cname)
        lab = None
        for l in cache.row_labels[i]:
            if l.get("is_instructed_target"):
                lab = l
                break
        examples.append((i, class_id, color_id, lab))
    return examples


def eval_failcases(
    model: TinyHeatmapUNet,
    device: str,
    data_root: str = "dataset/det_failcases",
) -> dict:
    """Runs the failcases acid-test suite: overall PR, per-episode instructed-target
    query behavior, and the ep12 twin-hijack conditioning check.

    Args:
        model: The TinyHeatmapUNet model to evaluate.
        device: Torch device string to run inference on.
        data_root: Root directory of the failcases dataset.

    Returns:
        A dict with keys 'overall', 'per_episode', 'per_frame_instructed_target',
        and 'ep12_twin'.
    """
    cache = load_failcase_cache(data_root)

    # (a) overall precision/recall: every labeled object as a positive + sampled negatives
    rng = np.random.default_rng(999)
    all_examples = build_example_index(cache, rng, neg_per_object_frame=4, neg_per_empty_frame=6)
    res_all = run_inference(model, cache, all_examples, device, batch_size=128, num_workers=0)
    best_all, _ = select_threshold(res_all, min_precision=0.9, bearing_tol=2.0, dist_tol=0.5)
    presence_all = presence_only_pr(res_all, tau=best_all["tau"])

    # (b) per-episode instructed-target query
    instr_examples = build_instructed_target_examples(cache)
    res_instr = run_inference(model, cache, instr_examples, device, batch_size=64, num_workers=0)
    # use the tau selected on (a) (a fixed, principled operating point) -- report both the
    # detection decision at that tau AND the raw confidence/decoded values per frame.
    tau = best_all["tau"]
    per_frame = []
    for j, (row_i, class_id, color_id, lab) in enumerate(instr_examples):
        conf = float(res_instr.confidence[j])
        dist_pred = float(res_instr.dist_pred[j])
        bearing_pred = float(res_instr.bearing_pred[j])
        is_pos = lab is not None
        detected = conf >= tau
        if is_pos:
            berr = abs(float(_angle_diff_deg(bearing_pred, lab["bearing_gt"])))
            derr = abs(dist_pred - lab["dist_gt"])
            correct = detected and berr < 2.0 and derr < 0.5
            outcome = "TP" if correct else ("FN" if not detected else "detected_but_mislocalized")
        else:
            berr = derr = None
            correct = not detected
            outcome = "TN(correct_reject)" if correct else "FP(hallucinated)"
        per_frame.append(dict(
            frame_uid=int(cache.frame_uid[row_i]), ep_tag=str(cache.ep_tag[row_i]),
            step=int(cache.step[row_i]), cam_type=str(cache.cam_type[row_i]),
            query=f"{COLOR_NAMES[color_id]} {CLASS_NAMES[class_id]}",
            gt_visible=is_pos, confidence=conf, detected=bool(detected),
            dist_pred=dist_pred, bearing_pred=bearing_pred,
            dist_gt=(lab["dist_gt"] if is_pos else None),
            bearing_gt=(lab["bearing_gt"] if is_pos else None),
            bearing_err_deg=berr, dist_err_m=derr, outcome=outcome,
        ))

    per_episode = {}
    for ep in sorted(set(p["ep_tag"] for p in per_frame)):
        rows = [p for p in per_frame if p["ep_tag"] == ep]
        n_vis = sum(1 for r in rows if r["gt_visible"])
        n_correct_when_vis = sum(1 for r in rows if r["gt_visible"] and r["outcome"] == "TP")
        n_falsepos_when_not_vis = sum(1 for r in rows if (not r["gt_visible"]) and r["outcome"].startswith("FP"))
        n_not_vis = sum(1 for r in rows if not r["gt_visible"])
        per_episode[ep] = dict(
            n_frames=len(rows), n_visible=n_vis, n_not_visible=n_not_vis,
            correct_when_visible=n_correct_when_vis,
            recall_when_visible=(n_correct_when_vis / n_vis if n_vis else None),
            false_fire_when_not_visible=n_falsepos_when_not_vis,
            false_fire_rate_when_not_visible=(n_falsepos_when_not_vis / n_not_vis if n_not_vis else None),
        )

    # (c) ep12 twin-hijack: query BOTH cyan cube and cyan ball on every demo_ep12 frame
    twin = []
    ep12_rows = [i for i in range(len(cache.frames)) if str(cache.ep_tag[i]) == "demo_ep12"]
    twin_examples = []
    for row_i in ep12_rows:
        for (sname, cname) in [("cube", "cyan"), ("ball", "cyan")]:
            class_id = CLASS_NAMES.index(sname)
            color_id = COLOR_NAMES.index(cname)
            lab = next((l for l in cache.row_labels[row_i]
                       if l["class_id"] == class_id and l["color_id"] == color_id), None)
            twin_examples.append((row_i, class_id, color_id, lab))
    if twin_examples:
        res_twin = run_inference(model, cache, twin_examples, device, batch_size=32, num_workers=0)
        for j, (row_i, class_id, color_id, lab) in enumerate(twin_examples):
            twin.append(dict(
                frame_uid=int(cache.frame_uid[row_i]), step=int(cache.step[row_i]),
                query=f"{COLOR_NAMES[color_id]} {CLASS_NAMES[class_id]}",
                gt_visible=lab is not None,
                dist_gt=(lab["dist_gt"] if lab else None), bearing_gt=(lab["bearing_gt"] if lab else None),
                confidence=float(res_twin.confidence[j]), detected=bool(res_twin.confidence[j] >= tau),
                dist_pred=float(res_twin.dist_pred[j]), bearing_pred=float(res_twin.bearing_pred[j]),
                peak_px=None,
            ))

    return dict(
        overall=dict(n_examples=len(all_examples), n_pos=best_all["n_pos"], tau=best_all["tau"],
                    precision=best_all["precision"], recall=best_all["recall"],
                    met_precision_gate=best_all["met_precision_gate"],
                    presence_precision=presence_all["precision"], presence_recall=presence_all["recall"]),
        per_episode=per_episode,
        per_frame_instructed_target=per_frame,
        ep12_twin=twin,
    )


def benchmark_latency(model: TinyHeatmapUNet, device: str) -> float:
    """Benchmarks single-frame (batch=1) eval-mode inference latency.

    Runs a warmup pass (cuDNN algo search etc.) followed by 100 timed
    steady-state iterations.

    Args:
        model: The TinyHeatmapUNet model to benchmark (put in eval mode).
        device: Torch device string ('cuda' or 'cpu') to run inference on.

    Returns:
        Mean steady-state latency per inference call, in milliseconds.
    """
    x = torch.randn(1, 4, TARGET_H, TARGET_W, device=device)
    q = torch.zeros(1, N_CLASS + N_COLOR, device=device)
    q[0, 1] = 1.0
    q[0, N_CLASS + 6] = 1.0
    model.eval()
    with torch.no_grad():
        for _ in range(5):  # warmup (cuDNN algo search etc.)
            model(x, q)
        if device == "cuda":
            torch.cuda.synchronize()
        N = 100
        t0 = time.time()
        for _ in range(N):
            model(x, q)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / N
    return dt * 1000.0  # ms


def main() -> None:
    """Parses CLI args and runs the full NX-6 heatmap-model evaluation suite."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/nx6_heatmap_A/model_best.pt")
    ap.add_argument("--data", default="dataset/det_v1")
    ap.add_argument("--failcases", default="dataset/det_failcases")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/nx6_heatmap_A/eval_results.json")
    args = ap.parse_args()

    device = args.device
    model, ckpt = load_model(args.ckpt, device)
    print(f"[eval] loaded {args.ckpt} (epoch {ckpt.get('epoch')}), "
          f"params={sum(p.numel() for p in model.parameters())}", flush=True)

    t0 = time.time()
    val_summary, _ = eval_split(model, args.data, "val", device)
    print(f"[eval] VAL: {val_summary}", flush=True)
    test_summary, _ = eval_split(model, args.data, "test", device)
    print(f"[eval] TEST: {test_summary}", flush=True)

    fail_summary = eval_failcases(model, device, args.failcases)
    print(f"[eval] FAILCASES overall: {fail_summary['overall']}", flush=True)
    for ep, s in fail_summary["per_episode"].items():
        print(f"  {ep}: {s}", flush=True)
    print("[eval] ep12 twin separation:", flush=True)
    for row in fail_summary["ep12_twin"]:
        print(f"    {row}", flush=True)

    lat_gpu = None
    if torch.cuda.is_available():
        lat_gpu = benchmark_latency(model, "cuda")
        print(f"[eval] single-frame GPU latency (steady-state, batch=1): {lat_gpu:.3f} ms "
              f"({1000.0/lat_gpu:.1f} Hz)", flush=True)
    model_cpu = TinyHeatmapUNet(**ckpt.get("model_cfg", {}))
    model_cpu.load_state_dict(ckpt["model_state"])
    model_cpu.to("cpu").eval()
    lat_cpu = benchmark_latency(model_cpu, "cpu")
    print(f"[eval] single-frame CPU latency (batch=1): {lat_cpu:.3f} ms ({1000.0/lat_cpu:.1f} Hz)", flush=True)

    results = dict(
        ckpt=args.ckpt, ckpt_epoch=ckpt.get("epoch"), params=sum(p.numel() for p in model.parameters()),
        val=val_summary, test=test_summary, failcases=fail_summary,
        latency_ms=dict(gpu_batch1=lat_gpu, cpu_batch1=lat_cpu),
        eval_wall_time_s=time.time() - t0,
    )
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
