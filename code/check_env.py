#!/usr/bin/env python3
"""
G1Nav Environment Verification Script
Checks all dependencies needed for the campaign.
Run with: /home/kiwoos/miniconda3/envs/g1nav/bin/python code/check_env.py
"""

import os
import sys
import traceback

# Force EGL for headless MuJoCo rendering before any import
os.environ["MUJOCO_GL"] = "egl"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

results = {}  # name -> (pass/fail, message)


def check(name, fn):
    try:
        msg = fn()
        results[name] = ("PASS", msg)
        print(f"[PASS] {name}: {msg}")
    except Exception as e:
        tb = traceback.format_exc()
        results[name] = ("FAIL", str(e))
        print(f"[FAIL] {name}: {e}")
        print(tb)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1: PyTorch + CUDA
# ─────────────────────────────────────────────────────────────────────────────
def check_torch():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Actual GPU matmul
    a = torch.randn(512, 512, device="cuda", dtype=torch.float32)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float32)
    c = torch.matmul(a, b)
    assert c.shape == (512, 512)
    torch.cuda.synchronize()

    return (
        f"torch={torch.__version__} | GPU={gpu_name} | VRAM={gpu_vram_gb:.1f}GB"
    )

check("1_torch_cuda", check_torch)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2: MuJoCo + EGL offscreen render of G1 model
# ─────────────────────────────────────────────────────────────────────────────
def check_mujoco():
    import mujoco
    # g1.xml references ../../../meshes/benchmark_bin_centered.stl (terrain mesh)
    # which is NOT in the repo. Use g1_gear_wbc.xml (robot-only, no terrain) instead.
    g1_xml = os.path.join(
        REPO_ROOT,
        "third_party/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl"
        "/gr00t_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.xml",
    )
    assert os.path.exists(g1_xml), f"G1 XML not found: {g1_xml}"

    model = mujoco.MjModel.from_xml_path(g1_xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=240, width=320)
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    rgb = renderer.render()
    assert rgb.shape == (240, 320, 3), f"Unexpected shape: {rgb.shape}"
    assert rgb.dtype.name == "uint8"
    renderer.close()

    return (
        f"mujoco={mujoco.__version__} | EGL render OK | shape={rgb.shape} | "
        f"model=g1_gear_wbc.xml (g1.xml skipped: missing terrain mesh)"
    )

check("2_mujoco_egl", check_mujoco)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3: onnxruntime + WBC ONNX policies
# ─────────────────────────────────────────────────────────────────────────────
onnx_io_report = {}

def check_onnxruntime():
    import onnxruntime as ort

    policy_dir = os.path.join(
        REPO_ROOT,
        "third_party/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl"
        "/gr00t_wbc/sim2mujoco/resources/robots/g1/policy",
    )
    walk_onnx = os.path.join(policy_dir, "GR00T-WholeBodyControl-Walk.onnx")
    balance_onnx = os.path.join(policy_dir, "GR00T-WholeBodyControl-Balance.onnx")

    for label, path in [("Walk", walk_onnx), ("Balance", balance_onnx)]:
        assert os.path.exists(path), f"ONNX not found: {path}"
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inputs = [
            {"name": i.name, "shape": i.shape, "dtype": i.type}
            for i in sess.get_inputs()
        ]
        outputs = [
            {"name": o.name, "shape": o.shape, "dtype": o.type}
            for o in sess.get_outputs()
        ]
        onnx_io_report[label] = {"inputs": inputs, "outputs": outputs}
        print(f"  [{label}] inputs:")
        for i in inputs:
            print(f"    {i['name']}: shape={i['shape']}  dtype={i['dtype']}")
        print(f"  [{label}] outputs:")
        for o in outputs:
            print(f"    {o['name']}: shape={o['shape']}  dtype={o['dtype']}")

    return f"onnxruntime={ort.__version__} | Walk+Balance loaded OK"

check("3_onnxruntime_wbc", check_onnxruntime)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4: gr00t imports + GR00T-N1.6-3B checkpoint
# ─────────────────────────────────────────────────────────────────────────────
groot_module_paths = {}

def check_groot():
    import torch
    import gr00t
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6

    ckpt_path = os.path.join(REPO_ROOT, "checkpoints/GR00T-N1.6-3B")
    assert os.path.isdir(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    print(f"  Loading GR00T-N1.6-3B from {ckpt_path} in bf16 on GPU...")
    model = Gr00tN1d6.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    model.eval()

    # Report VRAM
    vram_used_gb = torch.cuda.memory_allocated() / 1e9
    vram_reserved_gb = torch.cuda.memory_reserved() / 1e9

    # Discover module paths
    backbone = model.backbone  # EagleBackbone wrapping .model
    inner = backbone.model     # the AutoModel (Eagle3VL)

    # vision_model / language_model are attributes of inner
    vision_path = "model.backbone.model.vision_model"
    lm_path = "model.backbone.model.language_model"

    # Verify they exist
    _ = inner.vision_model
    _ = inner.language_model

    groot_module_paths["vision_tower"] = vision_path
    groot_module_paths["language_model"] = lm_path

    # Count params
    total_params = sum(p.numel() for p in model.parameters()) / 1e9

    return (
        f"gr00t={gr00t.__version__ if hasattr(gr00t,'__version__') else 'editable'} | "
        f"GR00T-N1.6-3B loaded OK | params≈{total_params:.2f}B | "
        f"VRAM_alloc={vram_used_gb:.2f}GB reserved={vram_reserved_gb:.2f}GB | "
        f"vision={vision_path} | lm={lm_path}"
    )

check("4_groot_n1d6", check_groot)


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5: OpenCV + imageio
# ─────────────────────────────────────────────────────────────────────────────
def check_media():
    import cv2
    import imageio
    import numpy as np

    # Quick sanity: create a small image and encode/decode
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    _, enc = cv2.imencode(".png", img)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    assert dec.shape == (64, 64, 3)

    return f"cv2={cv2.__version__} | imageio={imageio.__version__}"

check("5_opencv_imageio", check_media)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
all_pass = True
for name, (status, msg) in results.items():
    tag = "✓" if status == "PASS" else "✗"
    print(f"  {tag} {status:4s}  {name}")
    if status == "FAIL":
        all_pass = False
        print(f"         Error: {msg}")

overall = "GREEN" if all_pass else "RED"
print(f"\nOverall: {overall}")

# Print ONNX I/O detail again for easy capture
if onnx_io_report:
    print("\nWBC ONNX I/O shapes (for S2/S3):")
    for model_name, io in onnx_io_report.items():
        print(f"  {model_name}:")
        print(f"    inputs:  {[(i['name'], i['shape']) for i in io['inputs']]}")
        print(f"    outputs: {[(o['name'], o['shape']) for o in io['outputs']]}")

# Print GR00T module paths
if groot_module_paths:
    print("\nGR00T module paths:")
    for k, v in groot_module_paths.items():
        print(f"  {k}: {v}")

print("=" * 70)
