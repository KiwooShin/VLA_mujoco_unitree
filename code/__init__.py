"""G1Nav package init: EGL vendor-selection safety net."""
# GPU-rendering fix (2026-07-11): steer glvnd to the NVIDIA EGL ICD when
# present, BEFORE mujoco initializes EGL — otherwise Mesa can win the vendor
# race and MuJoCo silently renders on llvmpipe (CPU) at ~400 ms/frame vs
# ~1.3 ms on the GPU. Idempotent; no-op when the ICD file is absent or the
# user already chose a vendor. See code/arena.py for the measured numbers.
import os as _os
_NVIDIA_EGL_ICD = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if _os.path.exists(_NVIDIA_EGL_ICD):
    _os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _NVIDIA_EGL_ICD)
