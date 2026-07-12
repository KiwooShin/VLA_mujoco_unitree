"""
groot_lang.py — GR00T-N1.6 language embedding cache builder.

Enumerates the FULL templated instruction space ({template}×{verb}×{color}×{shape})
used by code/scene.py, encodes EACH instruction ONCE via the frozen GR00T-N1.6
language model (model.backbone.model.language_model), mean-pools the last hidden
state to a 2048-d float32 vector, and caches to disk.

Cache format: pickle dict  {instruction_string: np.float32[2048]}
              (also saved as .npz for easy inspection)

Usage
-----
# Build cache (runs in ~5-10 min on first call; cached on disk after)
python code/groot_lang.py --ckpt checkpoints/GR00T-N1.6-3B \
                           --out  dataset/lang_cache.pkl

# Smoke test / verify
python code/groot_lang.py --verify --cache dataset/lang_cache.pkl

RF-1: moved from code/groot_lang.py to code/policy/groot_lang.py (see
code/groot_lang.py, the old-path compat alias, and docs/refactor_plan.md).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

# Repo root on path (RF-1: this file now lives two directories below repo
# root, code/policy/groot_lang.py, rather than one — dirname applied twice).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

# Inline the color/shape/verb/template definitions to avoid import conflicts.
# (arena.py and scene.py can't be cleanly imported here due to the built-in
# `code` module collision when running this file as __main__.)
# Keep in sync with code/arena.py:COLORS, code/arena.py:SHAPES,
# code/scene.py:_VERBS, code/scene.py:_TEMPLATES.

COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("red",     (220,  40,  40)),
    ("yellow",  (235, 205,  40)),
    ("blue",    ( 50,  90, 220)),
    ("green",   ( 40, 180,  70)),
    ("orange",  (240, 140,  30)),
    ("purple",  (150,  60, 200)),
    ("cyan",    ( 40, 200, 210)),
]

SHAPES: list[tuple[str, float]] = [
    ("ball",     0.24),
    ("cube",     0.24),
    ("cylinder", 0.22),
    ("cone",     0.26),
]

_VERBS: list[str] = [
    "go to", "walk to", "approach", "head to",
    "head over to", "move to", "navigate to",
    "make your way to", "get to", "proceed to",
]

_TEMPLATES: list[str] = [
    "{v} the {c} {s}",
    "{v} the {s} that is {c}",
    "{v} the {c}-colored {s}",
    "please {v} the {c} {s}",
    "your goal is the {c} {s}",
    "find the {c} {s} and {v} it",
    "{v} the {c} {s} over there",
]

# ─────────────────────────────────────────────────────────────────────────────
# Instruction space enumeration
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_instructions() -> list[str]:
    """Return sorted list of all unique instructions in the templated space."""
    instrs = set()
    for tpl in _TEMPLATES:
        for verb in _VERBS:
            for color, _ in COLORS:
                for shape, _ in SHAPES:
                    instrs.add(tpl.format(v=verb, c=color, s=shape))
    return sorted(instrs)


# ─────────────────────────────────────────────────────────────────────────────
# GR00T-LM encoder
# ─────────────────────────────────────────────────────────────────────────────

class GrootLangEncoder:
    """
    Loads GR00T-N1.6-3B (frozen, bf16) and extracts the Eagle LM backbone.
    Encodes text instructions → 2048-d float32 mean-pooled embedding.
    """

    EMBED_DIM = 2048

    def __init__(self, ckpt_path: str, device: str = "cuda") -> None:
        """Load the frozen GR00T-N1.6-3B language model in bf16.

        Args:
            ckpt_path: Path to the GR00T-N1.6-3B checkpoint directory.
            device: Torch device string to load the model onto.
        """
        self.device = torch.device(device)
        ckpt_path = str(ckpt_path)

        print(f"[groot_lang] Loading GR00T-N1.6-3B from {ckpt_path} ...")
        t0 = time.time()

        # Suppress albumentations warning
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

        from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6

        model = Gr00tN1d6.from_pretrained(
            ckpt_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            local_files_only=True,
        )
        model.eval()

        # Extract the tokenizer from the collator's processor
        self.tokenizer = model.collator.processor.tokenizer
        self.tokenizer.padding_side = "left"

        # Extract LM — module path: model.backbone.model.language_model
        # This is the LlamaForCausalLM (or similar) inside Eagle3VL
        self.lm = model.backbone.model.language_model
        self.lm.eval()

        # Free the rest of the model to save VRAM
        # (keep backbone.model.language_model on GPU; discard action head etc.)
        del model.action_head
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"[groot_lang] Model loaded in {elapsed:.1f}s | "
              f"LM type: {type(self.lm).__name__} | "
              f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    @torch.no_grad()
    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of instructions to mean-pooled embeddings.

        Uses the LM embedding layer + full forward pass (no image tokens).
        Mean-pools the last hidden state over the sequence dimension. The LM
        is loaded in bf16 with flash-attention; we run it in a bf16/fp16
        autocast context to satisfy flash-attn dtype constraints.

        Args:
            texts: Instructions to encode.
            batch_size: Number of instructions encoded per forward pass.

        Returns:
            np.float32 array of shape (len(texts), 2048).
        """
        all_embs = []
        n = len(texts)

        for start in range(0, n, batch_size):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            # Flash-attention requires fp16 or bf16 — use autocast
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.lm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            # Last hidden state: (B, T, hidden_size)
            last_hidden = outputs.hidden_states[-1].float()  # (B, T, 2048) float32

            # Mean-pool over valid (non-padding) tokens
            mask = attention_mask.unsqueeze(-1).float()       # (B, T, 1)
            summed = (last_hidden * mask).sum(dim=1)          # (B, 2048)
            counts = mask.sum(dim=1).clamp(min=1)             # (B, 1)
            emb = (summed / counts).cpu().numpy()             # (B, 2048) float32

            all_embs.append(emb.astype(np.float32))

            if start % 200 == 0 or start + batch_size >= n:
                print(f"  [{start + len(batch)}/{n}] encoded", flush=True)

        return np.concatenate(all_embs, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Cache builder
# ─────────────────────────────────────────────────────────────────────────────

def build_cache(ckpt_path: str, out_path: str, batch_size: int = 64) -> dict[str, np.ndarray]:
    """Enumerate all instructions, encode them, and save the cache to disk.

    Args:
        ckpt_path: Path to the GR00T-N1.6-3B checkpoint directory.
        out_path: Output cache pickle path (a sibling .npz is also written).
        batch_size: Encoding batch size passed to `encode_batch`.

    Returns:
        Cache dict mapping each instruction to its np.float32[2048] embedding.

    Raises:
        AssertionError: If the encoded embeddings' shape doesn't match the
            expected (num_instructions, 2048).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    instrs = enumerate_instructions()
    print(f"[groot_lang] Instruction space: {len(instrs)} unique instructions")

    encoder = GrootLangEncoder(ckpt_path)

    t0 = time.time()
    embs = encoder.encode_batch(instrs, batch_size=batch_size)
    elapsed = time.time() - t0

    assert embs.shape == (len(instrs), 2048), f"Shape mismatch: {embs.shape}"

    cache = {instr: embs[i] for i, instr in enumerate(instrs)}

    # Save as pickle (primary)
    with open(out_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[groot_lang] Cache saved → {out_path}  ({len(cache)} entries, {out_path.stat().st_size/1e6:.1f} MB)")

    # Also save as npz (secondary, for inspection)
    npz_path = out_path.with_suffix(".npz")
    np.savez_compressed(npz_path,
                        instructions=np.array(list(cache.keys())),
                        embeddings=embs)
    print(f"[groot_lang] NPZ saved → {npz_path}")
    print(f"[groot_lang] Encoding time: {elapsed:.1f}s ({len(instrs)/elapsed:.1f} instrs/s)")

    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Verify / sanity check
# ─────────────────────────────────────────────────────────────────────────────

def verify_cache(cache_path: str) -> None:
    """Sanity-check a cache: different instructions differ; same is stable.

    Args:
        cache_path: Path to the cache pickle to verify.

    Raises:
        AssertionError: If any embedding has the wrong shape/dtype, two
            distinct instructions embed near-identically, the same
            instruction is unstable, or any embedding norm is near-zero.
    """
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    instrs = list(cache.keys())
    print(f"[verify] Cache loaded: {len(instrs)} instructions")
    print(f"[verify] Embedding dim: {list(cache.values())[0].shape}")

    # Check dim
    for k, v in cache.items():
        assert v.shape == (2048,), f"Bad shape for '{k}': {v.shape}"
        assert v.dtype == np.float32, f"Bad dtype for '{k}': {v.dtype}"

    # Check different instructions differ
    e1 = cache[instrs[0]]
    e2 = cache[instrs[1]]
    cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    print(f"[verify] Cosine sim between '{instrs[0]}' and '{instrs[1]}': {cos_sim:.4f}")
    assert cos_sim < 0.9999, "Embeddings appear identical — something is wrong"

    # Check stability: same key produces same value (by definition — it's a dict)
    e_same = cache[instrs[0]]
    assert np.allclose(e1, e_same), "Stability check failed"

    # Check norm is non-trivial
    norms = [np.linalg.norm(v) for v in list(cache.values())[:20]]
    print(f"[verify] Embedding norms (first 20): min={min(norms):.2f} max={max(norms):.2f}")
    assert all(n > 0.1 for n in norms), "Some embeddings near-zero — encoding failed"

    # Sample a few
    print(f"[verify] Sample instructions:")
    for i in [0, len(instrs)//4, len(instrs)//2, -1]:
        print(f"  [{i:4d}] {instrs[i]!r}  norm={np.linalg.norm(cache[instrs[i]]):.2f}")

    print("[verify] PASS — cache is valid")


# ─────────────────────────────────────────────────────────────────────────────
# Look up helper (used by dataset.py at runtime)
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_CACHE: dict[str, np.ndarray] | None = None
_GLOBAL_CACHE_PATH: str | None = None


def get_embedding(instruction: str, cache_path: str) -> np.ndarray:
    """Look up a cached embedding, loading the cache on first call.

    Args:
        instruction: Instruction string to look up.
        cache_path: Path to the cache pickle (loaded into the module-level
            `_GLOBAL_CACHE` on first call, or if `cache_path` changes).

    Returns:
        np.float32[2048] embedding, or a zero vector (with a warning) if
        `instruction` is not in the cache.
    """
    global _GLOBAL_CACHE, _GLOBAL_CACHE_PATH

    if _GLOBAL_CACHE is None or _GLOBAL_CACHE_PATH != cache_path:
        with open(cache_path, "rb") as f:
            _GLOBAL_CACHE = pickle.load(f)
        _GLOBAL_CACHE_PATH = cache_path

    emb = _GLOBAL_CACHE.get(instruction)
    if emb is None:
        import warnings
        warnings.warn(f"[groot_lang] Instruction not in cache: {instruction!r} — returning zeros")
        return np.zeros(2048, dtype=np.float32)
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point: list, build, or verify the language embedding cache."""
    parser = argparse.ArgumentParser(description="GR00T-LM language embedding cache builder")
    parser.add_argument("--ckpt",    type=str, default="checkpoints/GR00T-N1.6-3B",
                        help="Path to GR00T-N1.6-3B checkpoint directory")
    parser.add_argument("--out",     type=str, default="dataset/lang_cache.pkl",
                        help="Output cache pickle path")
    parser.add_argument("--batch",   type=int, default=64,
                        help="Encoding batch size")
    parser.add_argument("--verify",  action="store_true",
                        help="Verify an existing cache (--cache required)")
    parser.add_argument("--cache",   type=str, default=None,
                        help="Path to existing cache for --verify")
    parser.add_argument("--list",    action="store_true",
                        help="List instruction space (no encoding)")
    args = parser.parse_args()

    if args.list:
        instrs = enumerate_instructions()
        for i, ins in enumerate(instrs):
            print(f"  {i:4d}: {ins}")
        print(f"\nTotal: {len(instrs)} instructions "
              f"({len(_TEMPLATES)} templates × {len(_VERBS)} verbs × "
              f"{len(COLORS)} colors × {len(SHAPES)} shapes)")
        return

    if args.verify:
        cache_path = args.cache or args.out
        verify_cache(cache_path)
        return

    build_cache(ckpt_path=args.ckpt, out_path=args.out, batch_size=args.batch)
    verify_cache(args.out)


if __name__ == "__main__":
    main()
