"""
VQA sample construction for preference-conditioned VLA Stage A main-method.

Adds a per-episode visual QA cotrain on top of the baseline action stream.
The LM head is asked a binary preference question given a uniform 8-frame
head_camera clip from the episode; the answer is a single bare token whose
ID is registered per-category.

Design choices (locked by doc 0522-a-giveobj.md delta + 2026-05-22 chat,
extended for 3 new categories 2026-05-23):
  - Spec §2.1 (c): NO base prompt in VQA input. Decouples VQA accuracy from
    task-hint leakage; the most honest test of "can the LM head read the
    preference axis from pixels."
  - Spec §2.2 recommended path: 8-frame uniform-linspace clip from the
    head_camera (Qwen3-VL is a video VLM and will attend to the relevant
    moment on its own).
  - COW-safe RAM cache (gotcha from 2026-05-22 chat): a SINGLE contiguous
    numpy uint8 ndarray, NOT a list of PIL/Python objects. Numpy buffer
    reads don't touch Python refcounts of the data pages, so DataLoader
    workers fork-share the cache via OS COW without 4× duplication.
  - Single-bare-token answer (no length-normalization headache): logit-
    softmax over the two category-specific token ids gives a well-calibrated
    2-vector binary classifier.

Per-category answer text choices (verified single bare-token via Qwen3-VL-4B
tokenizer 2026-05-23):

  category=giveobj  Q="Question: low or high contact at grasp? Answer:"
                    25 -> "low"        (10303)
                    75 -> "high"       (11892)
  category=height   Q="Question: high or low drop? Answer:"
                    high -> "high"     (11892)
                    low  -> "low"      (10303)
  category=hvlv     Q="Question: far from or near the obstacle? Answer:"
                    hv -> "far"        (23559)
                    lv -> "near"       (51659)
                    NB: answer text != action-prompt label ("wide/narrow detour")
                    because "narrow" is not single bare-token. Semantically equivalent
                    (hvlv = obstacle-clearance distance).
  category=orient   Q="Question: horizontal or vertical grasp? Answer:"
                    0  -> "horizontal" (30629)
                    90 -> "vertical"   (15292)

Cache size: 1600 episodes × 8 frames × 224×224×3 bytes ≈ 1.93 GB uint8.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image

# Module-level registry — VQA clip caches keyed by split name ('train'/'val').
_VQA_CLIP_CACHE: Dict[str, np.ndarray] = {}


def register_vqa_clip_cache(name: str, cache: np.ndarray) -> None:
    _VQA_CLIP_CACHE[name] = cache


def get_vqa_clip_cache(name: str) -> Optional[np.ndarray]:
    return _VQA_CLIP_CACHE.get(name)


VQA_NUM_FRAMES = 8
VQA_CAMERA = "head_camera"
VQA_IMAGE_SIZE: Tuple[int, int] = (224, 224)


# ============================================================
# Per-category VQA config registry
# ============================================================

@dataclass(frozen=True)
class VQACategoryConfig:
    """Per-category VQA question + binary answer tokens.

    pref_keys: ordered tuple of the two pref_keys for this category (must
        match prompt.PREF_CATEGORIES[c].pref_keys). Used by QwenPI_VQA for
        the binary classifier slot order.
    answer_token_ids: pref_key -> first-token id (bare form) of the answer.
    answer_text: pref_key -> the literal string the assistant should emit.
    question: full text of the VQA question (Q + "? Answer:" suffix).
    """
    pref_keys: Tuple[str, str]
    answer_token_ids: Dict[str, int]
    answer_text: Dict[str, str]
    question: str


VQA_CATEGORIES: Dict[str, VQACategoryConfig] = {
    "giveobj": VQACategoryConfig(
        pref_keys=("25", "75"),
        answer_token_ids={"25": 10303, "75": 11892},
        answer_text={"25": "low", "75": "high"},
        question="Question: low or high contact at grasp? Answer:",
    ),
    # `contact` is the canonical alias for legacy `giveobj` (see
    # prompt.PREF_CATEGORIES for the rename note). Same question, same
    # answer tokens.
    "contact": VQACategoryConfig(
        pref_keys=("25", "75"),
        answer_token_ids={"25": 10303, "75": 11892},
        answer_text={"25": "low", "75": "high"},
        question="Question: low or high contact at grasp? Answer:",
    ),
    "height": VQACategoryConfig(
        pref_keys=("high", "low"),
        answer_token_ids={"high": 11892, "low": 10303},
        answer_text={"high": "high", "low": "low"},
        question="Question: high or low drop? Answer:",
    ),
    "hvlv": VQACategoryConfig(
        pref_keys=("hv", "lv"),
        answer_token_ids={"hv": 23559, "lv": 51659},   # far / near
        answer_text={"hv": "far", "lv": "near"},
        question="Question: far from or near the obstacle? Answer:",
    ),
    "orient": VQACategoryConfig(
        pref_keys=("0", "90"),
        answer_token_ids={"0": 30629, "90": 15292},   # horizontal / vertical
        answer_text={"0": "horizontal", "90": "vertical"},
        question="Question: horizontal or vertical grasp? Answer:",
    ),
    "place": VQACategoryConfig(
        pref_keys=("center", "corner"),
        answer_token_ids={"center": 3057, "corner": 73425},
        answer_text={"center": "center", "corner": "corner"},
        question="Question: center or corner placement? Answer:",
    ),
}


def get_vqa_category(category: str) -> VQACategoryConfig:
    if category not in VQA_CATEGORIES:
        raise KeyError(
            f"Unknown VQA category {category!r}; expected one of {tuple(VQA_CATEGORIES)}"
        )
    return VQA_CATEGORIES[category]


# ============================================================
# Legacy module-level exports (giveobj backward compat for any import
# that hasn't been updated to use VQA_CATEGORIES). QwenPI_VQA.py is the
# main consumer and now reads from VQA_CATEGORIES directly when given a
# category arg, but falling back to these on no category arg.
# ============================================================

VQA_QUESTION = VQA_CATEGORIES["giveobj"].question
VQA_ANSWER_TOKEN_IDS = VQA_CATEGORIES["giveobj"].answer_token_ids
VQA_ANSWER_TEXT      = VQA_CATEGORIES["giveobj"].answer_text


# ============================================================
# Clip cache builder
# ============================================================

def uniform_clip_indices(T: int, n: int = VQA_NUM_FRAMES) -> np.ndarray:
    """Uniform-linspace frame indices into a T-frame episode.

    This is what the training-time VQA clip cache uses. For inference, prefer
    `mid_clip_indices` or `gripper_anchored_clip_indices` — both validated to
    push contact taskB acc 0.61 -> 0.97-0.99 (see r-preference/doc/0523-stageA-analysis.md
    §4 for the strategy comparison).
    """
    return np.linspace(0, T - 1, n).round().astype(np.int64)


# Validated fractions: pushed contact taskB acc from 0.61 (uniform_8) to 0.99.
# Hand-tuned to be denser in the middle (diff 0.07/0.08/0.05/0.05/0.05/0.07/0.08)
# so more samples land near the typical grasp moment.
# DO NOT change without re-running the Stage A gate eval — the 0.99 number
# is bound to THIS sequence of fractions.
DEFAULT_MID_FRACS: Tuple[float, ...] = (0.25, 0.32, 0.40, 0.45, 0.50, 0.55, 0.62, 0.70)


def mid_clip_indices(
    T: int,
    n: int = VQA_NUM_FRAMES,
    fracs: Sequence[float] = DEFAULT_MID_FRACS,
) -> np.ndarray:
    """Frame indices at hand-tuned fractions clustered in mid-episode.

    Validated for contact taskB (put_boxdrink3_plate, T~193) where it gives
    0.99 acc on the 35k VQA ckpt vs 0.61 for uniform_8. Default fracs are
    the ones used in that validation; pass a different sequence at your
    own risk.
    """
    fracs_arr = np.asarray(fracs, dtype=np.float64)
    if len(fracs_arr) != n:
        raise ValueError(
            f"mid_clip_indices: len(fracs)={len(fracs_arr)} must match n={n}"
        )
    return (fracs_arr * (T - 1)).round().astype(np.int64)


# Validated gripper-close threshold + half-window: see analysis doc §4.
# half_window=4 → span = 8, total clip covers ~64 frames around grasp_t.
# Threshold 0.5 works for contact's gripper convention (1.0=open, 0.0=closed);
# verify per category before reusing.
DEFAULT_GRIPPER_CLOSE_THRESH: float = 0.5
DEFAULT_GRIPPER_HALF_WINDOW: int = 4


def find_grasp_frame(
    h5: h5py.File,
    threshold: float = DEFAULT_GRIPPER_CLOSE_THRESH,
) -> Optional[int]:
    """Find the first frame where either gripper crosses below `threshold`.

    Detection priority:
      1. First open→closed transition (gripper[t] < thr AND gripper[t-1] >= thr)
         on either left or right gripper.
      2. If no transition (gripper starts already closed), first frame
         where any gripper is below threshold.
      3. None if no closure ever happens (rare; caller should fall back).
    """
    L = np.asarray(h5["endpose/left_gripper"][:])
    R = np.asarray(h5["endpose/right_gripper"][:])
    T = len(L)
    for t in range(1, T):
        if (L[t] < threshold and L[t - 1] >= threshold) or \
           (R[t] < threshold and R[t - 1] >= threshold):
            return t
    where = np.where((L < threshold) | (R < threshold))[0]
    return int(where[0]) if len(where) > 0 else None


def gripper_anchored_clip_indices(
    h5: h5py.File,
    n: int = VQA_NUM_FRAMES,
    half_window: int = DEFAULT_GRIPPER_HALF_WINDOW,
    threshold: float = DEFAULT_GRIPPER_CLOSE_THRESH,
) -> Tuple[np.ndarray, Optional[int]]:
    """Frame indices centered on the detected grasp moment.

    Returns (indices, grasp_t). If no grasp moment detected, returns
    uniform indices + grasp_t=None (caller can see this and decide what
    to do — current pipeline accepts the fallback).

    Validated on contact taskB: 0.98 acc on 35k VQA, signal +45.16.
    """
    L_len = len(h5["endpose/left_gripper"])
    T = L_len
    grasp_t = find_grasp_frame(h5, threshold=threshold)
    if grasp_t is None:
        return uniform_clip_indices(T, n), None
    span = max(half_window * 2, 1)
    t0 = max(0, grasp_t - span * n // 2)
    t1 = min(T - 1, grasp_t + span * n // 2)
    if t0 == 0:
        t1 = min(T - 1, t0 + span * (n - 1))
    elif t1 == T - 1:
        t0 = max(0, t1 - span * (n - 1))
    return np.linspace(t0, t1, n).round().astype(np.int64), grasp_t


# ============================================================
# Unified clip-loading dispatcher (used by Stage A eval + Stage B labeler)
# ============================================================

CLIP_STRATEGY_NAMES = ("uniform_8", "mid_8", "gripper_anchored")


def load_clip_by_strategy(
    h5_path: Path,
    strategy: str = "mid_8",
    n: int = VQA_NUM_FRAMES,
    camera: str = VQA_CAMERA,
    image_size: Tuple[int, int] = VQA_IMAGE_SIZE,
) -> Tuple[List["Image.Image"], Dict]:
    """Decode an n-frame clip from an HDF5 episode using the given strategy.

    Returns (frames, info) where info has keys:
      - "strategy", "indices" (list[int]), "T" (total frames),
      - "grasp_t" (int|None, only for gripper_anchored).

    Pipeline contract: Stage A eval and Stage B labeler MUST both call into
    this function to guarantee the strategy used at pseudo-label time
    matches the one validated at gate time.
    """
    H, W = image_size
    with h5py.File(h5_path, "r") as h5:
        T = h5[f"observation/{camera}/rgb"].shape[0]
        grasp_t = None
        if strategy == "uniform_8":
            idx = uniform_clip_indices(T, n)
        elif strategy == "mid_8":
            idx = mid_clip_indices(T, n)
        elif strategy == "gripper_anchored":
            idx, grasp_t = gripper_anchored_clip_indices(h5, n)
        else:
            raise ValueError(
                f"Unknown clip strategy {strategy!r}; expected one of {CLIP_STRATEGY_NAMES}"
            )

        frames: List[Image.Image] = []
        for t in idx:
            raw = h5[f"observation/{camera}/rgb"][int(t)]
            data = raw.tobytes() if hasattr(raw, "tobytes") else raw
            img = Image.open(io.BytesIO(data)).convert("RGB")
            if img.size != (W, H):
                img = img.resize((W, H))
            frames.append(img)

    return frames, {
        "strategy": strategy,
        "indices": idx.tolist(),
        "T": int(T),
        "grasp_t": int(grasp_t) if grasp_t is not None else None,
    }


def _decode_clip(
    h5: h5py.File,
    camera: str,
    indices: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:
    H, W = image_size
    out = np.empty((len(indices), H, W, 3), dtype=np.uint8)
    for i, t in enumerate(indices):
        raw = h5[f"observation/{camera}/rgb"][int(t)]
        data = raw.tobytes() if hasattr(raw, "tobytes") else raw
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if img.size != (W, H):
            img = img.resize((W, H))
        out[i] = np.asarray(img, dtype=np.uint8)
    return out


def build_vqa_clip_cache(
    data_root: Path,
    episode_keys: Sequence[Tuple[str, int]],
    num_frames: int = VQA_NUM_FRAMES,
    camera: str = VQA_CAMERA,
    image_size: Tuple[int, int] = VQA_IMAGE_SIZE,
    verbose: bool = True,
) -> np.ndarray:
    """
    Pre-decode head_camera clips for every (task_dir, ep_id) into a single
    contiguous uint8 buffer of shape (N_ep, num_frames, H, W, 3).

    Children spawned by DataLoader share its pages via fork-COW as long as
    no writes happen.
    """
    N = len(episode_keys)
    H, W = image_size
    cache = np.empty((N, num_frames, H, W, 3), dtype=np.uint8)
    t0 = time.time()
    for i, (task_dir, ep_id) in enumerate(episode_keys):
        h5_path = Path(data_root) / task_dir / "data" / f"episode{ep_id}.hdf5"
        with h5py.File(h5_path, "r") as h5:
            T = h5[f"observation/{camera}/rgb"].shape[0]
            idx = uniform_clip_indices(T, num_frames)
            cache[i] = _decode_clip(h5, camera, idx, image_size)
        if verbose and (i + 1) % 200 == 0:
            dt = time.time() - t0
            eta = dt / (i + 1) * (N - i - 1)
            print(f"  vqa_cache: {i+1}/{N} eps ({dt:.1f}s, ETA {eta:.1f}s)")
    if verbose:
        gb = cache.nbytes / 1024 ** 3
        print(f"  vqa_cache built: shape={cache.shape}, {gb:.2f} GB uint8")
    return cache


def cache_row_to_pil(cache_row: np.ndarray) -> List[Image.Image]:
    """(num_frames, H, W, 3) uint8 view -> list of PIL.Image (PIL copies into its own buffer)."""
    return [Image.fromarray(cache_row[i]) for i in range(cache_row.shape[0])]


def _sanity():
    """Smoke probe used by tests/test_smoke_vqa.py and CLI."""
    # Tokenizer sanity: every registered answer is single bare-token.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        for cat_name, cat in VQA_CATEGORIES.items():
            for pk, text in cat.answer_text.items():
                ids = tok.encode(text, add_special_tokens=False)
                # We accept first-token CE if the text doesn't tokenize as a single
                # token (current case: "narrow" → ["n","arrow"]). The registered
                # ID is the FIRST token id; verify it matches.
                assert ids[0] == cat.answer_token_ids[pk], (
                    f"{cat_name}/{pk}: expected first-token id "
                    f"{cat.answer_token_ids[pk]} for {text!r}, got {ids[0]}"
                )
        print(" tokenizer sanity OK across all 4 categories")
    except ImportError:
        print(" tokenizer sanity SKIPPED (transformers not available)")

    # Clip cache shape probe on giveobj (small fixture).
    root = Path("/mnt/localssd/kaiwenh/pref/data/giveobj")
    if root.exists():
        keys = [("give_boxdrink_25", 0), ("give_boxdrink_75", 0)]
        cache = build_vqa_clip_cache(root, keys, verbose=False)
        assert cache.shape == (2, 8, 224, 224, 3), cache.shape
        assert cache.dtype == np.uint8
        assert cache.flags["C_CONTIGUOUS"]
        pil_list = cache_row_to_pil(cache[0])
        assert len(pil_list) == 8 and pil_list[0].size == (224, 224)
        print(" clip cache sanity OK (giveobj)")
    print("vqa_sample.py sanity OK")


if __name__ == "__main__":
    _sanity()
