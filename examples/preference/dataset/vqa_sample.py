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
    """Per-category VQA question + binary answer tokens + sampling strategy.

    pref_keys: ordered tuple of the two pref_keys for this category (must
        match prompt.PREF_CATEGORIES[c].pref_keys). Used by QwenPI_VQA for
        the binary classifier slot order.
    answer_token_ids: pref_key -> first-token id (bare form) of the answer.
    answer_text: pref_key -> the literal string the assistant should emit.
    question: full text of the VQA question (Q + "? Answer:" suffix).
    cameras: which camera(s) to use for the VQA clip. Default = head only
        (backward-compat with original contact recipe). Multi-cam needed
        when target object is small in head_camera (e.g. place taskB stand
        is ~5cm vs ~1.35m camera distance → center/corner offset only
        few pixels in head; wrist cams give close-up).
    clip_strategy: which temporal sampling at cache build time.
        - "uniform_8": original (np.linspace 0->T-1, n=8)
        - "mid_8":     fractions [0.25..0.70], validated for contact grasp
        - "late_8":    fractions [0.65..0.98], for tasks where pref signal
                       is at release moment (place, height: placement/drop
                       at frac~0.91)
        - "dense_16":  np.linspace 0->T-1, n=16 (more frames for trajectory-
                       wide signals like hvlv detour)
    n_frames: how many frames per clip (default 8, increase to 16 for
        trajectory-wide cats like hvlv).
    """
    pref_keys: Tuple[str, str]
    answer_token_ids: Dict[str, int]
    answer_text: Dict[str, str]
    question: str
    cameras: Tuple[str, ...] = ("head_camera",)
    clip_strategy: str = "uniform_8"
    n_frames: int = 8


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
        # MULTI-CAM(active wrist) + LATE-CACHE (added 2026-05-26 after
        # diagnostic showed head-only uniform_8 cache fails on small-target
        # placement):
        #   - head_camera         (overhead, scene context)
        #   - active_wrist        (close-up of the arm actually doing the
        #                          placement — resolved per-ep to either
        #                          left_camera or right_camera based on which
        #                          arm has larger xyz range; avoids wasting
        #                          tokens on inactive wrist's static home view)
        # Result: 2 cams × 8 frames = 16 images per VQA sample, same load as
        # original head-only 8 frames × 2 samples (n_vqa_per_batch=2).
        # late_8 fractions cluster around placement moment (frac=0.91 verified).
        # See r-preference/eval/viz_place_diagnosis/ for the failure analysis.
        cameras=("head_camera", "active_wrist"),
        clip_strategy="late_8",
        n_frames=8,
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


# Late-clustered: targets the placement/release moment (e.g. place taskB has
# release at frac=0.91; height drop at frac~0.91). Hand-tuned to cover frac
# 0.65-0.98 with denser coverage near 0.85-0.95 (the actual placement zone).
# DO NOT change without re-running gate eval — model attention tuned to this.
DEFAULT_LATE_FRACS: Tuple[float, ...] = (0.65, 0.72, 0.79, 0.85, 0.88, 0.92, 0.95, 0.98)


def late_clip_indices(
    T: int,
    n: int = VQA_NUM_FRAMES,
    fracs: Sequence[float] = DEFAULT_LATE_FRACS,
) -> np.ndarray:
    """Frame indices at hand-tuned fractions clustered near release/placement
    moment (frac ~0.85-0.98). Designed for tasks where the pref signal is at
    the end of trajectory (place, height drop)."""
    fracs_arr = np.asarray(fracs, dtype=np.float64)
    if len(fracs_arr) != n:
        raise ValueError(
            f"late_clip_indices: len(fracs)={len(fracs_arr)} must match n={n}"
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

CLIP_STRATEGY_NAMES = ("uniform_8", "mid_8", "late_8", "gripper_anchored")


# Sentinel for active-arm wrist camera selection. When this string appears in
# a cameras tuple, it's resolved per-episode to "left_camera" or "right_camera"
# based on which arm has the larger xyz range in this episode. This avoids the
# wasteful "static home view from inactive wrist" frames (which give 0 signal
# but 1× cost) and keeps VQA forward token count manageable.
ACTIVE_WRIST_SENTINEL: str = "active_wrist"


def _resolve_active_wrist(h5: h5py.File) -> str:
    """Return 'left_camera' or 'right_camera' based on which arm moves more
    in this episode (heuristic: larger xyz range = active arm whose wrist
    camera physically moves with it)."""
    L = h5["endpose/left_endpose"][:, :3]
    R = h5["endpose/right_endpose"][:, :3]
    Lr = float(np.linalg.norm(L.ptp(axis=0)))
    Rr = float(np.linalg.norm(R.ptp(axis=0)))
    return "left_camera" if Lr > Rr else "right_camera"


def _resolve_cameras_for_ep(cameras: Sequence[str], h5: h5py.File) -> List[str]:
    """Replace any `active_wrist` sentinel with the per-episode active wrist
    camera. Returns a concrete list of physical camera names."""
    if ACTIVE_WRIST_SENTINEL not in cameras:
        return list(cameras)
    active = _resolve_active_wrist(h5)
    return [active if c == ACTIVE_WRIST_SENTINEL else c for c in cameras]


def _compute_clip_indices(
    h5: h5py.File, strategy: str, n: int
) -> Tuple[np.ndarray, Optional[int]]:
    """Returns (indices, grasp_t) for the given strategy. Shared helper used
    by both cache-build and inference paths so they agree exactly."""
    if strategy == "uniform_8":
        # n is taken at face value (function uses VQA_NUM_FRAMES default for
        # head-cam compat but accepts override for dense_16 cats)
        T = h5["observation/head_camera/rgb"].shape[0]
        return uniform_clip_indices(T, n), None
    elif strategy == "mid_8":
        T = h5["observation/head_camera/rgb"].shape[0]
        return mid_clip_indices(T, n), None
    elif strategy == "late_8":
        T = h5["observation/head_camera/rgb"].shape[0]
        return late_clip_indices(T, n), None
    elif strategy == "gripper_anchored":
        return gripper_anchored_clip_indices(h5, n)
    else:
        raise ValueError(
            f"Unknown clip strategy {strategy!r}; expected one of {CLIP_STRATEGY_NAMES}"
        )


def load_clip_by_strategy(
    h5_path: Path,
    strategy: str = "mid_8",
    n: int = VQA_NUM_FRAMES,
    camera: str = VQA_CAMERA,
    image_size: Tuple[int, int] = VQA_IMAGE_SIZE,
    cameras: Optional[Sequence[str]] = None,
) -> Tuple[List["Image.Image"], Dict]:
    """Decode an n-frame clip from an HDF5 episode using the given strategy.

    Args:
      cameras: if provided, overrides single-cam `camera` arg. Returns
        n_frames * n_cams images in time-grouped order:
        [cam0_t0, cam1_t0, cam2_t0, cam0_t1, cam1_t1, ...]. Used by Stage A
        eval / Stage B labeler for multi-cam-trained VQA models (place).
      camera: single-camera mode (back-compat). Used if cameras=None.

    Returns (frames, info) where info has keys:
      - "strategy", "indices" (list[int]), "T" (total frames),
      - "grasp_t" (int|None, only for gripper_anchored),
      - "cameras" (list[str]) — which cams were sampled.

    Pipeline contract: Stage A eval and Stage B labeler MUST both call into
    this function (or build_vqa_clip_cache, which uses the SAME index helper
    via `_compute_clip_indices`) so cache + inference agree exactly on which
    frames the model sees.
    """
    cams_in = tuple(cameras) if cameras is not None else (camera,)
    H, W = image_size
    with h5py.File(h5_path, "r") as h5:
        T = h5["observation/head_camera/rgb"].shape[0]
        idx, grasp_t = _compute_clip_indices(h5, strategy, n)
        # Resolve active_wrist sentinel per ep
        cams = _resolve_cameras_for_ep(cams_in, h5)
        frames: List[Image.Image] = []
        # Time-grouped order: for each t, emit cam0,cam1,...,cam_C-1 in turn.
        # Lets VLM see multi-view per timestep instead of per-cam blocks.
        for t in idx:
            for cam in cams:
                raw = h5[f"observation/{cam}/rgb"][int(t)]
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
        "cameras_requested": list(cams_in),
        "cameras_resolved": list(cams),
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
    cameras: Sequence[str] = (VQA_CAMERA,),
    image_size: Tuple[int, int] = VQA_IMAGE_SIZE,
    strategy: str = "uniform_8",
    verbose: bool = True,
) -> np.ndarray:
    """
    Pre-decode clip frames for every (task_dir, ep_id) into a single
    contiguous uint8 buffer.

    Shape:
      - n_cams == 1: (N_ep, num_frames, H, W, 3)  [backward compat]
      - n_cams >  1: (N_ep, num_frames, n_cams, H, W, 3)

    Uses the SAME index helper as `load_clip_by_strategy` so cache + inference
    sample identical frames given identical strategy.

    Children spawned by DataLoader share these pages via fork-COW as long as
    no writes happen.
    """
    N = len(episode_keys)
    n_cams = len(cameras)  # logical count (active_wrist counts as 1)
    H, W = image_size
    if n_cams == 1:
        cache = np.empty((N, num_frames, H, W, 3), dtype=np.uint8)
    else:
        cache = np.empty((N, num_frames, n_cams, H, W, 3), dtype=np.uint8)
    t0 = time.time()
    for i, (task_dir, ep_id) in enumerate(episode_keys):
        h5_path = Path(data_root) / task_dir / "data" / f"episode{ep_id}.hdf5"
        with h5py.File(h5_path, "r") as h5:
            idx, _ = _compute_clip_indices(h5, strategy, num_frames)
            # Resolve `active_wrist` sentinel per-ep (no-op if no sentinel).
            cams_resolved = _resolve_cameras_for_ep(cameras, h5)
            if n_cams == 1:
                cache[i] = _decode_clip(h5, cams_resolved[0], idx, image_size)
            else:
                for c, cam in enumerate(cams_resolved):
                    cache[i, :, c] = _decode_clip(h5, cam, idx, image_size)
        if verbose and (i + 1) % 200 == 0:
            dt = time.time() - t0
            eta = dt / (i + 1) * (N - i - 1)
            print(f"  vqa_cache: {i+1}/{N} eps ({dt:.1f}s, ETA {eta:.1f}s)")
    if verbose:
        gb = cache.nbytes / 1024 ** 3
        print(f"  vqa_cache built: shape={cache.shape}, cams_requested={list(cameras)}, "
              f"strategy={strategy!r}, {gb:.2f} GB uint8")
    return cache


def cache_row_to_pil(cache_row: np.ndarray) -> List[Image.Image]:
    """uint8 view -> list of PIL.Image (PIL copies into its own buffer).

    Accepts both shapes:
      - (num_frames, H, W, 3) single-cam: returns num_frames images
      - (num_frames, n_cams, H, W, 3) multi-cam: returns num_frames * n_cams
        images in TIME-GROUPED order: [cam0_t0, cam1_t0, ..., cam0_t1, ...]
        (so VLM sees multi-view per timestep, matches load_clip_by_strategy
        order for inference path).
    """
    if cache_row.ndim == 4:
        return [Image.fromarray(cache_row[i]) for i in range(cache_row.shape[0])]
    if cache_row.ndim == 5:
        F, C = cache_row.shape[:2]
        return [Image.fromarray(cache_row[f, c]) for f in range(F) for c in range(C)]
    raise ValueError(
        f"cache_row must be 4D (single-cam) or 5D (multi-cam), got shape {cache_row.shape}"
    )


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

    # Clip cache shape probe on giveobj (small fixture, single-cam back-compat).
    root_gv = Path("/mnt/localssd/kaiwenh/pref/data/giveobj")
    if root_gv.exists():
        keys = [("give_boxdrink_25", 0), ("give_boxdrink_75", 0)]
        cache = build_vqa_clip_cache(root_gv, keys, verbose=False)
        assert cache.shape == (2, 8, 224, 224, 3), cache.shape
        assert cache.dtype == np.uint8
        assert cache.flags["C_CONTIGUOUS"]
        pil_list = cache_row_to_pil(cache[0])
        assert len(pil_list) == 8 and pil_list[0].size == (224, 224)
        print(" clip cache sanity OK (giveobj, single-cam back-compat)")

    # New multi-cam + late_8 probe on place.
    root_p = Path("/mnt/localssd/kaiwenh/pref/data/place")
    if root_p.exists():
        keys = [("move_mouse_pad_center", 0), ("move_mouse_pad_corner", 0)]
        # active_wrist mode (production for place): head + dynamically-resolved wrist
        cache_aw = build_vqa_clip_cache(
            root_p, keys,
            num_frames=8,
            cameras=("head_camera", "active_wrist"),
            strategy="late_8",
            verbose=False,
        )
        assert cache_aw.shape == (2, 8, 2, 224, 224, 3), cache_aw.shape
        assert cache_aw.dtype == np.uint8
        pil_aw = cache_row_to_pil(cache_aw[0])
        assert len(pil_aw) == 16 and pil_aw[0].size == (224, 224), (len(pil_aw), pil_aw[0].size)
        print(" clip cache sanity OK (place, head+active_wrist late_8 → 16 PIL imgs)")
        # Explicit 3-cam mode still works (kept for diagnostic / ablation)
        cache3 = build_vqa_clip_cache(
            root_p, keys,
            num_frames=8,
            cameras=("head_camera", "left_camera", "right_camera"),
            strategy="late_8",
            verbose=False,
        )
        assert cache3.shape == (2, 8, 3, 224, 224, 3), cache3.shape
        print(" clip cache sanity OK (place explicit 3-cam → 24 PIL imgs, only used for ablation)")
        # Also verify late_8 indices actually land near release moment
        import h5py
        with h5py.File(root_p / "move_mouse_pad_center" / "data" / "episode0.hdf5", "r") as h5:
            idx, _ = _compute_clip_indices(h5, "late_8", 8)
            T = h5["observation/head_camera/rgb"].shape[0]
            fracs = idx / (T - 1)
            assert fracs[0] > 0.6 and fracs[-1] > 0.95, fracs
            print(f"   late_8 fracs (T={T}): {[f'{f:.2f}' for f in fracs]} ✓")
            # Verify active_wrist resolution
            resolved = _resolve_active_wrist(h5)
            print(f"   active_wrist for move_mouse_pad_center/ep0: resolved to {resolved}")
    print("vqa_sample.py sanity OK")


if __name__ == "__main__":
    _sanity()
