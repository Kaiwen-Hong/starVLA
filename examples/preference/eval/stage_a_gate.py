"""
Stage A gate eval — preference-conditioned VLA validation suite.

Given a (baseline, VQA) ckpt pair for one category, runs:

  (1) VQA per-task accuracy on TASKB  (gate-main metric — VQA generalization
      to unseen objects/tasks that Stage B will pseudo-label).
  (2) VQA per-task accuracy on TASKA val  (sanity — VQA learned its training
      distribution).
  (3) Counterfactual action MSE on TASKA val  (per ckpt). Each (img, state)
      run twice with prompt suffix "Preference: <A>" vs "<B>"; MSE
      between the two normalized action chunks. Action head should respond
      to the prompt token even in baseline.
  (4) Sign accuracy on the 3 contact "strong tasks" (put_boxdrink_dustbin /
      put_callbell_dustbin / put_fork_dustbin — only meaningful in contact,
      from doc 0522 §2.6).
  (5) Action drift baseline-vs-VQA (sanity — VQA didn't break action quality).

By design from the user: (3) is expected to be close baseline≈VQA (both
action heads read pref from the prompt token); the value of VQA is in (1).

Loads ckpts sequentially (model is ~16 GB; only one fits in single-GPU
inference at a time). Per-sample (a_low, a_high) tensors from baseline are
cached to disk between passes so the comparison at (5) can run against the
VQA cache without holding both models in memory.

CLI:
    python -m examples.preference.eval.stage_a_gate \
        --category contact \
        --baseline_yaml examples/preference/train_files/starvla_pref_stage_a_baseline_contact.yaml \
        --vqa_yaml      examples/preference/train_files/starvla_pref_stage_a_vqa_contact.yaml \
        --baseline_ckpt results/Checkpoints/pref_baseline_stage_a_v1_noVQA_contact/checkpoints/steps_35000_pytorch_model.pt \
        --vqa_ckpt      results/Checkpoints/pref_main_stage_a_v1_VQA_contact/checkpoints/steps_35000_pytorch_model.pt \
        --taskA_n_eps_per_task 5 \
        --taskB_data_root /mnt/localssd/kaiwenh/pref/data/contact/taskB \
        --out r-preference/eval/stage_a_gate_contact_35k.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from examples.preference.dataset.prompt import PREF_CATEGORIES, build_action_prompt
from examples.preference.dataset.pref_hdf5_dataset import PrefHDF5Dataset
from examples.preference.dataset.vqa_sample import (
    VQA_CATEGORIES,
    uniform_clip_indices,
)


# Strong tasks for contact only (doc 0522 §2.6): (task_group, axis_idx_in_20D, expected_sign_75_minus_25).
# Standing objects → +L_z; tabletop fork → +L_x. Sign computed on first-frame
# delta of normalized actions.
STRONG_TASKS_CONTACT = {
    "put_boxdrink_dustbin":  (2,  +1),   # L_z, "high contact (75)" → higher z
    "put_callbell_dustbin":  (2,  +1),   # L_z
    "put_fork_dustbin":      (0,  +1),   # L_x
}


@dataclass
class FrameSample:
    """One (task_dir, episode, frame) sample for counterfactual eval."""
    task_dir: str
    task_group: str
    pref_key: str
    ep_id: int
    frame_idx: int


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_clip_from_h5(h5_path: Path, n_frames: int = 8, camera: str = "head_camera",
                       image_size: Tuple[int, int] = (224, 224),
                       cameras: Optional[Sequence[str]] = None,
                       strategy: str = "uniform_8") -> List[Image.Image]:
    """Read clip from one episode.

    Back-compat default: 8-frame uniform head_camera (= original contact eval).
    Multi-cam path: pass `cameras=("head_camera", "active_wrist")` etc; resolves
    per-ep + uses same time-grouped PIL ordering as training cache (per
    `vqa_sample.cache_row_to_pil`), so inference matches train.
    """
    from examples.preference.dataset.vqa_sample import load_clip_by_strategy
    frames, _info = load_clip_by_strategy(
        h5_path,
        strategy=strategy,
        n=n_frames,
        camera=camera,
        image_size=image_size,
        cameras=cameras,
    )
    return frames


def list_taskB_episodes(taskB_root: Path, pref_keys: Tuple[str, str]) -> List[FrameSample]:
    """
    Enumerate all taskB episodes for contact.
    Convention: <taskB_root>/<task_group>_<pref_key>/data/episode*.hdf5
    """
    out = []
    if not taskB_root.is_dir():
        raise FileNotFoundError(f"taskB root not found: {taskB_root}")
    for sub in sorted(taskB_root.iterdir()):
        if not sub.is_dir() or not (sub / "data").is_dir():
            continue
        name = sub.name
        # find which pref_key matches the suffix
        pk = None
        tg = None
        for cand_pk in pref_keys:
            suffix = f"_{cand_pk}"
            if name.endswith(suffix):
                pk = cand_pk
                tg = name[: -len(suffix)]
                break
        if pk is None:
            print(f"  [skip] taskB dir {name!r}: no matching pref_key in {pref_keys}")
            continue
        for h5p in sorted((sub / "data").glob("episode*.hdf5")):
            ep_id = int(h5p.stem.replace("episode", ""))
            out.append(FrameSample(task_dir=name, task_group=tg, pref_key=pk,
                                   ep_id=ep_id, frame_idx=-1))  # frame N/A for VQA-only
    return out


def subsample_taskA_episodes(ds: PrefHDF5Dataset, n_eps_per_task: int,
                              rng: random.Random) -> List[FrameSample]:
    """Pick first n_eps_per_task episodes per (task_dir) deterministically.
    For VQA we collapse on episode; for counterfactual we'll add a random
    frame to each separately. Here we return one FrameSample per (task_dir,
    ep_id) with frame_idx left to be picked later.
    """
    by_task: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
    seen_eps = set()
    for task_dir, tg, pk, ep_id, _t in ds._index:
        key = (task_dir, ep_id)
        if key in seen_eps:
            continue
        seen_eps.add(key)
        by_task[task_dir].append((tg, pk, ep_id))

    picked: List[FrameSample] = []
    for task_dir, eps in by_task.items():
        rng.shuffle(eps)
        for tg, pk, ep_id in eps[:n_eps_per_task]:
            picked.append(FrameSample(task_dir=task_dir, task_group=tg,
                                      pref_key=pk, ep_id=ep_id, frame_idx=-1))
    return picked


def pick_random_frame_for_sample(ds: PrefHDF5Dataset, fs: FrameSample,
                                  rng: random.Random) -> int:
    """Pick a random valid frame_idx for the (task_dir, ep_id) — must allow
    a full chunk_size lookahead (consistent with training-time __getitem__)."""
    h5p = ds.data_root / fs.task_dir / "data" / f"episode{fs.ep_id}.hdf5"
    with h5py.File(h5p, "r") as h5:
        T = h5["endpose/left_endpose"].shape[0]
    last = T - ds.chunk_size
    if last < 0:
        return 0
    return rng.randint(0, last)


def load_sample_for_action(ds: PrefHDF5Dataset, fs: FrameSample) -> dict:
    """Build a single sample dict (image, lang_low/high, state) for counterfactual eval.

    Action loss isn't needed; we only call predict_action twice per (image,
    state), so we return a dict with placeholder lang — caller substitutes
    the two pref prompts.
    """
    # The dataset internally builds full sample. Use its __getitem__ then
    # discard `action`. But __getitem__ uses idx into _index where frame_idx
    # is the training one. Reuse its image loader directly.
    h5p = ds.data_root / fs.task_dir / "data" / f"episode{fs.ep_id}.hdf5"
    with h5py.File(h5p, "r") as h5:
        images = ds._load_images(h5, fs.frame_idx)
        if ds.include_state:
            state_raw = ds._build_ee_20d(
                h5["endpose/left_endpose"][fs.frame_idx:fs.frame_idx + 1],
                h5["endpose/left_gripper"][fs.frame_idx:fs.frame_idx + 1],
                h5["endpose/right_endpose"][fs.frame_idx:fs.frame_idx + 1],
                h5["endpose/right_gripper"][fs.frame_idx:fs.frame_idx + 1],
            )
            if ds.state_q01 is not None:
                state = ds._normalize(state_raw, ds.state_q01, ds.state_q99)
            else:
                state = state_raw
            state = state.astype(np.float16)
        else:
            state = None
    return {"image": images, "state": state}


# ---------------------------------------------------------------------------
# Model load / instantiate
# ---------------------------------------------------------------------------

def build_framework(yaml_path: str, framework_name: str):
    """Instantiate framework from YAML + framework registry. Skips dataloader."""
    cfg = OmegaConf.load(yaml_path)
    if cfg.framework.name != framework_name:
        raise ValueError(
            f"yaml framework.name={cfg.framework.name!r} != expected {framework_name!r}"
        )
    # Import here so registration side-effects happen.
    from starVLA.model.framework.QwenPI import Qwen_PI  # noqa: F401
    if framework_name == "QwenPI_VQA":
        from starVLA.model.framework.QwenPI_VQA import Qwen_PI_VQA  # noqa: F401
    from starVLA.model.tools import FRAMEWORK_REGISTRY
    cls = FRAMEWORK_REGISTRY[framework_name]
    model = cls(config=cfg)
    return model, cfg


def load_ckpt_into_model(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load a flat state_dict ckpt; strict=False to tolerate Zero2 optimizer
    keys (they'll be silently dropped if present)."""
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "module" in sd and isinstance(sd["module"], dict):
        sd = sd["module"]
    # Strip "module." prefix if any (DDP-wrapped saves)
    sd = {k[len("module."):]: v for k, v in sd.items()} if all(
        k.startswith("module.") for k in sd
    ) else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded ckpt: missing={len(missing)} keys, unexpected={len(unexpected)} keys")
    if missing[:3]:
        print(f"    missing sample: {missing[:3]}")
    if unexpected[:3]:
        print(f"    unexpected sample: {unexpected[:3]}")


# ---------------------------------------------------------------------------
# Eval passes
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_vqa_acc(model, episode_samples: List[FrameSample], category: str,
                taskB_root: Optional[Path], val_ds: Optional[PrefHDF5Dataset],
                tag: str) -> Dict:
    """For each episode, build 8-frame head_camera clip and call predict_preference.
    Returns {per_task: {tg: {acc, n}}, overall: {acc, n}, confidences: [...]}.

    `episode_samples` come from either taskB enumeration or taskA subsample.
    `val_ds` is required when source = taskA (we need ds.data_root).
    `taskB_root` required when source = taskB.
    """
    cat_cfg = VQA_CATEGORIES[category]
    print(f"\n  === VQA acc [{tag}], {len(episode_samples)} episodes ===")
    t0 = time.time()
    per_task: Dict[str, List[int]] = defaultdict(list)
    confidences: List[float] = []
    pred_dist: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))  # tg -> pk -> count

    for i, fs in enumerate(episode_samples):
        if val_ds is not None:
            h5p = val_ds.data_root / fs.task_dir / "data" / f"episode{fs.ep_id}.hdf5"
        else:
            h5p = taskB_root / fs.task_dir / "data" / f"episode{fs.ep_id}.hdf5"
        # Use per-cat cameras + strategy from registry so multi-cam cats (e.g. place
        # with head+active_wrist+late_8) get evaluated with the SAME clip layout
        # the VQA model was trained on. Single-cam cats (contact etc) fall back to
        # head_camera + uniform_8 = original behavior.
        clip = load_clip_from_h5(
            h5p,
            n_frames=cat_cfg.n_frames,
            cameras=cat_cfg.cameras,
            strategy=cat_cfg.clip_strategy,
        )
        out = model.predict_preference(clip)
        correct = int(out["pref_key"] == fs.pref_key)
        per_task[fs.task_group].append(correct)
        confidences.append(out["confidence"])
        pred_dist[fs.task_group][out["pref_key"]] += 1
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(episode_samples)} ({time.time()-t0:.1f}s)")

    overall_n = sum(len(v) for v in per_task.values())
    overall_acc = sum(sum(v) for v in per_task.values()) / max(overall_n, 1)
    out = {
        "overall_acc": overall_acc,
        "overall_n": overall_n,
        "per_task_acc": {tg: float(np.mean(v)) for tg, v in per_task.items()},
        "per_task_n":   {tg: len(v) for tg, v in per_task.items()},
        "confidence_mean": float(np.mean(confidences)) if confidences else 0.0,
        "confidence_p10":  float(np.percentile(confidences, 10)) if confidences else 0.0,
        "pred_dist": {tg: dict(d) for tg, d in pred_dist.items()},
    }
    print(f"  overall_acc={overall_acc:.3f} (N={overall_n}) "
          f"per_task={out['per_task_acc']}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    return out


@torch.inference_mode()
def run_counterfactual_action(model, frame_samples: List[FrameSample],
                               val_ds: PrefHDF5Dataset, category: str,
                               tag: str) -> Dict:
    """For each FrameSample with valid frame_idx, run predict_action twice
    (pref=A and pref=B). Returns per_task MSE + per-sample (a_A, a_B) cache
    (for cross-ckpt comparison later)."""
    cat_cfg = PREF_CATEGORIES[category]
    pk_A, pk_B = cat_cfg.pref_keys[0], cat_cfg.pref_keys[1]
    print(f"\n  === Counterfactual action [{tag}], {len(frame_samples)} frames ===")
    t0 = time.time()
    per_task_mse: Dict[str, List[float]] = defaultdict(list)
    per_task_sign_pos: Dict[str, List[int]] = defaultdict(list)
    sample_cache: List[dict] = []

    for i, fs in enumerate(frame_samples):
        s = load_sample_for_action(val_ds, fs)
        lang_A = build_action_prompt(fs.task_group, pk_A, paraphrase=None, category=category)
        lang_B = build_action_prompt(fs.task_group, pk_B, paraphrase=None, category=category)
        ex_A = {"image": s["image"], "lang": lang_A, "state": s["state"]}
        ex_B = {"image": s["image"], "lang": lang_B, "state": s["state"]}

        out_A = model.predict_action([ex_A])["normalized_actions"][0]  # (T, D)
        out_B = model.predict_action([ex_B])["normalized_actions"][0]

        diff = out_B - out_A
        # Normalize per-element so MSE is comparable across categories.
        mse = float(np.sqrt(np.mean(diff ** 2)))
        per_task_mse[fs.task_group].append(mse)

        # Sign acc for contact strong tasks
        if category == "contact" and fs.task_group in STRONG_TASKS_CONTACT:
            axis_idx, expected_sign = STRONG_TASKS_CONTACT[fs.task_group]
            # First-frame delta in normalized space
            delta_first = float(out_B[0, axis_idx] - out_A[0, axis_idx])
            sign_correct = int(np.sign(delta_first) == expected_sign)
            per_task_sign_pos[fs.task_group].append(sign_correct)

        sample_cache.append({
            "task_dir": fs.task_dir, "task_group": fs.task_group,
            "pref_key": fs.pref_key, "ep_id": fs.ep_id, "frame_idx": fs.frame_idx,
            "a_A": out_A.tolist(), "a_B": out_B.tolist(),
        })

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(frame_samples)} ({time.time()-t0:.1f}s)")

    overall_mse = float(np.mean([m for ml in per_task_mse.values() for m in ml]))
    out = {
        "overall_mse_normalized": overall_mse,
        "per_task_mse":  {tg: float(np.mean(v)) for tg, v in per_task_mse.items()},
        "per_task_n":    {tg: len(v) for tg, v in per_task_mse.items()},
        "sign_accuracy_strong": {tg: float(np.mean(v)) for tg, v in per_task_sign_pos.items()},
        "sign_accuracy_n":      {tg: len(v) for tg, v in per_task_sign_pos.items()},
    }
    print(f"  overall counterfactual MSE={overall_mse:.4f}")
    print(f"  per_task_mse={out['per_task_mse']}")
    if out["sign_accuracy_strong"]:
        print(f"  sign_accuracy_strong={out['sign_accuracy_strong']}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    return out, sample_cache


def compute_drift_between_ckpts(baseline_cache: List[dict], vqa_cache: List[dict]) -> Dict:
    """Compute baseline-vs-VQA action drift per sample, same (task_dir, ep_id, frame_idx)."""
    # index by (task_dir, ep_id, frame_idx)
    bd = {(c["task_dir"], c["ep_id"], c["frame_idx"]): c for c in baseline_cache}
    vd = {(c["task_dir"], c["ep_id"], c["frame_idx"]): c for c in vqa_cache}
    keys = sorted(set(bd) & set(vd))
    per_task_drift: Dict[str, List[float]] = defaultdict(list)
    for k in keys:
        b = bd[k]; v = vd[k]
        # Drift averaged across the two pref directions (A and B).
        for which in ("a_A", "a_B"):
            d = np.array(b[which]) - np.array(v[which])
            per_task_drift[b["task_group"]].append(float(np.sqrt(np.mean(d ** 2))))
    overall = float(np.mean([d for dl in per_task_drift.values() for d in dl]))
    return {
        "overall_drift_normalized": overall,
        "per_task_drift": {tg: float(np.mean(v)) for tg, v in per_task_drift.items()},
        "n_compared": len(keys),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=list(PREF_CATEGORIES))
    ap.add_argument("--baseline_yaml", required=True)
    ap.add_argument("--vqa_yaml", required=True)
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--vqa_ckpt", required=True)
    ap.add_argument("--taskA_n_eps_per_task", type=int, default=5)
    ap.add_argument("--taskB_data_root", default=None,
                    help="If provided, enumerate ALL episodes under <root>/<task>_<pk>/. "
                         "Skipped if not given (taskA-only run).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_baseline", action="store_true",
                    help="Skip baseline pass (VQA only).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cat_cfg = PREF_CATEGORIES[args.category]
    print(f"[stage_a_gate] category={args.category} pref_keys={cat_cfg.pref_keys}")

    # Build taskA val dataset (used for both VQA-on-taskA and counterfactual)
    cfg_base = OmegaConf.load(args.baseline_yaml)
    val_ds = PrefHDF5Dataset(
        data_root_dir=cfg_base.datasets.vla_data.data_root_dir,
        split="val",
        chunk_size=int(cfg_base.datasets.vla_data.future_action_window_size) + 1,
        past_window=int(cfg_base.datasets.vla_data.past_action_window_size),
        include_state=bool(cfg_base.datasets.vla_data.include_state),
        cameras=tuple(cfg_base.datasets.vla_data.cameras),
        image_size=tuple(cfg_base.datasets.vla_data.image_size),
        stats_json_path=cfg_base.datasets.vla_data.stats_json_path,
        category=args.category,
    )
    print(f"[stage_a_gate] taskA val ds: {len(val_ds)} samples, "
          f"{len(set((t,e) for t,_,_,e,_ in val_ds._index))} unique episodes")

    # Sample episodes for taskA (VQA + counterfactual)
    taskA_eps = subsample_taskA_episodes(val_ds, args.taskA_n_eps_per_task, rng)
    print(f"[stage_a_gate] taskA picked: {len(taskA_eps)} episodes "
          f"({args.taskA_n_eps_per_task} per task_dir)")

    # For counterfactual: each episode gets ONE random frame
    taskA_frames = []
    for fs in taskA_eps:
        frame_idx = pick_random_frame_for_sample(val_ds, fs, rng)
        taskA_frames.append(FrameSample(
            task_dir=fs.task_dir, task_group=fs.task_group, pref_key=fs.pref_key,
            ep_id=fs.ep_id, frame_idx=frame_idx,
        ))

    # taskB enumeration (if provided)
    taskB_eps = []
    if args.taskB_data_root:
        taskB_eps = list_taskB_episodes(Path(args.taskB_data_root), cat_cfg.pref_keys)
        print(f"[stage_a_gate] taskB enumerated: {len(taskB_eps)} episodes from {args.taskB_data_root}")

    # === PASS 1: baseline ===
    results = {"category": args.category,
               "baseline_ckpt": args.baseline_ckpt,
               "vqa_ckpt": args.vqa_ckpt,
               "n_taskA_eps": len(taskA_eps),
               "n_taskB_eps": len(taskB_eps)}

    baseline_cache = []
    if not args.skip_baseline:
        print("\n========== PASS 1: BASELINE ckpt ==========")
        t0 = time.time()
        baseline, _ = build_framework(args.baseline_yaml, "QwenPI")
        load_ckpt_into_model(baseline, args.baseline_ckpt)
        baseline = baseline.cuda().eval()
        print(f"[stage_a_gate] baseline loaded in {time.time()-t0:.1f}s")

        b_cf, baseline_cache = run_counterfactual_action(
            baseline, taskA_frames, val_ds, args.category, tag="baseline")
        results["baseline_counterfactual_taskA"] = b_cf

        del baseline
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("[stage_a_gate] baseline unloaded")

    # === PASS 2: VQA ===
    print("\n========== PASS 2: VQA ckpt ==========")
    t0 = time.time()
    vqa, _ = build_framework(args.vqa_yaml, "QwenPI_VQA")
    load_ckpt_into_model(vqa, args.vqa_ckpt)
    vqa = vqa.cuda().eval()
    print(f"[stage_a_gate] VQA loaded in {time.time()-t0:.1f}s")

    # VQA acc on taskB — GATE
    if taskB_eps:
        results["vqa_taskB"] = run_vqa_acc(vqa, taskB_eps, args.category,
                                            taskB_root=Path(args.taskB_data_root),
                                            val_ds=None, tag="taskB")
    # VQA acc on taskA val
    results["vqa_taskA"] = run_vqa_acc(vqa, taskA_eps, args.category,
                                        taskB_root=None, val_ds=val_ds, tag="taskA")
    # Counterfactual on same taskA frames
    v_cf, vqa_cache = run_counterfactual_action(
        vqa, taskA_frames, val_ds, args.category, tag="vqa")
    results["vqa_counterfactual_taskA"] = v_cf

    # Drift
    if baseline_cache:
        results["baseline_vs_vqa_drift_taskA"] = compute_drift_between_ckpts(
            baseline_cache, vqa_cache)

    # === Gate verdict ===
    if taskB_eps:
        taskB_acc = results["vqa_taskB"]["per_task_acc"]
        passed = [tg for tg, a in taskB_acc.items() if a >= 0.90]
        results["gate"] = {
            "taskB_per_task_acc": taskB_acc,
            "passed_ge_0.90":     passed,
            "verdict": ("GREEN: ≥1 taskB task ≥90%, Stage B prep" if passed
                        else "RED: no taskB task ≥90%, diagnose before Stage B"),
        }

    # Save JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[stage_a_gate] saved JSON to {out_path}")

    # Save md summary (same stem)
    md_path = out_path.with_suffix(".md")
    md_path.write_text(_render_markdown(results))
    print(f"[stage_a_gate] saved md  to {md_path}")

    if "gate" in results:
        print(f"[stage_a_gate] GATE: {results['gate']['verdict']}")


def _render_markdown(r: dict) -> str:
    """Render the eval results dict as a paste-ready md summary."""
    L = []
    L.append(f"# Stage A gate — `{r['category']}`")
    L.append("")
    L.append(f"- baseline ckpt: `{r['baseline_ckpt']}`")
    L.append(f"- VQA ckpt:      `{r['vqa_ckpt']}`")
    L.append(f"- taskA val sample: {r['n_taskA_eps']} eps")
    L.append(f"- taskB sample:    {r['n_taskB_eps']} eps")
    L.append("")
    if "gate" in r:
        v = r["gate"]
        L.append(f"## GATE: **{v['verdict']}**")
        L.append("")
        L.append(f"- passed (≥0.90 on taskB): `{v['passed_ge_0.90']}`")
        L.append("")

    if "vqa_taskB" in r:
        b = r["vqa_taskB"]
        L.append("## ① VQA per-task accuracy — **taskB** (GATE METRIC)")
        L.append(f"Overall: **{b['overall_acc']:.3f}** (N={b['overall_n']}); "
                 f"confidence mean={b['confidence_mean']:.3f}, p10={b['confidence_p10']:.3f}")
        L.append("")
        L.append("| task_group | acc | N | pred_dist |")
        L.append("|---|---:|---:|---|")
        for tg, acc in sorted(b["per_task_acc"].items()):
            n = b["per_task_n"][tg]
            pd = b.get("pred_dist", {}).get(tg, {})
            L.append(f"| `{tg}` | {acc:.3f} | {n} | `{pd}` |")
        L.append("")

    if "vqa_taskA" in r:
        a = r["vqa_taskA"]
        L.append("## ② VQA per-task accuracy — taskA val (sanity)")
        L.append(f"Overall: **{a['overall_acc']:.3f}** (N={a['overall_n']}); "
                 f"confidence mean={a['confidence_mean']:.3f}")
        L.append("")
        L.append("| task_group | acc | N |")
        L.append("|---|---:|---:|")
        for tg, acc in sorted(a["per_task_acc"].items()):
            L.append(f"| `{tg}` | {acc:.3f} | {a['per_task_n'][tg]} |")
        L.append("")

    def _action_block(title: str, key: str):
        if key not in r:
            return
        block = r[key]
        L.append(f"## {title}")
        L.append(f"Overall counterfactual MSE (normalized 20D): **{block['overall_mse_normalized']:.4f}**")
        L.append("")
        L.append("| task_group | counterfactual MSE | sign acc (strong) | N |")
        L.append("|---|---:|---:|---:|")
        for tg, mse in sorted(block["per_task_mse"].items()):
            sa = block["sign_accuracy_strong"].get(tg, "—")
            sa_str = f"{sa:.3f}" if isinstance(sa, float) else sa
            L.append(f"| `{tg}` | {mse:.4f} | {sa_str} | {block['per_task_n'][tg]} |")
        L.append("")

    _action_block("③ Counterfactual action MSE — baseline ckpt (taskA val)",
                  "baseline_counterfactual_taskA")
    _action_block("④ Counterfactual action MSE — VQA ckpt (taskA val)",
                  "vqa_counterfactual_taskA")

    if "baseline_vs_vqa_drift_taskA" in r:
        d = r["baseline_vs_vqa_drift_taskA"]
        L.append("## ⑤ Baseline vs VQA action drift (sanity — should be small)")
        L.append(f"Overall drift (normalized): **{d['overall_drift_normalized']:.4f}**, N={d['n_compared']}")
        L.append("")
        L.append("| task_group | drift |")
        L.append("|---|---:|")
        for tg, dv in sorted(d["per_task_drift"].items()):
            L.append(f"| `{tg}` | {dv:.4f} |")
        L.append("")

    L.append("---")
    L.append("")
    L.append("Notes:")
    L.append("- Counterfactual MSE ≈ baseline≈VQA is **expected** (both action heads read pref from prompt token). VQA's value is in (1).")
    L.append("- Sign accuracy reported only for contact strong tasks (`put_boxdrink_dustbin`, `put_callbell_dustbin`, `put_fork_dustbin`) per doc 0522 §2.6.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
