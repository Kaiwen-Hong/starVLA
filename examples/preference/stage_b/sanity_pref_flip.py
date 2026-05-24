"""
Stage B sanity check (§7.5) — pref-flip diagnostic.

On a Stage B-trained ckpt:
  - Pick 10 taskB episodes (5 GT_low + 5 GT_high)  ← uses dir-derived GT,
    BUT this script is OUT of the train path (eval-only). Stage B
    training pipeline never reads dir-name GT (per §2.3 firewall);
    here we use it legitimately, just like §3.2 launch gate.
  - For each episode:
      * Choose grasp-relevant frame (gripper-state-detected grasp_t,
        falling back to mid-episode if no grasp closure detected)
      * Build sample (image, state) at that frame
      * Run predict_action twice:
          - with " Preference: low contact"
          - with " Preference: high contact"
      * Compute Δaction[chunk_max_step, z_axis] = action_high[k,2] - action_low[k,2]
        where k = argmax over chunk steps of |Δz|  (the "most-pref-sensitive" step)
  - Report: sign-acc (Δz > 0 for boxdrink3 → high contact = higher z); mean Δz
  - PASS: sign-acc ≥ 8/10 AND |mean Δz| clearly > 0

If FAIL, see §4.3 Caveat: first check earlier ckpts (under-training), not
declare Stage B failure.

CLI:
  python -m examples.preference.stage_b.sanity_pref_flip \
      --category contact \
      --policy_yaml examples/preference/train_files/starvla_pref_stage_b_main_contact.yaml \
      --policy_ckpt results/Checkpoints/pref_main_stage_b_v1_contact/checkpoints/steps_1500_pytorch_model.pt \
      --taskB_data_root /mnt/localssd/kaiwenh/pref/data/contact/taskB \
      --task_group put_boxdrink3_plate \
      --n_per_class 5 \
      --axis 2  # 0=L_x, 1=L_y, 2=L_z (default for standing object)
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
from PIL import Image

from examples.preference.dataset.prompt import PREF_CATEGORIES, build_action_prompt
from examples.preference.dataset.vqa_sample import find_grasp_frame
from examples.preference.eval.stage_a_gate import build_framework, load_ckpt_into_model


def load_frame_obs(
    h5_path: Path,
    frame_idx: int,
    cameras: List[str] = ("head_camera", "left_camera", "right_camera"),
    image_size=(224, 224),
) -> dict:
    """Load image (3 cams) + state at frame_idx. Image at training resolution."""
    H, W = image_size
    with h5py.File(h5_path, "r") as h5:
        images = []
        for cam in cameras:
            raw = h5[f"observation/{cam}/rgb"][frame_idx]
            data = raw.tobytes() if hasattr(raw, "tobytes") else raw
            img = Image.open(io.BytesIO(data)).convert("RGB")
            if img.size != (W, H):
                img = img.resize((W, H))
            images.append(img)
        # State at this frame (20D, same as Stage A)
        l_ee = h5["endpose/left_endpose"][frame_idx:frame_idx + 1]
        r_ee = h5["endpose/right_endpose"][frame_idx:frame_idx + 1]
        l_gr = h5["endpose/left_gripper"][frame_idx:frame_idx + 1]
        r_gr = h5["endpose/right_gripper"][frame_idx:frame_idx + 1]
    # Build EE 20D (need to redo the 7→9 quat-to-6D conversion)
    from examples.preference.dataset.rotation import quat_xyzw_to_6d
    L_xyz = l_ee[:, :3]
    L_6d = quat_xyzw_to_6d(l_ee[:, 3:7])
    L_grip = l_gr.reshape(-1, 1)
    R_xyz = r_ee[:, :3]
    R_6d = quat_xyzw_to_6d(r_ee[:, 3:7])
    R_grip = r_gr.reshape(-1, 1)
    state_raw = np.concatenate([L_xyz, L_6d, L_grip, R_xyz, R_6d, R_grip],
                                 axis=-1).astype(np.float32)
    return {"image": images, "state_raw": state_raw}


def normalize_state(state_raw: np.ndarray, stats_json_path: Path) -> np.ndarray:
    s = json.loads(Path(stats_json_path).read_text())
    st = s.get("state") or s["action"]
    q01 = np.asarray(st["q01"], dtype=np.float32)
    q99 = np.asarray(st["q99"], dtype=np.float32)
    span = q99 - q01
    safe = span != 0
    out = np.zeros_like(state_raw)
    out[..., safe] = 2 * (state_raw[..., safe] - q01[safe]) / span[safe] - 1
    return np.clip(out, -1.0, 1.0).astype(np.float16)


def pick_grasp_or_mid_frame(h5_path: Path) -> int:
    """Get grasp_t from gripper state; fall back to T*0.45 if no closure."""
    with h5py.File(h5_path, "r") as h5:
        T = h5["observation/head_camera/rgb"].shape[0]
        grasp_t = find_grasp_frame(h5)
    if grasp_t is None:
        return int(0.45 * (T - 1))
    return min(grasp_t, T - 17)  # leave room for chunk_size=16 lookahead


def sample_episodes(
    taskB_root: Path,
    task_group: str,
    n_per_class: int,
    seed: int = 42,
) -> List[tuple]:
    """Return [(task_dir, gt_pref_key, ep_id), ...]. Uses dir-name GT
    (eval-only; not in train path)."""
    import random
    rng = random.Random(seed)
    out = []
    for gt_pk in ("25", "75"):
        d = taskB_root / f"{task_group}_{gt_pk}" / "data"
        eps = sorted(int(p.stem.replace("episode", "")) for p in d.glob("episode*.hdf5"))
        rng.shuffle(eps)
        for ep_id in eps[:n_per_class]:
            out.append((f"{task_group}_{gt_pk}", gt_pk, ep_id))
    return out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="contact")
    ap.add_argument("--policy_yaml", required=True)
    ap.add_argument("--policy_ckpt", required=True)
    ap.add_argument("--taskB_data_root", required=True)
    ap.add_argument("--task_group", default="put_boxdrink3_plate")
    ap.add_argument("--n_per_class", type=int, default=5)
    ap.add_argument("--axis", type=int, default=2,
                    help="action dim to check Δ sign on (default 2 = L_z; "
                         "for tabletop/x-axis tasks use 0)")
    ap.add_argument("--expected_sign", type=int, default=+1,
                    help="expected sign of (a_high − a_low) on the chosen axis "
                         "(+1 for standing object → high contact = +z)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="Optional path to dump per-episode records as JSON")
    ap.add_argument("--framework_name", default="QwenPI",
                    help="Framework class name to instantiate (default QwenPI; "
                         "use QwenPI_VQA for Stage A VQA ckpts).")
    ap.add_argument("--dump_chunks", action="store_true",
                    help="Dump full a_low and a_high action chunks (16, 20) per episode")
    ap.add_argument("--frame_fraction", type=float, default=None,
                    help="If set, use frame at this fraction of T (e.g. 0.30 = pre-grasp). "
                         "Default None = use gripper-detected grasp_t (fallback 0.45*T).")
    args = ap.parse_args()

    cat = PREF_CATEGORIES[args.category]
    pref_labels = cat.pref_labels  # 25 → "low contact", 75 → "high contact"

    # Need stats for state normalization
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.policy_yaml)
    stats_path = cfg.datasets.vla_data.stats_json_path

    # Load policy
    print(f"[sanity] loading policy ckpt: {args.policy_ckpt} (framework={args.framework_name})")
    t0 = time.time()
    model, _ = build_framework(args.policy_yaml, args.framework_name)
    load_ckpt_into_model(model, args.policy_ckpt)
    model = model.cuda().eval()
    print(f"[sanity] loaded in {time.time()-t0:.1f}s\n")

    # Sample episodes
    eps = sample_episodes(Path(args.taskB_data_root), args.task_group, args.n_per_class, args.seed)
    print(f"[sanity] {len(eps)} episodes ({args.n_per_class}/class)\n")

    base_prompt = cat.clean_templates[args.task_group]
    prompt_low  = f"{base_prompt} Preference: {pref_labels['25']}"
    prompt_high = f"{base_prompt} Preference: {pref_labels['75']}"
    print(f"  prompt_low  = {prompt_low!r}")
    print(f"  prompt_high = {prompt_high!r}\n")

    records = []
    for i, (task_dir, gt_pk, ep_id) in enumerate(eps):
        h5p = Path(args.taskB_data_root) / task_dir / "data" / f"episode{ep_id}.hdf5"
        if args.frame_fraction is not None:
            with h5py.File(h5p, "r") as h5:
                T = h5["observation/head_camera/rgb"].shape[0]
            frame_idx = max(0, min(T - 17, int(args.frame_fraction * (T - 1))))
        else:
            frame_idx = pick_grasp_or_mid_frame(h5p)
        obs = load_frame_obs(h5p, frame_idx)
        state = normalize_state(obs["state_raw"], stats_path)

        ex_low  = {"image": obs["image"], "lang": prompt_low,  "state": state}
        ex_high = {"image": obs["image"], "lang": prompt_high, "state": state}
        a_low  = model.predict_action([ex_low])["normalized_actions"][0]   # (T, D)
        a_high = model.predict_action([ex_high])["normalized_actions"][0]

        delta = a_high[:, args.axis] - a_low[:, args.axis]  # (T,)
        # Take chunk-max |Δ| step (most pref-sensitive)
        k_max = int(np.argmax(np.abs(delta)))
        delta_at_max = float(delta[k_max])
        mean_delta = float(delta.mean())

        rec = {
            "task_dir": task_dir, "ep_id": ep_id, "gt_pk": gt_pk,
            "frame_idx": frame_idx,
            "delta_at_chunk_max": delta_at_max,
            "delta_chunk_mean": mean_delta,
            "k_max": k_max,
            "sign_correct": int(np.sign(delta_at_max) == args.expected_sign),
        }
        if args.dump_chunks:
            rec["a_low"] = a_low.tolist()
            rec["a_high"] = a_high.tolist()
        records.append(rec)
        print(f"  [{i+1}/{len(eps)}] {task_dir}/ep{ep_id}  frame={frame_idx}  "
              f"Δ[k={k_max}]={delta_at_max:+.4f}  mean={mean_delta:+.4f}  "
              f"sign_correct={records[-1]['sign_correct']}")

    # === Summary ===
    n_correct = sum(r["sign_correct"] for r in records)
    sign_acc = n_correct / len(records)
    deltas = np.array([r["delta_at_chunk_max"] for r in records])

    print(f"\n" + "=" * 70)
    print(f"SANITY (pref-flip) on {args.policy_ckpt}")
    print(f"=" * 70)
    print(f"  sign-acc:    {n_correct}/{len(records)} = {sign_acc:.2f}  (need ≥ 8/10)")
    print(f"  mean Δ:      {deltas.mean():+.4f}  (need clearly > 0; "
          f"sign should match expected +1)")
    print(f"  std Δ:       {deltas.std():.4f}")
    print(f"  range Δ:     [{deltas.min():+.4f}, {deltas.max():+.4f}]")

    if sign_acc >= 0.8 and deltas.mean() > 0.01:
        print(f"\n  ✓ PASS — policy responds to pref-flip in expected direction")
    elif sign_acc < 0.8:
        print(f"\n  ✗ FAIL (sign-acc < 8/10) — see §4.3 Caveat:")
        print(f"     1. Try earlier ckpts (under-training)")
        print(f"     2. Try later ckpts or bump max_train_steps to 2500")
        print(f"     3. Check §3.2 launch-gate cache acc still ~1.000")
        print(f"     4. Check prompt is really pref-conditioned (printed above)")
        print(f"     Only after above → suspect method.")
    else:
        print(f"\n  ⚠ MARGINAL (sign-acc OK but |mean Δ| small) — try later ckpts")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "policy_ckpt": args.policy_ckpt, "axis": args.axis,
            "expected_sign": args.expected_sign,
            "sign_acc": sign_acc, "mean_delta": float(deltas.mean()),
            "records": records,
        }, indent=2))
        print(f"\n[sanity] saved {args.out}")


if __name__ == "__main__":
    main()
