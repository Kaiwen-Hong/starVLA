"""
Smoke tests for the VQA cotrain path (main-method Stage A).

Mirror baseline tests/test_smoke.py:
  test 1: dataset sample shape + VQA fields present + no pref leak in VQA input
  test 2: dataloader iteration + collate
  test 3 (--full): build Qwen_PI_VQA, forward over a mixed batch, get aggregated
                   action_loss + L_action/L_vqa/vqa_acc in output_dict.

Usage:
  python -m examples.preference.tests.test_smoke_vqa            # tests 1+2
  python -m examples.preference.tests.test_smoke_vqa --full     # + heavy test 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import os as _os
# Host-aware data root (lf=kaiwenh, H200=kevin).
for _candidate in (
    "/mnt/localssd/kaiwenh/pref/data/giveobj",
    "/mnt/localssd/kevin/pref/data/giveobj",
):
    if _os.path.isdir(_candidate):
        DATA_ROOT = _candidate
        break
else:
    raise FileNotFoundError("giveobj data not found on either kaiwenh or kevin local SSD")
STATS_JSON = "examples/preference/dataset/stats_giveobj_v1.json"


def test_dataset_sample():
    from examples.preference.dataset.pref_hdf5_vqa_dataset import PrefHDF5VQADataset
    from examples.preference.dataset.vqa_sample import get_vqa_clip_cache
    from examples.preference.dataset.prompt import _LEAK_RE

    # Subset (1 task group) to keep init under ~5s.
    ds = PrefHDF5VQADataset(
        data_root_dir=DATA_ROOT,
        split="val",
        include_state=True,
        task_groups=("give_boxdrink",),
        stats_json_path=STATS_JSON,
    )
    assert len(ds) > 0, "dataset empty"

    s = ds[0]
    # Action stream same shape as baseline.
    assert s["action"].shape == (16, 20), s["action"].shape
    assert s["state"].shape == (1, 20), s["state"].shape
    assert len(s["image"]) == 3, len(s["image"])

    # VQA fields present.
    assert "vqa_episode_id" in s and isinstance(s["vqa_episode_id"], tuple)
    assert s["vqa_episode_id"][0] == "val"
    assert isinstance(s["vqa_episode_id"][1], int)
    assert s["vqa_pref_key"] in ("25", "75")
    assert s["vqa_task_group"] in ("give_boxdrink",)

    # Lang field is the ACTION prompt (with " Preference: " suffix). That's
    # for the action stream only. The VQA stream constructs its OWN prompt
    # ("Question: ... Answer:") inside the framework — NO pref leak, NO
    # task hint. Verify the action prompt is also leak-free (baseline guarantee).
    base = s["lang"].split(" Preference:")[0]
    assert not _LEAK_RE.search(base), f"LEAK in action base prompt: {base!r}"

    # Cache should be registered under split name.
    cache = get_vqa_clip_cache("val")
    assert cache is not None
    assert cache.dtype == np.uint8
    assert cache.flags["C_CONTIGUOUS"]
    print("test_dataset_sample OK")
    print(f"  cache: shape={cache.shape}, {cache.nbytes/1024**3:.3f} GB uint8")


def test_dataloader_iter():
    from starVLA.dataloader import build_dataloader
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "datasets": {"vla_data": {
            "data_root_dir": DATA_ROOT,
            "per_device_batch_size": 4,
            "num_workers": 0,
            "future_action_window_size": 15,
            "past_action_window_size": 0,
            "include_state": True,
            "cameras": ["head_camera", "left_camera", "right_camera"],
            "image_size": [224, 224],
            "stats_json_path": STATS_JSON,
            "vqa_num_frames": 8,
            "vqa_camera": "head_camera",
        }},
        "output_dir": "/tmp/_test_smoke_vqa",
    })
    # PrefHDF5VQADataset doesn't take task_groups via factory kwargs, so to
    # keep this fast we go directly via the subclass for the lightweight check.
    from examples.preference.dataset.pref_hdf5_vqa_dataset import PrefHDF5VQADataset
    from examples.preference.dataset.pref_hdf5_dataset import collate_fn
    from torch.utils.data import DataLoader
    ds = PrefHDF5VQADataset(
        data_root_dir=DATA_ROOT, split="val", include_state=True,
        task_groups=("give_boxdrink",),
        stats_json_path=STATS_JSON,
    )
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(dl))
    assert isinstance(batch, list) and len(batch) == 4
    keys = sorted(batch[0].keys())
    assert "vqa_episode_id" in keys
    assert "vqa_pref_key" in keys
    print("test_dataloader_iter OK")
    print(f"  batch[0] keys: {keys}")


def test_full_forward():
    """Heavy: build Qwen_PI_VQA, run forward over a 4-sample batch."""
    import torch
    from omegaconf import OmegaConf
    from starVLA.model.framework import build_framework
    from examples.preference.dataset.pref_hdf5_vqa_dataset import PrefHDF5VQADataset
    from examples.preference.dataset.pref_hdf5_dataset import collate_fn
    from torch.utils.data import DataLoader

    cfg = OmegaConf.load(
        Path("examples/preference/train_files/starvla_pref_stage_a_vqa.yaml")
    )
    # Test rig: smaller per_device for memory; skip wandb.
    cfg.datasets.vla_data.per_device_batch_size = 4
    cfg.datasets.vla_data.num_workers = 0
    cfg.output_dir = "/tmp/_test_smoke_vqa"
    cfg.framework.qwenvl.base_vlm = "Qwen/Qwen3-VL-4B-Instruct"

    # Build the dataset FIRST so the VQA clip cache is registered before
    # framework.forward runs.
    ds = PrefHDF5VQADataset(
        data_root_dir=DATA_ROOT, split="val", include_state=True,
        task_groups=("give_boxdrink",),
        stats_json_path=STATS_JSON,
    )
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn, num_workers=0)
    batch = next(iter(dl))

    model = build_framework(cfg)
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()  # no dropout noise; we just want a clean loss value

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.forward(batch)

    assert "action_loss" in out
    assert "log/L_action" in out and "log/L_vqa" in out and "log/vqa_acc" in out
    print("test_full_forward OK")
    print(f"  action_loss (total) = {out['action_loss'].item():.4f}")
    print(f"  L_action            = {out['log/L_action']:.4f}")
    print(f"  L_vqa               = {out['log/L_vqa']:.4f}")
    print(f"  vqa_acc             = {out['log/vqa_acc']:.4f}")
    print(f"  vqa_n_samples       = {out['log/vqa_n_samples']}")
    for k in sorted(out):
        if k.startswith("log/vqa_acc/"):
            print(f"  {k:30s} = {out[k]:.4f}")

    # predict_preference round-trip (Stage B API).
    from examples.preference.dataset.vqa_sample import get_vqa_clip_cache
    from PIL import Image
    cache = get_vqa_clip_cache("val")
    clip_np = cache[0]
    clip = [Image.fromarray(clip_np[i]) for i in range(clip_np.shape[0])]
    pred = model.predict_preference(clip)
    print(f"  predict_preference  = {pred}")
    assert pred["label"] in ("low", "high")
    assert 0.0 <= pred["confidence"] <= 1.0
    assert abs(pred["p_low"] + pred["p_high"] - 1.0) < 1e-4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Also run the heavy test 3 (loads Qwen3-VL-4B).")
    args = parser.parse_args()

    test_dataset_sample()
    test_dataloader_iter()
    if args.full:
        test_full_forward()
    print("\nALL SMOKE TESTS OK")
