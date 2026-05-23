"""
Smoke tests for the preference-conditioned VLA Stage A baseline.

Test 1 (~seconds): dataset returns one sample with the expected structure
                   and normalized action/state values in [-1, 1].
Test 2 (~seconds): build_dataloader returns a working DataLoader; one batch
                   has the expected per-sample structure.
Test 3 (heavy, requires GPU + ~8GB VRAM): full Qwen_PI forward → scalar loss.

Run:
  python -m examples.preference.tests.test_smoke           # tests 1+2
  python -m examples.preference.tests.test_smoke --full    # also test 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))


def test_dataset_sample():
    from examples.preference.dataset.pref_hdf5_dataset import PrefHDF5Dataset
    from examples.preference.dataset.prompt import _LEAK_RE

    ds = PrefHDF5Dataset(
        data_root_dir="/mnt/localssd/kaiwenh/pref/data/giveobj",
        split="val",
        include_state=True,
        stats_json_path="examples/preference/dataset/stats_giveobj_v1.json",
    )
    assert len(ds) > 0
    s = ds[0]

    assert isinstance(s["image"], list)
    assert len(s["image"]) == 3, f"expected 3 cameras, got {len(s['image'])}"
    assert s["image"][0].size == (224, 224), s["image"][0].size

    assert s["action"].shape == (16, 20), s["action"].shape
    assert s["state"].shape == (1, 20), s["state"].shape

    assert (s["action"] >= -1).all() and (s["action"] <= 1).all(), (
        f"action out of [-1, 1]: min={s['action'].min()}, max={s['action'].max()}"
    )
    assert (s["state"] >= -1).all() and (s["state"] <= 1).all(), (
        f"state out of [-1, 1]: min={s['state'].min()}, max={s['state'].max()}"
    )

    assert " Preference: " in s["lang"], s["lang"]
    base = s["lang"].split(" Preference:")[0]
    assert not _LEAK_RE.search(base), f"leak in base prompt: {base!r}"

    print(f"[test_dataset_sample]    OK (val len={len(ds)}; lang={s['lang']!r})")
    return s


def test_build_dataloader():
    from omegaconf import OmegaConf
    from starVLA.dataloader import build_dataloader

    cfg = OmegaConf.load("examples/preference/train_files/starvla_pref_stage_a_baseline.yaml")
    cfg.output_dir = "/tmp/pref_smoke_output"
    cfg.datasets.vla_data.num_workers = 0
    cfg.datasets.vla_data.per_device_batch_size = 2

    dl = build_dataloader(cfg, dataset_py="pref_hdf5")
    batch = next(iter(dl))
    assert isinstance(batch, list) and len(batch) == 2, type(batch)
    for s in batch:
        assert s["action"].shape == (16, 20), s["action"].shape
        assert s["state"].shape == (1, 20), s["state"].shape
        assert isinstance(s["image"], list) and len(s["image"]) == 3
        assert " Preference: " in s["lang"], s["lang"]
    print(f"[test_build_dataloader]  OK (batch size={len(batch)})")
    return cfg, batch


def test_full_forward(cfg, batch):
    import torch
    from starVLA.model.framework.QwenPI import Qwen_PI

    model = Qwen_PI(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test_full_forward]      device={device}")
    model = model.to(device)
    out = model(batch)
    loss = out["action_loss"]
    assert torch.is_tensor(loss) and loss.ndim == 0 and torch.isfinite(loss), loss
    print(f"[test_full_forward]      OK (action_loss={loss.item():.4f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true",
                   help="Also run full Qwen3-VL-4B forward (slow, GPU recommended)")
    args = p.parse_args()

    test_dataset_sample()
    cfg, batch = test_build_dataloader()
    if args.full:
        test_full_forward(cfg, batch)
    print("\nAll requested smoke tests passed ✓")


if __name__ == "__main__":
    main()
