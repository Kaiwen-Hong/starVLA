"""
Unit test: verify image-in-parquet dataset loading works correctly.

Tests two issues:
1. LeRobotMixtureDataset.__getitem__ hangs in while-True loop for image datasets
   (video_path never exists because there are no .mp4 files)
2. LeRobotSingleDataset.get_video() crashes for image datasets
   (tries to open non-existent video file via get_frames_by_timestamps)

Run:
    cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
    python test_image_dataset.py
"""

import json
import os
import sys
import signal
import numpy as np
from pathlib import Path

STARVLA_ROOT = Path("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA")
CUSTOM_ROOT = STARVLA_ROOT / "playground" / "Datasets" / "Custom"
ROBOTWIN_ROOT = STARVLA_ROOT / "playground" / "Datasets" / "RoboTwin"


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out (likely stuck in infinite loop)")


# ── Test 0: Verify dataset structure differences ──
def test_dataset_structure():
    """Confirm Custom is image-based and RoboTwin is video-based."""
    print("=" * 60)
    print("TEST 0: Dataset structure verification")
    print("=" * 60)

    custom_info = json.load(open(CUSTOM_ROOT / "adjust_bottle" / "meta" / "info.json"))
    print(f"  Custom total_videos: {custom_info['total_videos']}")
    assert custom_info["total_videos"] == 0, "Custom dataset should have 0 videos"

    # Check feature dtype
    cam_feat = custom_info["features"]["observation.images.cam_high"]
    print(f"  Custom cam_high dtype: {cam_feat['dtype']}")
    assert cam_feat["dtype"] == "image", "Custom dataset features should be dtype=image"

    # Check no videos directory
    assert not (CUSTOM_ROOT / "adjust_bottle" / "videos").exists(), \
        "Custom dataset should have no videos/ directory"

    robotwin_info = json.load(open(ROBOTWIN_ROOT / "adjust_bottle" / "meta" / "info.json"))
    print(f"  RoboTwin total_videos: {robotwin_info['total_videos']}")
    assert robotwin_info["total_videos"] > 0, "RoboTwin dataset should have videos"

    cam_feat_rw = robotwin_info["features"]["observation.images.cam_high"]
    print(f"  RoboTwin cam_high dtype: {cam_feat_rw['dtype']}")
    assert cam_feat_rw["dtype"] == "video", "RoboTwin dataset features should be dtype=video"

    print("  PASSED: Custom=image-in-parquet, RoboTwin=video\n")


# ── Test 1: get_video_path returns non-existent path for image datasets ──
def test_video_path_nonexistent():
    """The video_path for image datasets points to files that don't exist."""
    print("=" * 60)
    print("TEST 1: video_path points to non-existent files for image datasets")
    print("=" * 60)

    custom_info = json.load(open(CUSTOM_ROOT / "adjust_bottle" / "meta" / "info.json"))
    pattern = custom_info["video_path"]
    # Construct what get_video_path() would return
    video_file = pattern.format(episode_chunk=0, episode_index=0,
                                video_key="observation.images.cam_high")
    full_path = CUSTOM_ROOT / "adjust_bottle" / video_file
    print(f"  Would look for: {full_path}")
    print(f"  Exists: {full_path.exists()}")
    assert not full_path.exists(), "Video file should NOT exist for image dataset"
    print("  PASSED: video_path correctly does not exist for image datasets\n")


# ── Test 2: Image data is readable from parquet ──
def test_read_images_from_parquet():
    """Verify we can read and decode images from parquet files."""
    print("=" * 60)
    print("TEST 2: Read images from parquet")
    print("=" * 60)

    import pandas as pd
    from PIL import Image
    import io

    parquet_path = CUSTOM_ROOT / "adjust_bottle" / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(parquet_path)
    print(f"  Parquet shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    cam_col = "observation.images.cam_high"
    assert cam_col in df.columns, f"{cam_col} not in parquet columns"

    # Read first frame
    first_val = df[cam_col].iloc[0]
    assert isinstance(first_val, dict), f"Expected dict, got {type(first_val)}"
    assert "bytes" in first_val, "Expected 'bytes' key in image dict"

    img = Image.open(io.BytesIO(first_val["bytes"]))
    arr = np.array(img)
    print(f"  Decoded image: shape={arr.shape}, dtype={arr.dtype}")
    assert arr.shape == (480, 640, 3), f"Unexpected shape: {arr.shape}"
    assert arr.dtype == np.uint8

    # Read multiple frames (simulating what get_video should do)
    step_indices = [0, 1, 2]
    frames = []
    for idx in step_indices:
        img_dict = df[cam_col].iloc[idx]
        img = Image.open(io.BytesIO(img_dict["bytes"]))
        frames.append(np.array(img))
    frames_arr = np.stack(frames)
    print(f"  Multi-frame array: shape={frames_arr.shape}")
    assert frames_arr.shape == (3, 480, 640, 3)

    print("  PASSED: Images correctly readable from parquet\n")


# ── Test 3: LeRobotSingleDataset.get_video() with image dataset ──
def test_single_dataset_get_video():
    """Test that get_video() works for image datasets after the fix.
    Before the fix, this would crash with FileNotFoundError.
    """
    print("=" * 60)
    print("TEST 3: LeRobotSingleDataset.get_video() on image dataset")
    print("=" * 60)

    sys.path.insert(0, str(STARVLA_ROOT))
    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset

    dataset = make_LeRobotSingleDataset(
        data_root_dir=CUSTOM_ROOT,
        data_name="adjust_bottle",
        robot_type="robotwin",
        data_cfg={"video_backend": "pyav"},
    )

    print(f"  Dataset: {dataset.dataset_name}")
    print(f"  total_videos: {dataset.lerobot_info_meta.get('total_videos', 'N/A')}")
    print(f"  Trajectory IDs: {dataset.trajectory_ids[:3]}...")
    print(f"  Modality keys (video): {dataset.modality_keys.get('video', [])}")

    # Test get_step_data with first trajectory
    traj_id = dataset.trajectory_ids[0]
    base_index = 5  # pick a step in the middle

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout
    try:
        raw_data = dataset.get_step_data(traj_id, base_index)
        signal.alarm(0)
    except TimeoutError:
        print("  FAILED: get_step_data() timed out (likely stuck in video reading)")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"  FAILED: get_step_data() raised {type(e).__name__}: {e}")
        return False

    # Verify output
    video_keys = dataset.modality_keys.get("video", [])
    for vk in video_keys:
        assert vk in raw_data, f"Missing video key {vk} in raw_data"
        arr = raw_data[vk]
        print(f"  {vk}: shape={arr.shape}, dtype={arr.dtype}")
        assert arr.ndim == 4, f"Expected 4D (T,H,W,C), got {arr.ndim}D"
        assert arr.shape[2] == 640 and arr.shape[1] == 480, f"Unexpected resolution: {arr.shape}"

    print("  PASSED: get_step_data() works for image dataset\n")
    return True


# ── Test 4: LeRobotMixtureDataset.__getitem__ doesn't hang ──
def test_mixture_getitem():
    """Test that __getitem__ doesn't get stuck in infinite loop for image datasets.
    Before the fix, this would hang forever.
    """
    print("=" * 60)
    print("TEST 4: LeRobotMixtureDataset.__getitem__ on image dataset")
    print("=" * 60)

    sys.path.insert(0, str(STARVLA_ROOT))
    from omegaconf import OmegaConf
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    # Use custom_task1 (single task) for faster test
    data_cfg = OmegaConf.create({
        "data_root_dir": str(CUSTOM_ROOT),
        "data_mix": "custom_task1",
        "video_backend": "pyav",
    })
    dataset = get_vla_dataset(data_cfg=data_cfg)

    print(f"  Mixture dataset with {len(dataset.datasets)} sub-datasets")
    print(f"  Total length: {len(dataset)}")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60 second timeout
    try:
        sample = dataset[0]
        signal.alarm(0)
    except TimeoutError:
        print("  FAILED: __getitem__ timed out (stuck in while True loop)")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"  FAILED: __getitem__ raised {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"  Sample keys: {list(sample.keys())}")
    if "image" in sample:
        print(f"  Image count: {len(sample['image'])}")
        for i, img in enumerate(sample['image']):
            print(f"    image[{i}]: {type(img).__name__} size={getattr(img, 'size', 'N/A')}")
    if "action" in sample:
        print(f"  Action shape: {sample['action'].shape}")
    if "lang" in sample:
        print(f"  Language: {sample['lang'][:80]}...")

    print("  PASSED: __getitem__ returned successfully\n")
    return True


if __name__ == "__main__":
    test_dataset_structure()
    test_video_path_nonexistent()
    test_read_images_from_parquet()

    # These tests require the starVLA env and will fail before the fix
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS (require starVLA imports)")
    print("=" * 60 + "\n")

    ok3 = test_single_dataset_get_video()
    if ok3:
        ok4 = test_mixture_getitem()
    else:
        print("Skipping Test 4 (Test 3 failed)\n")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Tests 0-2: Basic structure verification (always pass)")
    print(f"Test 3 (get_video on image dataset): {'PASS' if ok3 else 'FAIL'}")
    if ok3:
        print(f"Test 4 (mixture __getitem__): {'PASS' if ok4 else 'FAIL'}")
