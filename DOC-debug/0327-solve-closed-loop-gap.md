# Closed-Loop Gap: Open-Loop Good, Closed-Loop Bad (2026-03-27)

## Symptom

- **Open-loop evaluation** (`step1-fm-dynamic-open_loop_offline_evaluation.py`) on the QwenPI dynamic checkpoint (`fastumi_pickandplace_qwenPI_dynamic_1/steps_20000`) shows very good MSE/L1 metrics.
- **Closed-loop control** (`step9-c-dynamic-closed-loop-smooth.py`) and **live inference visualization** (`step5-fm-dynamic-sync-live-inference-viz.py`) produce wildly incorrect predictions from the same model.

## Root Cause

**Image resolution mismatch between training and live inference.**

During training, the dataset loader resizes all images to **224x224**:

```python
# starVLA/dataloader/gr00t_lerobot/datasets.py, _pack_sample() line 1262
image = Image.fromarray(image).resize((224, 224))
```

During live inference, `QwenPI.predict_action()` only resizes if `config.datasets.vla_data.image_size` is set:

```python
# starVLA/model/framework/QwenPI.py, predict_action() lines 172-174
train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
if train_obs_image_size:
    batch_images = resize_images(batch_images, target_size=train_obs_image_size)
```

This checkpoint's `config.yaml` does **not** have `image_size`, so no resize happens. The live camera feeds **1080x1080** images directly to Qwen2.5-VL's processor, which generates far more vision tokens than the 224x224 images used during training. The action head's cross-attention receives completely different-length context, leading to garbage predictions.

### Why open-loop eval was unaffected

Open-loop evaluation loads data through the same dataset pipeline (`get_vla_dataset` + `DataLoader`), so images go through `_pack_sample()` and arrive at 224x224 — identical to training.

## Fix

Resize images to 224x224 in `build_example()` before passing to the model, matching the training-time `_pack_sample` behavior:

```python
TRAIN_IMAGE_SIZE = (224, 224)

def build_example(image, instruction, state_10d=None):
    image = image.resize(TRAIN_IMAGE_SIZE)  # <-- added
    example = {"image": [image], "lang": instruction}
    if state_10d is not None:
        example["state"] = state_10d.reshape(1, -1)
    return example
```

### Files modified

- `ur5/step5-fm-dynamic-sync-live-inference-viz.py` — added resize in `build_example()`
- `ur5/step9-c-dynamic-closed-loop-smooth.py` — added resize in `build_example()`

## Lesson

When a checkpoint's config does not include `image_size`, `predict_action()` skips the resize step. Any inference script that feeds images at a different resolution than training will silently produce bad predictions. Always check what resolution `_pack_sample()` uses for the training dataset and match it at inference time.
