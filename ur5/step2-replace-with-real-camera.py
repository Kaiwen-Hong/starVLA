#!/usr/bin/env python3
"""
Step 2: Replace dataset images with real camera capture.

Provides a RealSenseCamera class that grabs frames from /dev/video0
and converts them to PIL Images compatible with QwenPI predict_action.

Image processing chain (matching FastUMI training data):
    (1080, 1920, 3) YU12/I420 raw
      ↓ I420 → BGR → RGB
    (1080, 1920, 3) RGB uint8
      ↓ center crop (square)
    (1080, 1080, 3) RGB uint8
      ↓ PIL.Image.fromarray
    PIL.Image (1080x1080)

    predict_action() internally does:
      ↓ resize_images → (224, 224)
      ↓ AutoProcessor  → tensor + ImageNet normalization
"""

import cv2
import numpy as np
from PIL import Image


class RealCamera:
    """Capture frames from USB camera (V4L2) for QwenPI inference."""

    def __init__(self, dev: int = 0, width: int = 1920, height: int = 1080, fps: int = 30):
        self.dev = dev
        self.W = width
        self.H = height

        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{dev}")

        # Configure YU12 (I420) raw capture
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YU12"))
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize latency
        except Exception:
            pass

        # Verify
        fcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fcc = "".join([chr((fcc_int >> (8 * i)) & 0xFF) for i in range(4)])
        print(f"[RealCamera] /dev/video{dev} opened, FOURCC={fcc}, "
              f"{width}x{height}@{fps}fps")

    def grab_rgb(self) -> np.ndarray | None:
        """Grab one frame, return as RGB uint8 (H, W, 3) or None on failure."""
        ok, raw = self.cap.read()
        if not ok:
            return None
        yuv = np.ascontiguousarray(raw).reshape(self.H * 3 // 2, self.W)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def grab_pil(self) -> Image.Image | None:
        """Grab one frame, center-crop to square, return as PIL Image."""
        rgb = self.grab_rgb()
        if rgb is None:
            return None
        cropped = center_crop(rgb)
        return Image.fromarray(cropped)

    def close(self):
        self.cap.release()

    def __del__(self):
        self.close()


def center_crop(img: np.ndarray) -> np.ndarray:
    """Center-crop to the largest square."""
    h, w = img.shape[:2]
    crop_size = min(h, w)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    return img[top : top + crop_size, left : left + crop_size]


# ── Quick test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    cam = RealCamera(dev=0)
    print("Press 'q' to quit.")
    while True:
        pil_img = cam.grab_pil()
        if pil_img is None:
            continue
        # Show with cv2 (needs BGR)
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("RealCamera center-cropped", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.close()
    cv2.destroyAllWindows()
