"""Webcam capture. Averages a few frames per capture to cut sensor noise."""
import os
import time

import cv2
import numpy as np


class Camera:
    def __init__(self, cam_cfg):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(cam_cfg["index"], backend)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera index {cam_cfg['index']}. "
                             "Try index 1 in config/settings.json.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        if cam_cfg.get("disable_autofocus"):
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # Let auto exposure settle first, then lock it if asked.
        for _ in range(cam_cfg.get("warmup_frames", 30)):
            self.cap.read()
        if cam_cfg.get("lock_exposure"):
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DirectShow: 0.25 = manual
            self.cap.set(cv2.CAP_PROP_EXPOSURE, cam_cfg.get("exposure", -6))
            for _ in range(10):
                self.cap.read()
        self.flush = cam_cfg.get("flush_frames", 5)
        self.avg = max(1, cam_cfg.get("average_frames", 5))

    def resolution(self):
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def live_frame(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def grab(self):
        """Return (image, seconds taken). Flushes stale buffered frames first."""
        t0 = time.perf_counter()
        for _ in range(self.flush):
            self.cap.read()
        acc, n = None, 0
        for _ in range(self.avg):
            ok, frame = self.cap.read()
            if not ok:
                continue
            f = frame.astype(np.float32)
            acc = f if acc is None else acc + f
            n += 1
        if n == 0:
            raise RuntimeError("Camera returned no frames.")
        img = (acc / n).clip(0, 255).astype(np.uint8)
        return img, time.perf_counter() - t0

    def close(self):
        self.cap.release()
