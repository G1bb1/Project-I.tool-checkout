"""Camera access: reuses toolcheck's Camera (exposure lock, frame averaging) when present."""
import time

import cv2


class _SimpleCamera:
    def __init__(self, cfg):
        self.cap = cv2.VideoCapture(cfg.get("index", 0))
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera index {cfg.get('index', 0)}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.get("width", 1280))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.get("height", 720))
        for _ in range(cfg.get("warmup_frames", 30)):
            self.cap.read()

    def grab(self):
        t0 = time.perf_counter()
        for _ in range(3):
            self.cap.read()
        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit("Camera read failed")
        return frame, time.perf_counter() - t0

    def resolution(self):
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def close(self):
        self.cap.release()


def open_camera(cfg):
    try:
        from toolcheck.camera import Camera   # same capture settings as the trials
        return Camera(cfg)
    except ImportError:
        return _SimpleCamera(cfg)
