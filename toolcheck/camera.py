"""Webcam capture. Averages a few frames per capture to cut sensor noise.

Exposure handling (config/settings.json -> "camera"):
  lock_exposure: true + exposure_mode:
    "fixed"        (old behaviour) after the warm-up frames, switch to manual and
                   set "exposure" (e.g. -6). Gain and white balance are only fixed if
                   "gain" / "wb_temperature" are given; otherwise the camera keeps
                   whatever its auto chose during the warm-up, which is why the
                   picture could come out brighter or darker from run to run.
    "settle_lock"  (Nov 20) leave EVERYTHING on auto for "settle_s" seconds (default
                   3) while frames are read, so the camera adapts to the light that is
                   there now. Then read back what auto chose (exposure, gain, white
                   balance), switch all three to manual and write those same values
                   back, so nothing drifts for the rest of the run. The locked values
                   are printed and kept in self.locked (log them in the logbook).
  Tip: once a settle_lock run looks right, copy the printed values into "exposure",
  "gain" and "wb_temperature" with exposure_mode "fixed", so every session uses
  identical settings.
  Some webcams/drivers cannot report what auto chose (readback returns 0 or -1, or
  the value doesn't change). Then the configured "exposure" is used instead, and a
  warning says so.
"""
import os
import time

import cv2
import numpy as np

_DSHOW = os.name == "nt"
# OpenCV's CAP_PROP_AUTO_EXPOSURE values differ by backend.
_AUTO_ON, _AUTO_OFF = (0.75, 0.25) if _DSHOW else (3, 1)


def _num(v):
    return v is not None and v == v and v not in (-1.0,)     # not NaN, not "unsupported"


class Camera:
    def __init__(self, cam_cfg):
        backend = cv2.CAP_DSHOW if _DSHOW else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(cam_cfg["index"], backend)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera index {cam_cfg['index']}. "
                             "Try index 1 in config/settings.json.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        if cam_cfg.get("disable_autofocus"):
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.locked = {}
        mode = cam_cfg.get("exposure_mode", "fixed")
        if cam_cfg.get("lock_exposure") and mode == "settle_lock":
            self._settle_lock(cam_cfg)
        else:
            # Let auto exposure settle first, then lock it if asked.
            for _ in range(cam_cfg.get("warmup_frames", 30)):
                self.cap.read()
            if cam_cfg.get("lock_exposure"):
                self._manual(cam_cfg.get("exposure", -6), cam_cfg.get("gain"), cam_cfg.get("wb_temperature"))
        self.flush = cam_cfg.get("flush_frames", 5)
        self.avg = max(1, cam_cfg.get("average_frames", 5))

    def _manual(self, exposure, gain=None, wb=None):
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, _AUTO_OFF)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        if gain is not None:
            self.cap.set(cv2.CAP_PROP_GAIN, gain)
        if wb is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
        for _ in range(10):
            self.cap.read()
        self.locked = {"exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),
                       "gain": self.cap.get(cv2.CAP_PROP_GAIN),
                       "wb_temperature": self.cap.get(cv2.CAP_PROP_WB_TEMPERATURE)}

    def _settle_lock(self, cam_cfg):
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, _AUTO_ON)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        settle = float(cam_cfg.get("settle_s", 3.0))
        print(f"Camera: auto exposure / gain / white balance adapting for {settle:.0f} s "
              "(keep the light as it will be for the run, hands off the board)...")
        t_end = time.perf_counter() + settle
        n = 0
        while time.perf_counter() < t_end or n < cam_cfg.get("warmup_frames", 30):
            self.cap.read()
            n += 1
        exp = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        gain = self.cap.get(cv2.CAP_PROP_GAIN)
        wb = self.cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
        if not _num(exp) or exp == 0:
            print(f"  WARNING: this camera did not report the auto exposure (got {exp}); "
                  f"using the configured exposure {cam_cfg.get('exposure', -6)} instead.")
            exp = cam_cfg.get("exposure", -6)
        self._manual(exp, gain if _num(gain) else None, wb if _num(wb) and wb > 0 else None)
        print("  Locked for this run: " + ", ".join(f"{k} {v:g}" for k, v in self.locked.items())
              + "  <- write these in the logbook")

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
