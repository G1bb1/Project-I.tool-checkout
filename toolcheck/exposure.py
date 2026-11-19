"""Live exposure check: pick the camera exposure by numbers, not by eye.

Shows the live camera with a readout of how bright the bare board is and how much
of it is clipped (pure white). Clipped pixels are painted red so you can see where.

Keys (click the window first):
  ] or +   one step brighter (exposure + 1, roughly double the light)
  [ or -   one step darker
  q        quit and print the value to put in config/settings.json

Target: board median about 200-230 out of 255, clipped under 0.5 %.
Too dark = more noise relative to the signal and weaker colour scores.
Clipped = a shadowed and an unshadowed patch can both read 255, which hides
shadows from the retake check and hides small changes from the slot scores.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from .features import gap_mask

TARGET_LO, TARGET_HI = 200, 230
CLIP_LEVEL, CLIP_MAX = 250, 0.005


def board_region(shape, slots_file="config/slots.json"):
    """Bare board between slots if slots are defined, else the whole frame."""
    p = Path(slots_file)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            slots = data["slots"] if isinstance(data, dict) else data
            return gap_mask(shape, slots), "board between slots"
        except (ValueError, KeyError, TypeError):
            pass
    return np.ones(shape[:2], dtype=bool), "whole frame (no slots defined yet)"


def exposure_stats(frame, mask):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    v = g[mask]
    med = float(np.median(v))
    clipped = float(np.mean(v >= CLIP_LEVEL))
    if clipped > CLIP_MAX:
        verdict = "CLIPPED - go darker"
    elif med < TARGET_LO:
        verdict = "DARK - go brighter" if med < TARGET_LO - 30 else "a bit dark"
    elif med > TARGET_HI:
        verdict = "a bit bright"
    else:
        verdict = "GOOD"
    return {"median": med, "clipped": clipped, "verdict": verdict}


def draw_readout(frame, stats, exposure, region_name):
    out = frame.copy()
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out[g >= CLIP_LEVEL] = (0, 0, 255)
    colour = (0, 200, 0) if stats["verdict"] == "GOOD" else (0, 165, 255)
    lines = [f"exposure {exposure:g}   [ darker   ] brighter   q quit",
             f"board median {stats['median']:.0f} / 255  (target {TARGET_LO}-{TARGET_HI})",
             f"clipped {stats['clipped']:.1%}  (max {CLIP_MAX:.1%})   -> {stats['verdict']}",
             f"measured on: {region_name}"]
    cv2.rectangle(out, (0, 0), (760, 30 + 30 * len(lines)), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(out, t, (12, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    colour if i == 2 else (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run_exposure_check(cam, camera_settings):
    """cam: an open toolcheck Camera (uses its cv2.VideoCapture, cam.cap)."""
    exposure = float(camera_settings.get("exposure", -6))
    cam.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual mode on DirectShow
    cam.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    mask, region = None, ""
    while True:
        ok, frame = cam.cap.read()
        if not ok:
            continue
        if mask is None or mask.shape != frame.shape[:2]:
            mask, region = board_region(frame.shape)
        stats = exposure_stats(frame, mask)
        view = draw_readout(frame, stats, exposure, region)
        cv2.imshow("toolcheck exposure", cv2.resize(view, None, fx=0.9, fy=0.9))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("]"), ord("+"), ord("=")):
            exposure += 1
        elif key in (ord("["), ord("-")):
            exposure -= 1
        elif key == ord("q"):
            break
        else:
            continue
        cam.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        for _ in range(5):  # let the new exposure settle
            cam.cap.read()
    cv2.destroyAllWindows()
    print(f"\nLast exposure: {exposure:g}  (board median {stats['median']:.0f}, "
          f"clipped {stats['clipped']:.1%}, {stats['verdict']})")
    print(f'Put this in config/settings.json -> "camera": {{ "exposure": {exposure:g} }}')
    print("Then redo capture-refs and calibrate before any trials.")
    return exposure