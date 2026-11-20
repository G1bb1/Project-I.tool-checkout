"""Training-time variety for lighting and placement (training sessions only).

Your sessions have one copy of each tool and mostly one lighting setup, so each
training crop gets a few extra copies with:
  brightness / gamma      room brighter or darker (the exposure drift you saw on Nov 17)
  colour temperature      warmer or cooler light (auto white balance drift, a desk lamp)
  soft shadow             a darkened band with a blurred edge (hand or body shadow)
  rotate / shift          a tool put back a little crooked or off-centre
  glare (v3)              a soft blob of the TOOL greyed out. In a tool-only crop, glare
                          makes part of the tool look like bare paper, so the
                          comparison with the empty board drops it and it turns grey.
                          Teaches "part of a tool is still that tool, not empty"
                          (session 3 Measuring tape read as empty in 40 of 63 crops).
Pink is removed AFTER this step, so the pink mask is also trained against
shadowed and tinted pink. Held-out crops are never augmented.
"""
import cv2
import numpy as np

FILL = 128          # the grey that tool-only crops use for 'not the tool'


def make_augmenter(cfg, rng):
    def aug(img):
        h, w = img.shape[:2]
        out = img.astype(np.float32)

        # placement
        ang = rng.uniform(-cfg["rotate_deg"], cfg["rotate_deg"])
        tx, ty = rng.uniform(-cfg["shift"], cfg["shift"], 2) * (w, h)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        M[:, 2] += (tx, ty)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # overall light
        out *= rng.uniform(*cfg["brightness"])
        g = rng.uniform(*cfg["gamma"])
        out = 255.0 * (np.clip(out, 0, 255) / 255.0) ** g
        t = rng.uniform(-cfg["colour_temp"], cfg["colour_temp"])
        out[..., 0] *= 1 - t          # blue
        out[..., 2] *= 1 + t          # red

        # soft shadow: a random half-plane darkened, with a blurred edge
        if rng.random() < cfg["shadow_prob"]:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            th = rng.uniform(0, 2 * np.pi)
            d = (xx - w / 2) * np.cos(th) + (yy - h / 2) * np.sin(th) - rng.uniform(-0.3, 0.3) * w
            edge = rng.uniform(2, 0.15 * w)
            inside = 1 / (1 + np.exp(-d / edge))
            k = rng.uniform(*cfg["shadow_strength"])
            out *= (1 - (1 - k) * inside)[..., None]

        # glare: grey out a soft blob of the tool itself (never the background)
        if rng.random() < cfg.get("glare_prob", 0.0):
            tool = np.abs(img.astype(np.float32) - FILL).max(axis=2) > 12
            ys, xs = np.nonzero(tool)
            if len(ys) > 20:
                i = rng.integers(len(ys))
                r = rng.uniform(*cfg.get("glare_size", [0.1, 0.35])) * w
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                blob = np.exp(-(((xx - xs[i]) ** 2 + (yy - ys[i]) ** 2) / (r * r)) ** 2)
                blob = cv2.warpAffine(blob * tool.astype(np.float32), M, (w, h))
                out = out * (1 - blob[..., None]) + FILL * blob[..., None]

        out += rng.normal(0, 2.0, out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)
    return aug
