"""Per-slot comparison features.

Two distances between a slot crop and a reference crop of the same slot:

structure  1 - zero-mean normalised cross-correlation on blurred greyscale.
           0 = same pattern, 1 = unrelated, up to 2 = inverted. Subtracting the
           mean and dividing by the spread makes it insensitive to uniform
           brightness changes.
colour     distance between the mean (a, b) chroma in CIELAB, ignoring lightness.
           Catches same-shape items of different colour (the two tapes).
"""
import cv2
import numpy as np


def crop(img, slot):
    x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
    return img[y:y + h, x:x + w]


def _grey(c, blur_sigma):
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if blur_sigma and blur_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), blur_sigma)
    return g


def structure_distance(a, b, blur_sigma=2.0, shift_px=0):
    """With shift_px > 0, the reference crop is inset by shift_px on every side and
    slid over the current crop; the best match is kept. A tool sitting a few pixels
    off its traced position then still matches, so small placement differences are
    not read as a change."""
    ga, gb = _grey(a, blur_sigma), _grey(b, blur_sigma)
    m = int(shift_px)
    if m > 0 and gb.shape[0] > 4 * m and gb.shape[1] > 4 * m:
        tpl = gb[m:-m, m:-m]
        if tpl.std() < 1e-3 and ga.std() < 1e-3:
            return 0.0
        if tpl.std() < 1e-3 or ga.std() < 1e-3:
            return 1.0
        res = cv2.matchTemplate(ga, tpl, cv2.TM_CCOEFF_NORMED)
        return 1.0 - float(res.max())
    za, zb = ga - ga.mean(), gb - gb.mean()
    denom = float(np.sqrt((za * za).sum() * (zb * zb).sum()))
    if denom < 1e-6:
        # Both crops flat: same. One flat, one not: different.
        flat_a, flat_b = za.std() < 1e-3, zb.std() < 1e-3
        return 0.0 if (flat_a and flat_b) else 1.0
    return 1.0 - float((za * zb).sum()) / denom


def colour_distance(a, b):
    la = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    lb = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    da = la[:, 1:].mean(axis=0) - lb[:, 1:].mean(axis=0)
    return float(np.hypot(da[0], da[1]))


def slot_scores(img, ref, slots, blur_sigma=2.0, shift_px=10):
    """{tool: {"s": structure, "c": colour}} for every slot."""
    out = {}
    for slot in slots:
        a, b = crop(img, slot), crop(ref, slot)
        out[slot["tool"]] = {"s": structure_distance(a, b, blur_sigma, shift_px),
                             "c": colour_distance(a, b)}
    return out
