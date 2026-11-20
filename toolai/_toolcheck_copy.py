"""Copy of the toolcheck functions the hybrid needs (Nov 12 methods.py, Nov 17 features.py).
Used ONLY when toolcheck itself cannot be imported. Do not edit: if your toolcheck
differs, the hybrid uses yours automatically."""
import time

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


# ---------------------------------------------------------------------------
# Retake check
# ---------------------------------------------------------------------------

def gap_mask(shape, slots, pad=8):
    """Bare-board pixels: inside the bounding box of all slots, outside every slot
    box grown by pad px (so a tool overhanging its box a little is not counted).
    Taking only the board area keeps people moving in the background out of it."""
    h, w = shape[:2]
    x0 = max(0, min(s["x"] for s in slots))
    y0 = max(0, min(s["y"] for s in slots))
    x1 = min(w, max(s["x"] + s["w"] for s in slots))
    y1 = min(h, max(s["y"] + s["h"] for s in slots))
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    for s in slots:
        m[max(0, s["y"] - pad):s["y"] + s["h"] + pad,
          max(0, s["x"] - pad):s["x"] + s["w"] + pad] = False
    if m.sum() < 0.01 * h * w:
        raise ValueError("Less than 1% of the frame is bare board between slots; "
                         "the retake check needs some visible board.")
    return m


def lighting_change(img, ref, mask, blur_sigma=8.0, dark_tol=0.15):
    """{"dark": fraction of gap pixels > dark_tol darker, "shift": |median ratio - 1|}.
    Heavy blur so pencil lines and camera noise don't count, only broad light changes."""
    gi, gr = _grey(img, blur_sigma), _grey(ref, blur_sigma)
    ratio = ((gi + 1.0) / (gr + 1.0))[mask]
    return {"dark": float(np.mean(ratio < 1.0 - dark_tol)),
            "shift": abs(float(np.median(ratio)) - 1.0)}



def flag(scores, thr):
    return [t for t, sc in scores.items()
            if sc["s"] > thr[t]["s"] or sc["c"] > thr[t]["c"]]


def method_a(signoff, checkout, slots, thr_a, blur, shift=10):
    t0 = time.perf_counter()
    scores = slot_scores(signoff, checkout, slots, blur, shift)
    flagged = flag(scores, thr_a)
    labels = {t: "changed" for t in flagged}
    return {"flagged": flagged, "labels": labels, "scores": scores,
            "process_s": time.perf_counter() - t0}


