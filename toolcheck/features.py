"""Per-slot comparison features, plus the capture-quality (retake) check.

Two distances between a slot crop and a reference crop of the same slot:

structure  1 - zero-mean normalised cross-correlation on blurred greyscale.
           0 = same pattern, 1 = unrelated, up to 2 = inverted. Subtracting the
           mean and dividing by the spread makes it insensitive to uniform
           brightness changes.
colour     distance between the mean (a, b) chroma in CIELAB, ignoring lightness.
           Catches same-shape items of different colour (the two tapes).
           Not fully shadow-proof: a and b shrink toward grey as pixels darken.

Retake check (risk mitigation for "lighting change breaks the detector"):
The bare board BETWEEN the slots should look the same in every capture, because
no tool lives there. If part of it got darker, a shadow (or a light change) is
on the board, and the slot scores of that capture can't be trusted. The check
does not decide whether a tool is missing; it only says "retake this photo".

  dark   fraction of gap pixels that got more than dark_tol darker than the
         reference. One-sided on purpose: shadows only darken, while removing a
         dark tool that overhangs its box makes the paper brighter.
  shift  |median brightness ratio - 1| over the gap: a whole-room light change.

Limits for dark and shift come from the complete-kit calibration captures
(max(floor, mean + k*std)), like every other threshold. They are never tuned on
trial results.
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


def lighting_limits_from_samples(samples, k=4.0, dark_floor=0.02, shift_floor=0.05):
    """samples: list of lighting_change() dicts from complete-kit calibration captures.
    Same rule as the slot thresholds: max(floor, mean + k*std)."""
    out = {}
    for key, floor in (("dark", dark_floor), ("shift", shift_floor)):
        v = np.array([s[key] for s in samples], dtype=float)
        out[key] = max(floor, float(v.mean() + k * v.std(ddof=1))) if len(v) > 1 else floor
    return out


def retake_reason(light, limits, n_flagged, max_flagged=3):
    """Return a short reason string if the capture should be retaken, else "".

    n_flagged is the most slots either method flagged on this capture. A single
    job in the test plan changes at most 2 slots, so 3 or more at once is more
    likely a lighting problem than 3 lost tools. A real multi-tool loss is still
    reported: it will flag again on the retake, and the retake's result stands."""
    reasons = []
    if light["dark"] > limits["dark"]:
        reasons.append(f"shadow on board ({light['dark']:.0%} of gap darker)")
    if light["shift"] > limits["shift"]:
        reasons.append(f"room light changed ({light['shift']:.0%})")
    if max_flagged and n_flagged >= max_flagged:
        reasons.append(f"{n_flagged} slots flagged at once")
    return "; ".join(reasons)