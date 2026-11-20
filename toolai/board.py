"""What the model is allowed to see of each slot, and at what exposure.

Two clean-ups applied to every photo before slots are cut out, in training AND
at check time (frozen in the model's meta.json, like the pink values):

1. Exposure / white balance evened out using the bare paper.
   The paper between the slots is the same paper in every session, so each
   colour channel is scaled until the paper's median is `target` (default 215).
   Fixes the exposure -6 vs -4 difference between sessions and auto white
   balance drift. Only the AI sees this; Method A is unchanged.

2. Footprint: only the slot's OWN shadow (plus a margin) is kept.
   From the empty-board photo of that session's references (refs/<dir>/empty.png),
   the pink shadow inside each slot box is found, holes filled (tape cores), and
   grown by `grow_px`. Everything outside it is painted grey. This removes the
   wood floor at the board edge and the neighbouring tool when two slot boxes
   overlap. A tool swapped into a smaller slot can stick out past the shadow;
   grow_px leaves room for that.
"""
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

from .bridge import gap_mask
from .common import ROOT, load_json
from .pink import pink_mask

FILL = 128


def _box(img_shape, slot, margin):
    h, w = img_shape[:2]
    mx, my = int(slot["w"] * margin), int(slot["h"] * margin)
    return (max(0, slot["x"] - mx), max(0, slot["y"] - my),
            min(w, slot["x"] + slot["w"] + mx), min(h, slot["y"] + slot["h"] + my))


def _fill_holes(m):
    m = m.astype(np.uint8)
    flood = m.copy()
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, pad, (0, 0), 1)
    return (m | (1 - flood)).astype(bool)


def footprint(empty_img, slot, margin, pink, grow_px):
    """Boolean mask in crop coordinates (slot box + margin): the slot's own shadow, grown."""
    x0, y0, x1, y1 = _box(empty_img.shape, slot, margin)
    pm = pink_mask(empty_img, pink)
    own = np.zeros(pm.shape, np.uint8)
    own[slot["y"]:slot["y"] + slot["h"], slot["x"]:slot["x"] + slot["w"]] = 1
    pm = (pm & own.astype(bool)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(pm, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.isin(lab, 1 + np.where(areas >= 0.3 * areas.max())[0])
    keep = _fill_holes(keep)
    k = 2 * grow_px + 1
    keep = cv2.dilate(keep.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return keep[y0:y1, x0:x1].astype(bool)


class Board:
    """Slots + per-slot footprints + exposure normalisation for one camera setup."""

    def __init__(self, slots, margin, cfg, pink=None, empty_img=None, source=""):
        self.slots, self.margin, self.cfg, self.source = slots, margin, cfg or {}, source
        self.footprints = [None] * len(slots)
        self.missing_footprints = []
        self._gap = None
        self.empty_norm = self.normalise(empty_img) if empty_img is not None else None
        fp = self.cfg.get("footprint", {})
        if fp.get("enabled") and self.empty_norm is not None and pink:
            for j, s in enumerate(slots):
                self.footprints[j] = footprint(self.empty_norm, s, margin, pink, fp.get("grow_px", 12))
                if self.footprints[j] is None:
                    self.missing_footprints.append(s["tool"])
        tc = self.cfg.get("tool_only", {})
        self.tool_only = bool(tc.get("enabled")) and self.empty_norm is not None
        self.window = tc.get("window_px") or int(1.15 * max(max(s["w"], s["h"]) for s in slots))
        self.last_content = []      # per slot: share of the footprint covered by "tool" pixels

    def has_footprints(self):
        return any(f is not None for f in self.footprints)

    def normalise(self, img):
        ex = self.cfg.get("exposure", {})
        if not ex.get("enabled"):
            return img
        if self._gap is None or self._gap.shape != img.shape[:2]:
            self._gap = gap_mask(img.shape, self.slots)
        med = np.array([np.median(img[..., c][self._gap]) for c in range(3)], np.float32)
        scale = np.clip(ex.get("target", 215) / np.maximum(med, 1.0), 0.4, 3.0)
        return np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    def _changed(self, a, b):
        """Pixels of `a` that differ from the empty board `b` by more than `diff`, allowing
        the board to sit up to shift_px off (a pixel only counts if NO empty-board pixel within
        shift_px looks like it). Without this, a board or camera moved by a few px since
        capture-refs turns every shadow edge into a thin "tool"."""
        tc = self.cfg.get("tool_only", {})
        k = 2 * int(tc.get("shift_px", 6)) + 1
        a = cv2.GaussianBlur(a, (5, 5), 0).astype(np.int16)
        b = cv2.GaussianBlur(b, (5, 5), 0)
        ker = np.ones((k, k), np.uint8)
        lo, hi = cv2.erode(b, ker).astype(np.int16), cv2.dilate(b, ker).astype(np.int16)
        d = tc.get("diff", 35)
        return ((a < lo - d) | (a > hi + d)).any(2)

    def ref_mismatch(self, img):
        """Share of the bare paper BETWEEN slots that no longer matches the empty-board photo.
        Near 0 when the reference is good. High = board/camera moved or light very different
        since capture-refs: retake the references before trusting the tool-only view."""
        if self.empty_norm is None:
            return None
        if self._gap is None or self._gap.shape != img.shape[:2]:
            self._gap = gap_mask(img.shape, self.slots)
        return float(self._changed(self.normalise(img), self.empty_norm)[self._gap].mean())

    def _content(self, img, j, s):
        """Tool pixels of slot j: pixels that differ from the SAME slot on the empty board
        (after exposure evening), inside the slot's own footprint, small specks removed."""
        tc = self.cfg.get("tool_only", {})
        x0, y0, x1, y1 = _box(img.shape, s, self.margin)
        m = self._changed(img[y0:y1, x0:x1], self.empty_norm[y0:y1, x0:x1]).astype(np.uint8)
        fpm = self.footprints[j]
        area = m.size
        if fpm is not None and fpm.shape == m.shape:
            m &= fpm.astype(np.uint8)
            area = int(fpm.sum())
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = []
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] < tc.get("min_area_px", 40):
                continue
            if fpm is None:
                # No footprint (no-pink version): drop pieces whose centre is nearer another
                # slot's centre, i.e. a neighbouring tool reaching into this box.
                gx, gy = cen[k][0] + x0, cen[k][1] + y0
                d = [(gx - (o["x"] + o["w"] / 2)) ** 2 + (gy - (o["y"] + o["h"] / 2)) ** 2 for o in self.slots]
                if int(np.argmin(d)) != j:
                    continue
            keep.append(k)
        m = np.isin(lab, keep)
        return (x0, y0, x1, y1), m, float(m.sum()) / max(1, area)

    def crops(self, img):
        img = self.normalise(img)
        if self.tool_only:
            return self._tool_crops(img)
        out = []
        for s, fpm in zip(self.slots, self.footprints):
            x0, y0, x1, y1 = _box(img.shape, s, self.margin)
            c = img[y0:y1, x0:x1].copy()
            if fpm is not None and fpm.shape == c.shape[:2]:
                c[~fpm] = FILL
            out.append(c)
        return out

    def _tool_crops(self, img):
        """Tool-only view: a fixed-size window centred on the tool, every non-tool pixel grey.
        No pink, no shadow outline, no slot box shape, same scale for every slot: the model
        can only use the tool itself. An empty slot gives a plain grey window."""
        W = self.window
        out, self.last_content = [], []
        for j, s in enumerate(self.slots):
            (x0, y0, x1, y1), m, share = self._content(img, j, s)
            self.last_content.append(share)
            win = np.full((W, W, 3), FILL, np.uint8)
            if m.any():
                ys, xs = np.nonzero(m)
                cy, cx = int(ys.mean()), int(xs.mean())
                src = img[y0:y1, x0:x1]
                top, left = cy - W // 2, cx - W // 2
                sy0, sx0 = max(0, top), max(0, left)
                sy1, sx1 = min(m.shape[0], top + W), min(m.shape[1], left + W)
                patch = win[sy0 - top:sy1 - top, sx0 - left:sx1 - left]
                mm = m[sy0:sy1, sx0:sx1]
                patch[mm] = src[sy0:sy1, sx0:sx1][mm]
            out.append(win)
        return out

    def preview(self, img, path):
        """Every slot crop of one photo side by side, as the model will see them."""
        cs = [cv2.resize(c, (160, 160)) for c in self.crops(img)]
        rows = [cv2.hconcat(cs[i:i + 6] + [np.full((160, 160, 3), 255, np.uint8)] * (6 - len(cs[i:i + 6])))
                for i in range(0, len(cs), 6)]
        cv2.imwrite(str(path), cv2.vconcat(rows))


def refs_empty_for(refs_dir):
    """empty.png for a session's refs_dir (session.json stores the Windows path)."""
    if not refs_dir:
        return None
    name = PureWindowsPath(str(refs_dir)).name or Path(str(refs_dir)).name
    p = ROOT / "refs" / name / "empty.png"
    return p if p.exists() else None


def current_refs_empty():
    cur = ROOT / "refs" / "current.json"
    if not cur.exists():
        return None
    p = ROOT / "refs" / load_json(cur)["dir"] / "empty.png"
    return p if p.exists() else None


def make_board(slots, margin, cfg, pink, empty_path, source=""):
    empty = cv2.imread(str(empty_path)) if empty_path else None
    b = Board(slots, margin, cfg, pink, empty, source or str(empty_path or "no empty.png"))
    b.empty = empty
    fp_on = (cfg or {}).get("footprint", {}).get("enabled")
    if fp_on and empty is None:
        print(f"  WARNING: no empty-board photo ({source}); slots are cut as plain boxes, "
              "so neighbours and floor can show.")
    elif fp_on and not pink:
        print("  WARNING: pink not measured yet (run `pink` first); footprints are off.")
    if b.missing_footprints:
        print(f"  WARNING: no pink shadow found for {b.missing_footprints}; those slots are plain boxes.")
    return b
