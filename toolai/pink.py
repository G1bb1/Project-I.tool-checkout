"""Ignore the pink paper shadows.

Why: on a swap, the tool sitting in another tool's slot leaves part of that slot's
pink shadow showing. A model could learn "this pink outline = wire cutter slot"
instead of the tool itself. Painting every pink pixel the same neutral grey takes
that shortcut away: the model only sees tool pixels (and plain paper).

How pink is recognised: by HUE in CIELAB (the angle of the a/b colour vector),
not by brightness. A shadow or a dimmer room makes pink darker and a bit less
saturated, but its hue angle barely moves, so the mask still finds it. Red or
orange tool handles have a different hue angle (red sits about 30-40 degrees away
from pink), so they are kept. `python ai_main.py pink` measures YOUR pink from
the empty-board reference photo and writes a preview so you can check that no
tool pixels are being erased.

A pixel is pink when
    chroma = sqrt(a^2 + b^2) >= min_chroma          (it is coloured, not grey/white/black)
    |hue - hue_deg| <= hue_tol_deg                   (and the colour is pink)
"""
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .common import CONFIG, load_json, save_json

PINK_FILE = CONFIG / "pink.json"


def _ab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[..., 1] - 128.0, lab[..., 2] - 128.0


def _angle_diff(h, centre):
    return np.abs((h - centre + 180.0) % 360.0 - 180.0)


def pink_mask(img, p):
    """Boolean mask of pink pixels (same height/width as img)."""
    a, b = _ab(img)
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    m = (chroma >= p["min_chroma"]) & (_angle_diff(hue, p["hue_deg"]) <= p["hue_tol_deg"])
    # Remove isolated specks, then grow by 1 px so the anti-aliased pink fringe goes too.
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return m.astype(bool)


def remove_pink(img, p):
    """Copy of img with pink pixels painted neutral grey. p=None or disabled -> unchanged."""
    if not p or not p.get("enabled", True):
        return img
    out = img.copy()
    out[pink_mask(img, p)] = p.get("fill", 128)
    return out


def load_pink(ai_cfg):
    """Measured pink (config/pink.json) if it exists, else the defaults in ai.json."""
    base = dict(ai_cfg.get("pink", {}))
    if not base.get("enabled", False):
        return None
    if PINK_FILE.exists():
        measured = load_json(PINK_FILE)
        base.update({k: measured[k] for k in ("hue_deg", "hue_tol_deg", "min_chroma") if k in measured})
        base["source"] = str(PINK_FILE.name)
    else:
        base["source"] = "ai.json defaults (run: python ai_main.py pink)"
    return base


def fit_pink(empty_img, slots, full_img=None, fill=128):
    """Measure the pink of the shadows from a photo of the EMPTY board.

    Pink pixels = pixels inside the slot boxes that are clearly more colourful than
    the bare paper between the slots. Their median hue is the centre; the tolerance
    covers 98% of them plus 6 degrees. min_chroma is a third of their median chroma,
    so pink that a shadow has darkened is still caught.
    """
    h, w = empty_img.shape[:2]
    inside = np.zeros((h, w), bool)
    for s in slots:
        inside[s["y"]:s["y"] + s["h"], s["x"]:s["x"] + s["w"]] = True
    # Bare paper = between the slots only (NOT the whole frame: the wood floor around the
    # board is brown, i.e. colourful, and would hide the pink).
    try:
        from .bridge import gap_mask
        paper = gap_mask(empty_img.shape, slots)
    except Exception:
        paper = ~inside
    # Even out exposure on the paper first, so a dark photo (exposure -4) still shows its pink.
    med = np.array([np.median(empty_img[..., c][paper]) for c in range(3)], np.float32)
    img = np.clip(empty_img.astype(np.float32) * np.clip(215.0 / np.maximum(med, 1.0), 0.4, 4.0),
                  0, 255).astype(np.uint8)
    a, b = _ab(img)
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    paper_chroma = float(np.median(chroma[paper]))
    cand = inside & (chroma > paper_chroma + 6.0)
    if cand.sum() < 0.02 * inside.sum():
        ins = chroma[inside]
        raise SystemExit(
            "Hardly any coloured pixels inside the slots of the empty-board photo.\n"
            f"  paper between slots: median brightness {med.mean():.0f}, colour {paper_chroma:.1f}\n"
            f"  inside slots: colour median {np.median(ins):.1f}, 90th pct {np.percentile(ins, 90):.1f} "
            f"(pink usually 15+ above the paper)\n"
            "Check: is it the EMPTY board with the pink shadows visible, does config/slots.json match "
            "this photo, and is the paper between the slots white? Or pass --empty <other empty.png>.")
    hc = hue[cand]
    # circular median: centre on the mean direction, then take the median offset
    mean_dir = np.degrees(np.arctan2(np.sin(np.radians(hc)).mean(), np.cos(np.radians(hc)).mean()))
    off = (hc - mean_dir + 180.0) % 360.0 - 180.0
    centre = float((mean_dir + np.median(off) + 180.0) % 360.0 - 180.0)
    spread = float(np.percentile(_angle_diff(hc, centre), 98))
    med_chroma = float(np.median(chroma[cand]))
    p = {"enabled": True, "hue_deg": round(centre, 2), "hue_tol_deg": round(min(35.0, spread + 6.0), 2),
         "min_chroma": round(max(6.0, med_chroma / 3.0, paper_chroma + 4.0), 2), "fill": fill,
         "measured": {"pink_pixels": int(cand.sum()), "median_chroma": round(med_chroma, 2),
                      "paper_chroma": round(paper_chroma, 2), "hue_spread_98pct": round(spread, 2)}}
    report = {}
    if full_img is not None:
        full_n = np.clip(full_img.astype(np.float32) * np.clip(215.0 / np.maximum(med, 1.0), 0.4, 4.0),
                         0, 255).astype(np.uint8)
        report = tool_pixels_erased(img, full_n, slots, p)
    return p, report


def tool_pixels_erased(empty_img, full_img, slots, p):
    """Per slot: share of TOOL pixels the mask would erase on the full board.
    Tool pixels = pixels that differ clearly between the full and the empty board
    (the tool covers its shadow), eroded by 4 px so the outline isn't counted.
    Should be near 0 for every slot."""
    m_full = pink_mask(full_img, p)
    diff = cv2.absdiff(cv2.GaussianBlur(full_img, (5, 5), 0), cv2.GaussianBlur(empty_img, (5, 5), 0)).max(2) > 40
    diff = cv2.erode(diff.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    out = {}
    for s in slots:
        sl = (slice(s["y"], s["y"] + s["h"]), slice(s["x"], s["x"] + s["w"]))
        tool = diff[sl]
        n = int(tool.sum())
        out[s["tool"]] = round(float((tool & m_full[sl]).sum()) / n, 4) if n else None
    return out


def write_preview(path, imgs, p, slots):
    """Side-by-side original / pink removed, for every image given."""
    rows = []
    for img in imgs:
        img, masked = img.copy(), remove_pink(img, p)
        for v in (img, masked):
            for s in slots:
                cv2.rectangle(v, (s["x"], s["y"]), (s["x"] + s["w"], s["y"] + s["h"]), (0, 200, 0), 1)
        rows.append(cv2.hconcat([img, masked]))
    out = cv2.vconcat(rows)
    scale = min(1.0, 2400 / out.shape[1])
    cv2.imwrite(str(path), cv2.resize(out, None, fx=scale, fy=scale))


def run_pink(slots, ai_cfg, empty_path, full_path=None):
    empty = cv2.imread(str(empty_path))
    if empty is None:
        raise SystemExit(f"Could not read {empty_path}")
    full = cv2.imread(str(full_path)) if full_path else None
    p, report = fit_pink(empty, slots, full, ai_cfg.get("pink", {}).get("fill", 128))
    p["created"] = datetime.now().isoformat(timespec="seconds")
    p["from"] = {"empty": str(empty_path), "full": str(full_path) if full_path else None}
    p["tool_pixels_erased"] = report
    save_json(PINK_FILE, p)
    prev = CONFIG / "pink_preview.png"
    write_preview(prev, [empty] + ([full] if full is not None else []), p, slots)
    print(f"Pink measured: hue {p['hue_deg']} deg +- {p['hue_tol_deg']}, min chroma {p['min_chroma']}")
    if report:
        print("Share of each tool's pixels the mask would erase (should be ~0):")
        for t, v in report.items():
            warn = "   <-- CHECK: tool colour close to pink" if v is not None and v > 0.05 else ""
            print(f"  {t:<24} {'-' if v is None else f'{v:.1%}'}{warn}")
    print(f"Saved {PINK_FILE} and {prev}. Look at the preview: every shadow grey, every tool untouched.")
    print("Freeze this before training. If you redo it, retrain (the model's meta.json keeps the values it used).")
    return p


def find_refs(root):
    """refs/<current>/empty.png and full.png from toolcheck, if present."""
    cur = Path(root) / "refs" / "current.json"
    if not cur.exists():
        return None, None
    d = Path(root) / "refs" / load_json(cur)["dir"]
    e, f = d / "empty.png", d / "full.png"
    return (e if e.exists() else None), (f if f.exists() else None)
