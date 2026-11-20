"""Shared helpers for the AI tool-recognition stage.

The AI stage sits on the same shadow board as toolcheck:
config/slots.json gives one box per slot (from `python main.py define-slots`),
and every slot crop is classified as WHICH tool is in it, or "empty".
A slot is flagged when the predicted content is not the tool that belongs there.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
DATASET = ROOT / "dataset"
MODELS = ROOT / "models"

EMPTY = "empty"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_ai_config():
    """config/ai.json with defaults filled in for any block an older file lacks."""
    cfg = load_json(CONFIG / "ai.json")
    defaults = {
        "pink": {"enabled": False},
        "board": {"footprint": {"enabled": False}, "exposure": {"enabled": False}, "tool_only": {"enabled": False}},
        "augment": {"copies": 0},
        "training": {"exclude_swap_crops": False, "skip_suspects": False},
        "decision": {"min_confidence": 0.6, "clear_confidence": 0.9, "ai_can_clear_a_flags": True},
        "lighting_override": {"enabled": False, "mode": "all_own", "confidence": 0.9},
        "retake": {"max_retakes": None, "skip_many_slot_retake_if_ai_confident": False},
        "checkout_check": {"enabled": True},
        "test_mode": {"allow_force_accept": False},
        "absent_tools": [],
        "plan": {"cycle_perturbations": False},
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v if not isinstance(v, (dict, list)) else type(v)(v))
        if isinstance(v, dict):
            for kk, vv in v.items():
                cfg[k].setdefault(kk, vv)
    return cfg


def load_slots():
    p = CONFIG / "slots.json"
    if not p.exists():
        raise SystemExit("No config/slots.json. Run: python main.py define-slots")
    return load_json(p)["slots"]


def load_camera_settings():
    return load_json(CONFIG / "settings.json")["camera"]


def lookalike_pairs(ai_cfg, tools):
    """Pairs from config/ai.json; every name must be a slot's tool."""
    pairs = [tuple(p) for p in ai_cfg.get("lookalike_pairs", [])]
    for a, b in pairs:
        if a not in tools or b not in tools:
            raise SystemExit(f"Look-alike pair {a!r}/{b!r} in config/ai.json is not in slots.json")
    return pairs


def class_names(tools):
    """Class order is fixed: the slot tools in slot order, then 'empty'."""
    return list(tools) + [EMPTY]


def crop_slot(img, slot, margin):
    """Slot box grown by `margin` (fraction of its size) on every side, clipped to the image."""
    h, w = img.shape[:2]
    mx, my = int(slot["w"] * margin), int(slot["h"] * margin)
    x0, y0 = max(0, slot["x"] - mx), max(0, slot["y"] - my)
    x1, y1 = min(w, slot["x"] + slot["w"] + mx), min(h, slot["y"] + slot["h"] + my)
    return img[y0:y1, x0:x1]


def letterbox(crop, size):
    """Resize keeping the aspect ratio, pad the rest with grey. Returns size x size x 3 uint8."""
    h, w = crop.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), 128, np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    out[y:y + nh, x:x + nw] = small
    return out


def read_manifest(session_dir):
    with open(Path(session_dir) / "manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_bad_crops():
    """config/ai_bad_crops.txt: crops you looked at and rejected (session/crops/file.png per line)."""
    p = CONFIG / "ai_bad_crops.txt"
    if not p.exists():
        return set()
    return {ln.strip().replace("\\", "/") for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")}


def row_skip_reason(session, row, ai_cfg, training):
    """Why a manifest row is left out ('' = keep). Swap crops only leave TRAINING."""
    tcfg = ai_cfg.get("training", {})
    if f"{session}/{row['file']}" in load_bad_crops():
        return "ai_bad_crops.txt"
    if tcfg.get("skip_suspects") and row.get("suspect") == "1":
        return "suspect label"
    if training and tcfg.get("exclude_swap_crops") and "swap_crop" in row.get("tags", ""):
        return "swap crop (training only)"
    return ""


def list_sessions():
    """Dataset sessions, oldest first (by the 'started' time in session.json, then name).
    Sorting by time matters because imported sessions are named tc-<id> and would
    otherwise sort after every collect session."""
    if not DATASET.exists():
        return []
    found = []
    for p in DATASET.iterdir():
        if not (p / "manifest.csv").exists():
            continue
        started = ""
        if (p / "session.json").exists():
            started = load_json(p / "session.json").get("started", "")
        found.append((started, p.name))
    return [name for _, name in sorted(found)]


def load_toolcheck_settings():
    """toolcheck's own settings.json (retake block, features), or {} if not found."""
    p = CONFIG / "settings.json"
    return load_json(p) if p.exists() else {}


def expected_contents(tools, absent=()):
    """What each slot SHOULD hold: its own tool, or EMPTY for a tool that is not in the kit
    this session (ai.json absent_tools, e.g. a tool left at school)."""
    return {t: (EMPTY if t in absent else t) for t in tools}


def slot_truth(tools, block_truth, block, absent=()):
    """What is physically in each slot for a toolcheck trial plan entry.
    removal [t]   -> slot t is empty
    swap [a, b]   -> slot a holds b, slot b holds a
    complete []   -> every slot holds its own tool"""
    content = expected_contents(tools, absent)
    if block == "swap" and len(block_truth) == 2:
        a, b = block_truth
        content[a], content[b] = b, a
    else:
        for t in block_truth:
            content[t] = EMPTY
    return content


def upper_bound(failures, n, conf=0.95):
    """Exact (Clopper-Pearson) one-sided upper bound on a failure rate."""
    if n == 0:
        return 1.0
    from scipy.stats import beta
    if failures >= n:
        return 1.0
    return float(beta.ppf(conf, failures + 1, n - failures))


def peak_memory_mb():
    """Peak resident memory of this process, in MB (None where the OS can't say)."""
    try:
        import resource
        import sys
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024
    except ImportError:          # Windows
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / (1024 * 1024)
        except Exception:
            return None
