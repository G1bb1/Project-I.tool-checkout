"""Synthetic end-to-end test: no webcam needed.

A fake camera renders a 4x4 board of drawn 'tools' with noise and small placement
jitter. Checks that removals are caught, a colour-only look-alike swap is caught,
untouched complete kits stay quiet, and the CSV/summary pipeline runs.
Run from the project root:  python -m tests.test_pipeline
"""
import builtins
import random
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from toolcheck import paths
from toolcheck.calibrate import thresholds_from_samples
from toolcheck.features import slot_scores
from toolcheck.methods import method_a, method_b, outcome
from toolcheck.stats import upper_bound

W, H = 1280, 720
TOOLS = [f"T{i}" for i in range(16)]
COLOURS = [(40, 40, 200), (40, 160, 40), (200, 60, 40), (30, 30, 30)]


def make_slots():
    slots = []
    for i, t in enumerate(TOOLS):
        r, c = divmod(i, 4)
        slots.append({"tool": t, "x": 60 + c * 300, "y": 20 + r * 175, "w": 260, "h": 150})
    return slots


SLOTS = make_slots()


def render(present, swap=None, jitter=2, rng=None, light=1.0):
    """present: set of tools on board. swap: (a, b) draws b's look in a's slot."""
    rng = rng or random.Random(0)
    img = np.full((H, W, 3), 235, np.uint8)
    for i, s in enumerate(SLOTS):
        x, y, w, h = s["x"], s["y"], s["w"], s["h"]
        cv2.rectangle(img, (x + 20, y + 20), (x + w - 20, y + h - 20), (150, 150, 150), 1)  # outline
        if s["tool"] not in present:
            continue
        idx = i
        if swap and s["tool"] in swap:
            other = swap[1] if s["tool"] == swap[0] else swap[0]
            idx = TOOLS.index(other)
        dx, dy = rng.randint(-jitter, jitter), rng.randint(-jitter, jitter)
        col = COLOURS[idx % 4]
        shape = idx // 4
        cx, cy = x + w // 2 + dx, y + h // 2 + dy
        if shape == 0:
            cv2.rectangle(img, (cx - 90, cy - 20), (cx + 90, cy + 20), col, -1)
        elif shape == 1:
            cv2.circle(img, (cx, cy), 50, col, -1)
            cv2.circle(img, (cx, cy), 20, (235, 235, 235), -1)
        elif shape == 2:
            cv2.line(img, (cx - 90, cy - 40), (cx + 90, cy + 40), col, 14)
        else:
            pts = np.array([[cx - 80, cy + 40], [cx, cy - 50], [cx + 80, cy + 40]])
            cv2.fillPoly(img, [pts], col)
    img = img.astype(np.float32) * light
    img += np.random.default_rng(rng.randint(0, 10**6)).normal(0, 3, img.shape)
    return img.clip(0, 255).astype(np.uint8)


def test_core():
    rng = random.Random(1)
    full_set = set(TOOLS)
    ref_full = render(full_set, rng=rng)
    ref_empty = render(set(), rng=rng)
    samples_a, samples_b, prev = [], [], render(full_set, rng=rng)
    for _ in range(12):
        img = render(full_set, rng=rng)
        samples_a.append(slot_scores(img, prev, SLOTS))
        samples_b.append(slot_scores(img, ref_full, SLOTS))
        prev = img
    thr = {"A": thresholds_from_samples(samples_a, TOOLS, 4.0, 0.05, 6.0),
           "B": thresholds_from_samples(samples_b, TOOLS, 4.0, 0.05, 6.0)}

    misses = {"A": 0, "B": 0}
    for t in TOOLS:
        checkout = render(full_set, rng=rng)
        signoff = render(full_set - {t}, rng=rng)
        ra = method_a(signoff, checkout, SLOTS, thr["A"], 2.0)
        rb = method_b(signoff, ref_full, ref_empty, SLOTS, thr["B"], 2.0)
        misses["A"] += outcome([t], ra["flagged"])["miss"]
        misses["B"] += outcome([t], rb["flagged"])["miss"]
        assert rb["labels"].get(t) == "missing", (t, rb["labels"])
    assert misses == {"A": 0, "B": 0}, misses

    # T0 and T1 share a shape and differ only in colour: a colour look-alike pair.
    checkout = render(full_set, rng=rng)
    signoff = render(full_set, swap=("T0", "T1"), rng=rng)
    for m, r in (("A", method_a(signoff, checkout, SLOTS, thr["A"], 2.0)),
                 ("B", method_b(signoff, ref_full, ref_empty, SLOTS, thr["B"], 2.0))):
        o = outcome(["T0", "T1"], r["flagged"])
        assert o["miss"] == 0, (m, r["flagged"])

    fa = 0
    for _ in range(20):
        checkout = render(full_set, rng=rng)
        signoff = render(full_set, rng=rng)
        fa += outcome([], method_a(signoff, checkout, SLOTS, thr["A"], 2.0)["flagged"])["false_alarm"]
        fa += outcome([], method_b(signoff, ref_full, ref_empty, SLOTS, thr["B"], 2.0)["flagged"])["false_alarm"]
    assert fa == 0, fa

    assert abs(upper_bound(0, 90) - (1 - 0.05 ** (1 / 90))) < 1e-6
    assert 0.03 < upper_bound(0, 90) < 0.034
    print("core: removals, colour swap, false alarms, bounds OK")


class FakeCam:
    """Plays the board the operator was told to set up, by reading the prompt."""
    def __init__(self):
        self.rng = random.Random(7)
        self.present, self.swap = set(TOOLS), None

    def resolution(self):
        return (W, H)

    def grab(self):
        return render(self.present, swap=self.swap, rng=self.rng), 0.2

    def close(self):
        pass


def test_session():
    from toolcheck import trials, analyze
    tmp = Path(tempfile.mkdtemp())
    old_data = paths.DATA
    trials.DATA = analyze.DATA = tmp
    cam = FakeCam()
    rng = random.Random(3)
    full = set(TOOLS)
    ref_full, ref_empty = render(full, rng=rng), render(set(), rng=rng)
    samples = [slot_scores(render(full, rng=rng), ref_full, SLOTS) for _ in range(12)]
    thr = thresholds_from_samples(samples, TOOLS, 4.0, 0.05, 6.0)
    thresholds = {"A": thr, "B": thr, "rule": "test", "created": "test"}
    settings = paths.load_settings()
    settings["trials"]["show_window"] = False
    settings["trials"]["seed"] = 11
    kit = {"lookalike_pairs": [["T0", "T1"]],
           "perturbations": ["No change (repeat check of an untouched complete kit)"]}

    real_input = builtins.input

    def fake_input(prompt=""):
        if "REMOVE:" in prompt:
            tool = prompt.split("REMOVE: ")[1].split("  (")[0]
            cam.present = full - {tool}
        elif "SWAP:" in prompt:
            cam.swap = ("T0", "T1")
        elif "Put the kit back" in prompt or "Kit complete?" in prompt:
            cam.present, cam.swap = set(full), None
        return ""

    builtins.input = fake_input
    try:
        sdir = trials.run_session(cam, SLOTS, {"dir": tmp, "full": ref_full, "empty": ref_empty},
                                  thresholds, settings, kit, 16, 2, 5)
        text = analyze.summarise(sdir.name)
    finally:
        builtins.input = real_input
        trials.DATA = analyze.DATA = old_data
    assert "| removal | A | 16 | 0 |" in text, text
    assert "| removal | B | 16 | 0 |" in text, text
    assert "| swap | B | 2 | 0 |" in text, text
    shutil.rmtree(tmp)
    print("session: run + analyze OK")


if __name__ == "__main__":
    test_core()
    test_session()
