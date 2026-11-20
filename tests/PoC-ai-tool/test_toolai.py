"""Synthetic end-to-end test: no webcam, no real tools, no real photos.

Draws a fake 12-slot shadow board with the real kit names: tool-shaped PINK
shadows, look-alike pairs that differ only a little, a red-handled tool (to check
the pink mask keeps red), and shadows / room-light changes. Then it runs the whole
chain the way you will on your PC:

  fake toolcheck sessions (data/<id>/ with images, trials.csv, session.json)
  -> import-trials -> pink -> train (svm, cnn) -> evaluate -> replay
  -> live hybrid session with a fake camera: checkout check, lighting retake,
     lighting override, test-only force-accept, retake categories, summary.

Run from the project root:  python -m tests.test_toolai
Everything happens in a temporary folder; your data/, dataset/ and models/ are untouched.
Synthetic accuracy only says the code runs. It is NOT evidence.
"""
import csv
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np



def _find_root():
    """Project root = the nearest folder above this file that holds toolai/ and ai_main.py,
    so the test works in tests/ or in a subfolder such as tests/PoC-ai-tool/."""
    for d in Path(__file__).resolve().parents:
        if (d / "toolai").is_dir() and (d / "ai_main.py").exists():
            return d
    raise SystemExit("Can't find the project root (a folder with toolai/ and ai_main.py) above "
                     f"{Path(__file__).resolve()}")


SRC = _find_root()
TOOLS = ["Measuring tape", "Wire stripper", "Pliers", "Scissors", "Box cutter", "Wire cutter",
         "Nippers", "Tweezers", "Flathead screwdriver", "Knife", "Painter's tape", "Electrical tape"]
# shape, colour (BGR), size. Look-alikes differ only slightly. Box cutter has a RED handle.
LOOK = {"Measuring tape": ("square", (40, 200, 230), 34), "Wire stripper": ("tri", (60, 60, 200), 40),
        "Pliers": ("tri", (160, 60, 40), 44), "Scissors": ("cross", (60, 40, 30), 36),
        "Box cutter": ("bar", (30, 30, 210), 44), "Wire cutter": ("tri", (60, 60, 200), 34),
        "Nippers": ("tri", (70, 70, 190), 28), "Tweezers": ("thin", (170, 170, 170), 40),
        "Flathead screwdriver": ("bar", (20, 180, 220), 50), "Knife": ("bar", (150, 150, 150), 46),
        "Painter's tape": ("circle", (90, 200, 230), 30), "Electrical tape": ("circle", (35, 35, 35), 30)}
SLOTS = [{"tool": t, "x": 12 + (i % 4) * 158, "y": 12 + (i // 4) * 155, "w": 128, "h": 125}
         for i, t in enumerate(TOOLS)]
PINK = (190, 140, 245)


def draw_shape(img, tool, cx, cy, col, grow=0, rng=None):
    shape, _, s = LOOK[tool]
    s += grow
    if shape == "square":
        cv2.rectangle(img, (cx - s, cy - s), (cx + s, cy + s), col, -1)
    elif shape == "bar":
        cv2.rectangle(img, (cx - s, cy - 9 - grow), (cx + s, cy + 9 + grow), col, -1)
    elif shape == "thin":
        cv2.rectangle(img, (cx - s, cy - 3 - grow), (cx + s, cy + 3 + grow), col, -1)
    elif shape == "tri":
        pts = np.array([[cx - s, cy + s // 2], [cx + s, cy + s // 2], [cx, cy - s]], np.int32)
        cv2.fillPoly(img, [pts], col)
    elif shape == "circle":
        cv2.circle(img, (cx, cy), s, col, -1)
        if not grow:
            cv2.circle(img, (cx, cy), s // 2, (250, 250, 250), -1)
    elif shape == "cross":
        cv2.line(img, (cx - s, cy - s), (cx + s, cy + s), col, 10 + 2 * grow)
        cv2.line(img, (cx - s, cy + s), (cx + s, cy - s), col, 10 + 2 * grow)


def render(content, rng, light="normal", gain=1.0):
    img = np.full((480, 640, 3), 238, np.uint8)
    for s in SLOTS:
        cx, cy = s["x"] + s["w"] // 2, s["y"] + s["h"] // 2
        draw_shape(img, s["tool"], cx, cy, PINK, grow=6)              # tool-shaped pink shadow
    for s in SLOTS:
        t = content[s["tool"]]
        if t != "empty":
            cx, cy = s["x"] + s["w"] // 2 + rng.randint(-4, 4), s["y"] + s["h"] // 2 + rng.randint(-4, 4)
            col = tuple(int(np.clip(c + rng.gauss(0, 6), 0, 255)) for c in LOOK[t][1])
            draw_shape(img, t, cx, cy, col)
    f = img.astype(np.float32) * rng.uniform(0.96, 1.04) * gain
    if light == "shadow":          # hand shadow over the left half
        f[:, :330] *= 0.6
    elif light == "lamp":          # whole room dimmer and warmer
        f *= 0.72
        f[..., 2] *= 1.08
    f += np.random.normal(0, 3, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


class FakeCam:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.content = {t: t for t in TOOLS}
        self.light = "normal"
        self.light_shots = 0       # how many next photos get self.light

    def grab(self):
        light = self.light if self.light_shots > 0 else "normal"
        self.light_shots -= 1
        return render(self.content, self.rng, light), 0.01

    def resolution(self):
        return (640, 480)

    def close(self):
        pass


def content_for(block, truth):
    c = {t: t for t in TOOLS}
    if block == "swap":
        a, b = truth
        c[a], c[b] = b, a
    else:
        for t in truth:
            c[t] = "empty"
    return c


def fake_toolcheck_session(root, sid, started, seed, lighting=False, gain=1.0):
    """Writes data/<sid>/ like toolcheck's `main.py run` (per_trial checkout)."""
    from toolai._toolcheck_copy import slot_scores
    rng = random.Random(seed)
    np.random.seed(seed)
    d = root / "data" / sid
    (d / "images").mkdir(parents=True)
    # calibration: 12 complete-kit captures, threshold = max(floor, mean + 4 std)
    prev = render({t: t for t in TOOLS}, rng, gain=gain)
    samples = []
    for _ in range(12):
        img = render({t: t for t in TOOLS}, rng, gain=gain)
        samples.append(slot_scores(img, prev, SLOTS, 2.0, 10))
        prev = img
    thr = {"A": {t: {"s": max(0.05, float(np.mean([x[t]["s"] for x in samples]) + 4 * np.std([x[t]["s"] for x in samples]))),
                     "c": max(0.1, float(np.mean([x[t]["c"] for x in samples]) + 4 * np.std([x[t]["c"] for x in samples])))}
                 for t in TOOLS},
           "lighting": {"dark": 0.02, "shift": 0.05}}
    pairs = [["Wire stripper", "Wire cutter"], ["Wire cutter", "Nippers"],
             ["Wire stripper", "Nippers"], ["Painter's tape", "Electrical tape"]]
    plan = [("removal", [t]) for t in TOOLS] + [("swap", pairs[i % 4]) for i in range(12)] + \
           [("complete", []) for _ in range(8)]
    trials = []
    for i, (block, truth) in enumerate(plan, start=1):
        instr = {"removal": f"REMOVE: {truth[0] if truth else ''}",
                 "swap": f"SWAP: {truth}", "complete": "KIT STAYS COMPLETE. Perturbation: No change"}[block]
        light = "normal"
        if lighting and block == "complete" and i % 2 == 0:
            light = "shadow" if i % 4 == 0 else "lamp"
            instr = "KIT STAYS COMPLETE. Perturbation: " + (
                "Cast a hand shadow over part of the board" if light == "shadow" else "Move a second lamp")
        cv2.imwrite(str(d / "images" / f"t{i:03d}_checkout.png"), render({t: t for t in TOOLS}, rng, gain=gain))
        c = content_for(block, truth)
        if seed == 0 and i == 30:          # operator forgot to put Pliers back: label says present
            c = dict(c, Pliers="empty")
        if light != "normal":
            cv2.imwrite(str(d / "images" / f"t{i:03d}_signoff_try1.png"), render(c, rng, light, gain))
        cv2.imwrite(str(d / "images" / f"t{i:03d}_signoff.png"), render(c, rng, gain=gain))
        trials.append({"session": sid, "trial": i, "block": block, "instruction": instr,
                       "truth": json.dumps(truth), "void": 0, "void_reason": "", "note": ""})
    with open(d / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trials[0]))
        w.writeheader()
        w.writerows(trials)
    settings = {"features": {"blur_sigma": 2.0, "shift_tolerance_px": 10},
                "retake": {"enabled": True, "max_flagged": 3, "max_retakes": 2, "dark_tol": 0.15,
                           "blur_sigma": 8.0}, "trials": {"seed": 1}}
    # this session's references: empty board (for footprints), stored under refs/ like toolcheck does
    rd = root / "refs" / f"ref-{sid}"
    rd.mkdir(parents=True)
    cv2.imwrite(str(rd / "empty.png"), render({t: "empty" for t in TOOLS}, rng, gain=gain))
    (d / "session.json").write_text(json.dumps({"session": sid, "started": started, "slots": SLOTS,
                                                "refs_dir": f"C:\\Users\\x\\project\\refs\\ref-{sid}",
                                                "thresholds": thr, "settings": settings,
                                                "checkout_mode": "per_trial"}))
    return d, thr, settings


def override_check():
    """Lighting override and retake categories, with the model's answers faked."""
    import toolai.hybrid as hy
    rng = random.Random(1)
    checkout = render({t: t for t in TOOLS}, rng)
    shadow = render({t: t for t in TOOLS}, rng, "shadow")
    thr = {"A": {t: {"s": 0.05, "c": 1.0} for t in TOOLS}, "lighting": {"dark": 0.02, "shift": 0.05}}
    settings = {"features": {"blur_sigma": 2.0, "shift_tolerance_px": 10},
                "retake": {"enabled": True, "max_flagged": 3, "blur_sigma": 8.0, "dark_tol": 0.15}}
    cfg = {"decision": {"min_confidence": 0.6, "clear_confidence": 0.9, "ai_can_clear_a_flags": True},
           "lighting_override": {"enabled": True, "mode": "all_own", "confidence": 0.9},
           "retake": {"skip_many_slot_retake_if_ai_confident": True}}
    orig = hy.classify_slots
    try:
        hy.classify_slots = lambda pred, img, slots: ([(t, 0.97) for t in TOOLS], 0.0)
        r = hy.check_capture(shadow, checkout, SLOTS, thr, settings, None, cfg)
        assert r["lighting_reason"] and r["lighting_override"] == 1 and not r["retake_reason"], r["retake_reason"]
        assert r["flagged"] == [], r["flagged"]
        hy.classify_slots = lambda pred, img, slots: ([("empty" if t == "Pliers" else t, 0.97) for t in TOOLS], 0.0)
        r = hy.check_capture(shadow, checkout, SLOTS, thr, settings, None, cfg)
        assert r["lighting_override"] == 0 and "lighting" in r["retake_category"], r
        assert r["flagged"] == ["Pliers"]
    finally:
        hy.classify_slots = orig
    print("override check OK (all own tools under shadow -> no retake; a missing tool -> retake)")


def board_check():
    """Footprint greys a neighbour inside an overlapping box; exposure evens out a darker photo."""
    from toolai.board import Board
    from toolai.pink import fit_pink
    rng = random.Random(3)
    empty = render({x: "empty" for x in TOOLS}, rng)
    p, _ = fit_pink(empty, SLOTS)
    big = [dict(s) for s in SLOTS]
    big[0]["w"] += 90                                   # slot 0's box now reaches into slot 1
    cfg = {"footprint": {"enabled": True, "grow_px": 8}, "exposure": {"enabled": True, "target": 215}}
    b = Board(big, 0.0, cfg, p, empty)
    full = render({x: x for x in TOOLS}, rng)
    c = b.crops(full)[0]
    assert (c[:, 150:] == 128).all(), "neighbour tool not greyed"
    dark = render({x: x for x in TOOLS}, rng, gain=0.75)
    c1, c2 = b.crops(full)[2], b.crops(dark)[2]
    m1, m2 = np.median(c1[c1 != 128]), np.median(c2[c2 != 128])
    assert abs(m1 - m2) < 12, (m1, m2)
    # tool-only view: the wire cutter looks the same in its own slot and in the stripper's slot
    cfg["tool_only"] = {"enabled": True}
    tb = Board(SLOTS, 0.10, cfg, p, empty)
    i_s, i_c = TOOLS.index("Wire stripper"), TOOLS.index("Wire cutter")
    own = tb.crops(render({x: x for x in TOOLS}, random.Random(7)))[i_c]
    swapped = tb.crops(render(content_for("swap", ["Wire stripper", "Wire cutter"]), random.Random(7)))[i_s]
    gone = tb.crops(render({x: "empty" for x in TOOLS}, random.Random(7)))[i_s]
    d_same = np.abs(own.astype(int) - swapped.astype(int)).mean()
    d_empty = np.abs(own.astype(int) - gone.astype(int)).mean()
    assert d_same < 0.5 * d_empty, (d_same, d_empty)
    assert (gone == 128).mean() > 0.98, (gone == 128).mean()
    # board moved 4 px since the empty-board photo: an empty slot must still look empty
    shifted = np.roll(render({x: "empty" for x in TOOLS}, random.Random(8)), (4, 4), (0, 1))
    tb.crops(shifted)
    assert max(tb.last_content) < 0.01, tb.last_content
    assert tb.ref_mismatch(shifted) < 0.05, tb.ref_mismatch(shifted)
    print(f"tool-only check OK (cutter own slot vs stripper slot differ {d_same:.1f}, vs empty {d_empty:.1f})")
    print(f"board check OK (neighbour greyed; slot median {m1:.0f} vs {m2:.0f} at 75% exposure)")


def unit_checks():
    from toolai.hybrid import slot_verdicts
    from toolai.pink import fit_pink, pink_mask
    d = {"min_confidence": 0.6, "clear_confidence": 0.9, "ai_can_clear_a_flags": True}
    t = ["A", "B", "C", "D", "E"]
    ai = [("A", 0.99), ("B", 0.95), ("C", 0.70), ("empty", 0.97), ("E", 0.40)]
    r = slot_verdicts(t, ["B", "C"], ai, d)
    assert [x["flag"] for x in r] == [False, False, True, True, True], r
    assert r[1]["cleared_by_ai"] and r[3]["label"] == "missing" and r[4]["label"].startswith("unsure"), r
    # pink mask: measured from an empty board, keeps the red handle and the tapes
    rng = random.Random(0)
    empty = render({x: "empty" for x in TOOLS}, rng)
    full = render({x: x for x in TOOLS}, rng)
    p, rep = fit_pink(empty, SLOTS, full)
    assert all(v is None or v < 0.05 for v in rep.values()), rep
    dark = render({x: "empty" for x in TOOLS}, rng, "shadow")
    s = SLOTS[0]
    frac = pink_mask(dark, p)[s["y"]:s["y"] + s["h"], s["x"]:s["x"] + s["w"]].mean()
    frac0 = pink_mask(empty, p)[s["y"]:s["y"] + s["h"], s["x"]:s["x"] + s["w"]].mean()
    assert frac > 0.8 * frac0, (frac, frac0)     # shadowed pink is still found
    print(f"unit checks OK (pink hue {p['hue_deg']} +- {p['hue_tol_deg']}, tool pixels erased max "
          f"{max(v for v in rep.values() if v is not None):.1%}, shadowed pink still masked {frac / frac0:.0%})")


def check_absent_tools():
    """absent_tools: slot expected empty, never planned, flagged if filled."""
    import random
    from toolai.common import EMPTY, expected_contents, load_ai_config, slot_truth
    from toolai.hybrid import score, slot_verdicts
    from toolai.hybrid_session import _plan
    tools, ab = ["A", "B", "S", "C"], ["S"]
    exp = expected_contents(tools, ab)
    d = load_ai_config()["decision"]
    rows = slot_verdicts(tools, [], [("A", .9), ("B", .9), (EMPTY, 1.0), ("C", .9)], d, expected=exp)
    assert not any(r["flag"] for r in rows), rows
    rows = slot_verdicts(tools, ["S"], [("A", .9), ("B", .9), ("C", .8), ("C", .9)], d, expected=exp)
    assert [r["slot"] for r in rows if r["flag"]] == ["S"], rows
    c = slot_truth(tools, ["B"], "removal", ab)
    sc = score(c, {"flagged": ["B"], "contents": {"A": "A", "B": EMPTY, "S": EMPTY, "C": "C"}}, exp)
    assert sc["miss"] == 0 and sc["false_alarm"] == 0, sc
    kit = {"lookalike_pairs": [["A", "B"], ["S", "C"]], "perturbations": ["lift {tool}", "shadow", "lamp"]}
    plan = _plan([t for t in tools if t not in ab], kit, 6, 2, 6, random.Random(0), True)
    assert not any("S" in st["truth"] for st in plan), plan
    comp = [st["instruction"] for st in plan if st["block"] == "complete"]
    assert any("shadow" in x for x in comp[:3]) and any("lamp" in x for x in comp[:3]), comp
    print("absent_tools + perturbation cycling: OK")


def main():
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(SRC / "toolai", tmp / "toolai")
    shutil.copy(SRC / "ai_main.py", tmp / "ai_main.py")
    (tmp / "config").mkdir()
    cfg_txt = json.loads((SRC / "config" / "ai.json").read_text(encoding="utf-8"))
    cfg_txt["absent_tools"] = []      # the synthetic board has all 12 tools; absent logic tested below
    (tmp / "config" / "ai.json").write_text(json.dumps(cfg_txt))
    (tmp / "config" / "slots.json").write_text(json.dumps({"slots": SLOTS}))
    (tmp / "config" / "kit.json").write_text(json.dumps({"tools": TOOLS, "lookalike_pairs": [
        ["Wire stripper", "Wire cutter"], ["Wire cutter", "Nippers"], ["Wire stripper", "Nippers"],
        ["Painter's tape", "Electrical tape"]]}))
    (tmp / "config" / "ai_exclude.json").write_text(json.dumps({"exclude": [
        {"session": "20990102-000000", "trial": 3, "reason": "test exclusion"}]}))
    sys.path.insert(0, str(tmp))
    for m in [m for m in sys.modules if m == "toolai" or m.startswith("toolai.")]:
        del sys.modules[m]

    from toolai import bridge
    from toolai.common import load_ai_config, load_slots
    from toolai.evaluate import evaluate
    check_absent_tools()
    from toolai.hybrid_session import run_hybrid_session
    from toolai.importer import import_trials
    from toolai.infer import Predictor
    from toolai.pink import run_pink
    from toolai.replay import replay
    from toolai.summary import summarise_session
    from toolai.train import train
    print("toolcheck code used:", bridge.SOURCE)
    unit_checks()
    board_check()
    override_check()

    # 1. three fake toolcheck sessions; the third has shadow / lamp trials with retake photos
    dirs = []
    for k, (sid, started) in enumerate((("20990101-000000", "2099-01-01T00:00:00"),
                                        ("20990102-000000", "2099-01-02T00:00:00"),
                                        ("20990103-000000", "2099-01-03T00:00:00"))):
        d, thr, settings = fake_toolcheck_session(tmp, sid, started, k, lighting=(k == 2),
                                                  gain=(0.8, 1.0, 1.1)[k])   # exposure differs by session
        dirs.append(d)
    # refs for `pink`
    rng = random.Random(5)
    rd = tmp / "refs" / "r1"
    rd.mkdir(parents=True)
    cv2.imwrite(str(rd / "empty.png"), render({t: "empty" for t in TOOLS}, rng))
    cv2.imwrite(str(rd / "full.png"), render({t: t for t in TOOLS}, rng))
    (tmp / "refs" / "current.json").write_text(json.dumps({"dir": "r1"}))

    cfg, slots = load_ai_config(), load_slots()
    run_pink(slots, cfg, rd / "empty.png", rd / "full.png")       # pink FIRST (footprints need it)
    names = import_trials([str(d) for d in dirs], TOOLS, cfg)
    assert (tmp / "dataset" / names[0] / "preview.png").exists()
    if cfg["pink"].get("enabled"):
        assert json.loads((tmp / "dataset" / names[0] / "session.json").read_text())["board"]["footprints"]
    man = list(csv.DictReader(open(tmp / "dataset" / names[0] / "manifest.csv")))
    sus = [(r["trial"], r["slot"]) for r in man if r["suspect"] == "1"]
    assert ("30", "Pliers") in sus, sus
    assert (tmp / "dataset" / names[0] / "suspects.png").exists()
    print("suspect check OK:", sus)
    assert names == ["tc-20990101-000000", "tc-20990102-000000", "tc-20990103-000000"], names
    sj = json.loads((tmp / "dataset" / names[1] / "session.json").read_text())
    assert [x["trial"] for x in sj["skipped_trials"]] == [3], sj["skipped_trials"]

    cfg = load_ai_config()
    cfg["epochs"] = 25
    cfg["augment"]["copies"] = 2

    # 2. train on sessions 1-2, hold out session 3 (the one with lighting trials)
    results = {}
    for kind in ("svm", "cnn"):
        mdir = train(slots, cfg, kind, test_sessions=[names[2]])
        meta = json.loads((mdir / "meta.json").read_text())
        if cfg["pink"].get("enabled"):
            assert meta["pink"] and meta["pink"]["source"] == "pink.json", meta["pink"]
        else:
            assert meta["pink"] is None, meta["pink"]
        results[kind] = evaluate(mdir)
    for kind, r in results.items():
        assert r["crop_accuracy"] > 0.85, (kind, r["crop_accuracy"])
        assert "retake_photo" in r["accuracy_by_tag"], r["accuracy_by_tag"]
        assert r["lighting_override_on_retake_photos"]["fired_but_kit_not_complete"] == 0

    # 3. replay the held-out session through the full hybrid
    pred = Predictor(mdir)
    out = replay(dirs[2], pred, cfg)
    rows = list(csv.DictReader(open(out / "trials.csv")))
    tries = [r for r in rows if r["photo"] == "signoff_try1"]
    assert tries, "no retake photos replayed"
    assert all(r["retake_category"] in ("", "lighting", "lighting+many_slots") for r in tries)
    print("replay: retake photos", len(tries), "overrides", sum(int(r["lighting_override"]) for r in tries))

    # 4. live hybrid session with a fake camera and a scripted operator
    cam = FakeCam(42)
    script = {"trial": 0}

    def ask(prompt):
        p = prompt
        if p.lstrip().startswith("Kit complete?"):
            if script.pop("checkout_retry", False):       # same trial, checkout retaken
                return ""
            script["trial"] += 1
            if script["trial"] == 1:                      # previous swap NOT undone (trial 14 mistake)
                cam.content = content_for("swap", ["Wire stripper", "Wire cutter"])
            return ""
        if "CHECKOUT NOT COMPLETE" in p:
            cam.content = {t: t for t in TOOLS}           # operator fixes the kit, retakes checkout
            script["checkout_retry"] = True
            return ""
        if "Enter when done" in p and "Put the kit back" not in p:
            line = p.strip().splitlines()[0]
            if line.startswith("REMOVE"):
                cam.content = content_for("removal", [line.split(":")[1].split("(")[0].strip()])
            elif line.startswith("SWAP"):
                a = line.split("put ")[1].split(" in ")[0]
                b = line.split(" and ")[1].split(" in ")[0]
                cam.content = content_for("swap", [a, b])
            if script["trial"] == 2:                      # shadow on the first sign-off photo only
                cam.light, cam.light_shots = "shadow", 1
            if script["trial"] == 3:                      # shadow that never goes away -> force
                cam.light, cam.light_shots = "shadow", 99
            return ""
        if "RETAKE" in p:
            return "f" if script["trial"] == 3 else ""
        if "Put the kit back" in p:
            cam.content = {t: t for t in TOOLS}
            cam.light_shots = 0
        return ""

    settings["retake"]["max_retakes"] = 5
    cfg["test_mode"]["allow_force_accept"] = True
    cfg["lighting_override"]["enabled"] = False     # so the shadow really asks for retakes here
    sdir = run_hybrid_session(cam, slots, thr, settings, {"lookalike_pairs": [
        ["Wire stripper", "Wire cutter"], ["Painter's tape", "Electrical tape"]]}, pred, cfg,
        n_removal=3, n_swap=2, n_complete=3, ask=ask, conditions={"note": "synthetic"}, seed=3)
    rows = list(csv.DictReader(open(sdir / "trials.csv")))
    assert int(rows[0]["checkout_retakes"]) >= 1, rows[0]
    assert int(rows[1]["retakes"]) >= 1 and "lighting" in rows[1]["retake_category"], rows[1]
    assert rows[2]["forced"] == "1", rows[2]
    assert all(k in rows[0] for k in ("retakes", "retake_reason", "retake_exhausted",
                                      "first_A_flagged", "first_H_flagged"))
    summarise_session(sdir)
    assert "FORCED (test-only force-accept): 1" in (sdir / "summary.md").read_text()

    print("\nOK: import -> pink -> train (svm, cnn) -> evaluate -> replay -> hybrid session "
          "(checkout check, lighting retake, force-accept) -> summary all ran;",
          {k: v["crop_accuracy"] for k, v in results.items()})
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
