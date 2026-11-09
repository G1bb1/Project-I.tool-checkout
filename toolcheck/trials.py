"""Bench session runner. Every trial is scored by BOTH methods on the same capture.

Blocks
  removal   one tool removed (balanced: each tool removed about equally often)
  swap      a look-alike pair swapped into each other's slots
  complete  nothing missing, a perturbation applied (false-alarm test)

Ground truth comes from the script's own instruction, so it cannot drift from
what is logged. A trial can be voided only for a procedural error, with a reason;
voided rows stay in the CSV and are counted in the summary.
"""
import csv
import json
import random
import time
from datetime import datetime

import cv2

from .methods import method_a, method_b, outcome
from .notify import send_result
from .paths import DATA, save_json

TRIAL_FIELDS = [
    "session", "trial", "block", "timestamp", "instruction", "truth",
    "capture_s",
    "A_flagged", "A_labels", "A_miss", "A_false_alarm", "A_named_exactly",
    "A_process_s", "A_total_s",
    "B_flagged", "B_labels", "B_miss", "B_false_alarm", "B_named_exactly",
    "B_process_s", "B_total_s",
    "void", "void_reason", "note",
]


def build_plan(tools, kit, n_removal, n_swap, n_complete, rng):
    plan = []
    pool = []
    for _ in range(n_removal):
        if not pool:
            pool = tools[:]
            rng.shuffle(pool)
        t = pool.pop()
        plan.append({"block": "removal", "truth": [t],
                     "instruction": f"REMOVE: {t}  (take it off the board entirely)"})
    pairs = [p for p in kit["lookalike_pairs"] if p[0] in tools and p[1] in tools]
    for i in range(n_swap):
        a, b = pairs[i % len(pairs)] if pairs else (None, None)
        if a is None:
            break
        plan.append({"block": "swap", "truth": [a, b],
                     "instruction": f"SWAP: put {a} in {b}'s slot and {b} in {a}'s slot"})
    for _ in range(n_complete):
        p = rng.choice(kit["perturbations"]).format(tool=rng.choice(tools))
        plan.append({"block": "complete", "truth": [],
                     "instruction": f"KIT STAYS COMPLETE. Perturbation: {p}"})
    return plan


def _annotate(img, slots, result, title):
    out = img.copy()
    for s in slots:
        hit = s["tool"] in result["flagged"]
        colour = (0, 0, 255) if hit else (0, 200, 0)
        cv2.rectangle(out, (s["x"], s["y"]), (s["x"] + s["w"], s["y"] + s["h"]), colour, 2)
        label = s["tool"][:14] + (f" [{result['labels'][s['tool']]}]" if hit else "")
        cv2.putText(out, label, (s["x"] + 3, s["y"] + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, colour, 1, cv2.LINE_AA)
    cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    return out


def ask_conditions():
    print("\nRecord the test conditions (these go into session.json for your logbook).")
    q = [("camera", "Camera model"), ("height_cm", "Camera height above board (cm)"),
         ("angle_deg", "Camera tilt from straight down (deg)"),
         ("lighting", "Lighting (e.g. ring light 100%, ceiling off)"),
         ("background", "Background (e.g. white paper, traced outlines)"),
         ("operator", "Who is running the trials"), ("logbook_entry", "Logbook entry number (e.g. L-12)")]
    return {k: input(f"  {label}: ").strip() for k, label in q}


def run_session(cam, slots, refs, thresholds, settings, kit, n_removal, n_swap, n_complete):
    tcfg = settings["trials"]
    blur = settings["features"]["blur_sigma"]
    shift = settings["features"].get("shift_tolerance_px", 10)
    seed = tcfg.get("seed")
    if seed is None:
        seed = int(time.time())
    rng = random.Random(seed)
    tools = [s["tool"] for s in slots]
    plan = build_plan(tools, kit, n_removal, n_swap, n_complete, rng)

    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    sdir = DATA / session_id
    (sdir / "images").mkdir(parents=True, exist_ok=True)
    conditions = ask_conditions()
    w, h = cam.resolution()
    save_json(sdir / "session.json", {
        "session": session_id, "started": datetime.now().isoformat(timespec="seconds"),
        "seed": seed, "resolution": [w, h], "conditions": conditions,
        "counts": {"removal": n_removal, "swap": n_swap, "complete": n_complete},
        "checkout_mode": tcfg["checkout_mode"], "refs_dir": str(refs["dir"]),
        "thresholds": thresholds, "settings": settings, "slots": slots, "plan": plan,
    })

    trials_f = open(sdir / "trials.csv", "w", newline="", encoding="utf-8")
    scores_f = open(sdir / "slot_scores.csv", "w", newline="", encoding="utf-8")
    tw = csv.DictWriter(trials_f, fieldnames=TRIAL_FIELDS)
    tw.writeheader()
    sw = csv.writer(scores_f)
    sw.writerow(["trial", "tool", "should_flag", "A_s", "A_c", "A_flag", "B_s", "B_c", "B_flag"])

    print(f"\nSession {session_id}: {len(plan)} trials. Data -> {sdir}")
    print("At each prompt press Enter to continue, or type q then Enter to stop early.\n")

    checkout = None
    if tcfg["checkout_mode"] == "session":
        input("Kit complete. Enter to take the session checkout photo (Method A reference)...")
        checkout, _ = cam.grab()
        cv2.imwrite(str(sdir / "images" / "checkout_session.png"), checkout)

    try:
        for i, step in enumerate(plan, start=1):
            print(f"--- Trial {i}/{len(plan)}  [{step['block']}]")
            if tcfg["checkout_mode"] == "per_trial":
                if input("  Kit complete? Enter = take checkout photo: ").strip().lower() == "q":
                    break
                checkout, _ = cam.grab()
            if input(f"  {step['instruction']}\n  Enter when done: ").strip().lower() == "q":
                break

            signoff, cap_s = cam.grab()
            ra = method_a(signoff, checkout, slots, thresholds["A"], blur, shift)
            rb = method_b(signoff, refs["full"], refs["empty"], slots, thresholds["B"], blur, shift)
            oa, ob = outcome(step["truth"], ra["flagged"]), outcome(step["truth"], rb["flagged"])

            send_result(settings["notify"], "missing" if rb["flagged"] else "complete", rb["flagged"])

            ann = cv2.hconcat([_annotate(signoff, slots, ra, "A before/after"),
                               _annotate(signoff, slots, rb, "B shadow board")])
            cv2.imwrite(str(sdir / "images" / f"t{i:03d}_signoff.png"), signoff)
            cv2.imwrite(str(sdir / "images" / f"t{i:03d}_annotated.jpg"), ann)
            if tcfg["checkout_mode"] == "per_trial":
                cv2.imwrite(str(sdir / "images" / f"t{i:03d}_checkout.png"), checkout)
            if tcfg.get("show_window"):
                scale = 1400 / ann.shape[1]
                cv2.imshow("toolcheck", cv2.resize(ann, None, fx=scale, fy=scale))
                cv2.waitKey(1)

            def verdict(o):
                if o["miss"]:
                    return f"MISSED {o['missed']}"
                return "caught" if step["truth"] else "ok"
            print(f"  A: {verdict(oa)}{'  +false alarm ' + str(oa['extra']) if oa['extra'] else ''}"
                  f"   ({cap_s + ra['process_s']:.2f} s)")
            print(f"  B: {verdict(ob)}{'  +false alarm ' + str(ob['extra']) if ob['extra'] else ''}"
                  f"   ({cap_s + rb['process_s']:.2f} s)")

            ans = input("  Enter = accept | v = void (procedural error) | n = add note: ").strip().lower()
            void, reason, note = 0, "", ""
            if ans == "v":
                void, reason = 1, input("  Void reason (what went wrong in the procedure): ").strip()
            elif ans == "n":
                note = input("  Note: ").strip()

            tw.writerow({
                "session": session_id, "trial": i, "block": step["block"],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "instruction": step["instruction"], "truth": json.dumps(step["truth"]),
                "capture_s": f"{cap_s:.4f}",
                "A_flagged": json.dumps(ra["flagged"]), "A_labels": json.dumps(ra["labels"]),
                "A_miss": oa["miss"], "A_false_alarm": oa["false_alarm"],
                "A_named_exactly": oa["named_exactly"], "A_process_s": f"{ra['process_s']:.4f}",
                "A_total_s": f"{cap_s + ra['process_s']:.4f}",
                "B_flagged": json.dumps(rb["flagged"]), "B_labels": json.dumps(rb["labels"]),
                "B_miss": ob["miss"], "B_false_alarm": ob["false_alarm"],
                "B_named_exactly": ob["named_exactly"], "B_process_s": f"{rb['process_s']:.4f}",
                "B_total_s": f"{cap_s + rb['process_s']:.4f}",
                "void": void, "void_reason": reason, "note": note,
            })
            for t in tools:
                sw.writerow([i, t, int(t in step["truth"]),
                             f"{ra['scores'][t]['s']:.4f}", f"{ra['scores'][t]['c']:.3f}",
                             int(t in ra["flagged"]),
                             f"{rb['scores'][t]['s']:.4f}", f"{rb['scores'][t]['c']:.3f}",
                             int(t in rb["flagged"])])
            trials_f.flush()
            scores_f.flush()

            if step["truth"]:
                input("  Put the kit back to complete. Enter when done: ")
    finally:
        trials_f.close()
        scores_f.close()
        cv2.destroyAllWindows()
    print(f"\nSession saved: {sdir}\nNext: python main.py analyze {session_id}")
    return sdir
