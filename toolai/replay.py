"""Replay the hybrid on photos you ALREADY took in a toolcheck session (no camera).

For every trial it loads the stored checkout and sign-off photos, and the first
(retake) photo where one exists, and runs Method A with that session's own frozen
thresholds plus the model. That gives A and the hybrid side by side on exactly
the same captures, and shows what the lighting override would have done on the
real shadow / lamp photos (trials 26, 28, 32 in session 3).

Evidence rule: a session the model was trained on is not evidence. replay marks
every row in_training=1 in that case and the summary says so at the top.

Output: <toolcheck session>/replay-<model>/ trials.csv + summary.md
Also useful on the Pi: it measures the model's time per photo without a camera.
"""
import csv
import json
from pathlib import Path

import cv2

from .bridge import SOURCE
from .common import load_json, save_json, slot_truth
from .hybrid import check_capture, checkout_issues, score, score_model_only
from .importer import load_exclusions
from .summary import write_summary

FIELDS = ["session", "trial", "block", "photo", "truth", "void", "excluded", "in_training",
          "checkout_issues", "A_flagged", "A_miss", "A_false_alarm", "AI_contents",
          "H_flagged", "H_labels", "H_cleared_by_ai", "H_miss", "H_false_alarm",
          "H_named_exactly", "H_named_correctly",
          "M_flagged", "M_miss", "M_false_alarm", "M_named_correctly", "retake_reason", "retake_category",
          "lighting_reason_final", "lighting_override", "many_slots_named", "forced",
          "A_process_s", "AI_process_s"]


def replay(tc_dir, pred, ai_cfg):
    tc_dir = Path(tc_dir)
    sj = load_json(tc_dir / "session.json")
    sid, slots, thr, settings = sj["session"], sj["slots"], sj["thresholds"], sj["settings"]
    tools = [s["tool"] for s in slots]
    if sorted(tools) != sorted(t for t in pred.classes if t != "empty"):
        raise SystemExit(f"{sid}: its tools don't match the model's classes.")
    if "retake" not in settings and "lighting" in thr:
        settings["retake"] = {"enabled": True, "max_flagged": 3, "blur_sigma": 8.0, "dark_tol": 0.15}
    from .board import refs_empty_for
    pred.use_board(slots, refs_empty_for(sj.get("refs_dir")), f"{sid}: refs {sj.get('refs_dir')}")
    trained = f"tc-{sid}" in (pred.meta.get("train_sessions") or [])
    excl = load_exclusions()
    out = tc_dir / f"replay-{pred.dir.name}"
    out.mkdir(exist_ok=True)
    rows = []
    with open(tc_dir / "trials.csv", newline="", encoding="utf-8") as f:
        trials = list(csv.DictReader(f))
    for tr in trials:
        n = int(tr["trial"])
        def img(name):
            p = tc_dir / "images" / name
            return cv2.imread(str(p)) if p.exists() else None
        checkout = img(f"t{n:03d}_checkout.png")
        if checkout is None:
            print(f"  trial {n}: no checkout photo, skipped")
            continue
        truth = json.loads(tr["truth"])
        content = slot_truth(tools, truth, tr["block"])
        issues = checkout_issues(pred, checkout, slots, ai_cfg)
        for photo in ("signoff", "signoff_try1"):
            signoff = img(f"t{n:03d}_{photo}.png")
            if signoff is None:
                continue
            res = check_capture(signoff, checkout, slots, thr, settings, pred, ai_cfg)
            sh = score(content, res)
            sm = score_model_only(content, res)
            a_should, a_fl = set(truth), set(res["A"]["flagged"])
            rows.append({
                "session": sid, "trial": n, "block": tr["block"], "photo": photo,
                "truth": json.dumps(truth), "void": tr.get("void", "0"),
                "excluded": int((sid, n) in excl), "in_training": int(trained),
                "checkout_issues": "; ".join(f"{t}: {c} ({p:.2f})" for t, c, p in issues),
                "A_flagged": json.dumps(res["A"]["flagged"]),
                "A_miss": int(bool(a_should - a_fl)), "A_false_alarm": int(bool(a_fl - a_should)),
                "AI_contents": json.dumps({t: f"{c}@{p:.2f}" for t, (c, p) in zip(tools, res["ai"])}),
                "H_flagged": json.dumps(res["flagged"]), "H_labels": json.dumps(res["labels"]),
                "H_cleared_by_ai": json.dumps(res["cleared_by_ai"]),
                "H_miss": sh["miss"], "H_false_alarm": sh["false_alarm"],
                "H_named_exactly": sh["named_exactly"], "H_named_correctly": sh["named_correctly"],
                "M_flagged": json.dumps(res["model_only"]["flagged"]), "M_miss": sm["miss"],
                "M_false_alarm": sm["false_alarm"], "M_named_correctly": sm["named_correctly"],
                "retake_reason": res["retake_reason"], "retake_category": res["retake_category"],
                "lighting_reason_final": res["lighting_reason"],
                "lighting_override": res["lighting_override"], "many_slots_named": res["many_slots_named"],
                "forced": 0,
                "A_process_s": f"{res['timing']['A_s']:.4f}", "AI_process_s": f"{res['timing']['AI_s']:.4f}"})
        last = [r for r in rows if r["trial"] == n]
        print(f"  trial {n:3d} {tr['block']:8s} " + " | ".join(
            f"{r['photo']}: A {json.loads(r['A_flagged'])} -> H {json.loads(r['H_labels']) or 'ok'}"
            + (" [override]" if r["lighting_override"] else "")
            + (f" [retake: {r['retake_category']}]" if r["retake_category"] else "") for r in last))

    with open(out / "trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    save_json(out / "replay.json", {"toolcheck_session": sid, "model": pred.dir.name,
                                    "model_file": pred.model_file.name, "in_training": trained,
                                    "toolcheck_code_used": SOURCE, "ai_config": ai_cfg})
    title = f"Replay of {sid} with model {pred.dir.name}"
    write_summary(out, rows, title, replay=True,
                  warning="THIS SESSION WAS USED TO TRAIN THE MODEL: these numbers are NOT evidence."
                  if trained else "")
    return out
