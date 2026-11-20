"""Evaluate a frozen model on its held-out sessions, scored the same way as the PoC.

Per slot:   should_flag = what is in the slot is not the slot's own tool.
            miss        = should_flag, but the model says the right tool is there.
            false alarm = the right tool is there, but the model flags it.
            misnamed    = flagged correctly, but the reason is wrong
                          (e.g. says "missing" when a look-alike is in the slot).
Per capture: a removal or swap capture is MISSED if any changed slot is missed;
             a complete capture is a FALSE ALARM if any slot is flagged.

Writes models/<model>/eval-<stamp>/: eval.json, confusion.csv, slot_results.csv
"""
import csv
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np

from .common import DATASET, EMPTY, load_ai_config, read_manifest, row_skip_reason, save_json, upper_bound
from .infer import Predictor


def evaluate(model_dir, sessions=None, use_int8=False):
    """Per-slot flag rule here = the model alone with ai.json's min_confidence
    (unsure -> flag). Method A is not involved; use `replay` for the full hybrid."""
    pred = Predictor(model_dir, use_int8)
    ai_cfg = load_ai_config()
    min_conf = ai_cfg["decision"]["min_confidence"]
    ov = ai_cfg["lighting_override"]
    sessions = sessions or pred.meta["test_sessions"]
    if not sessions:
        raise SystemExit("This model has no held-out session. Collect another session and retrain.")
    overlap = set(sessions) & set(pred.meta["train_sessions"])
    if overlap:
        print(f"WARNING: {sorted(overlap)} were used for training. Results on them are not evidence.")
    classes = pred.classes
    cidx = {c: i for i, c in enumerate(classes)}
    conf_m = np.zeros((len(classes), len(classes)), int)

    out = model_dir / f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    captures = defaultdict(list)
    infer_times = []
    left_out = {}

    for s in sessions:
        rows = []
        for r in read_manifest(DATASET / s):
            why = row_skip_reason(s, r, ai_cfg, training=False)
            if why:
                left_out[why] = left_out.get(why, 0) + 1
            else:
                rows.append(r)
        by_cap = defaultdict(list)
        for r in rows:
            by_cap[int(r["capture"])].append(r)
        for cap, rs in sorted(by_cap.items()):
            crops = [cv2.imread(str(DATASET / s / r["file"])) for r in rs]
            X = pred.prepare(crops)
            t0 = cv2.getTickCount()
            idx, conf = pred.predict(X)
            lim = (pred.meta.get("board") or {}).get("tool_only", {}).get("empty_share", 0.0)
            if lim:     # same "empty by rule" as live checks, from the tool share saved at import
                e = pred.classes.index(EMPTY)
                for k, r in enumerate(rs):
                    if r.get("tool_share") not in (None, "") and float(r["tool_share"]) < lim:
                        idx[k], conf[k] = e, 1.0
            infer_times.append((cv2.getTickCount() - t0) / cv2.getTickFrequency())
            for k, r in enumerate(rs):
                p = classes[int(idx[k])]
                conf_m[cidx[r["label"]], int(idx[k])] += 1
                should = r["label"] != r["slot"]
                c = float(conf[k])
                flagged = p != r["slot"] or c < min_conf
                row = {"session": s, "capture": cap, "block": r["block"], "slot": r["slot"],
                       "truth": r["label"], "predicted": p,
                       "confidence": f"{c:.3f}", "tags": r.get("tags", ""),
                       "correct": int(p == r["label"]),
                       "should_flag": int(should), "flagged": int(flagged),
                       "miss": int(should and not flagged),
                       "false_alarm": int(flagged and not should),
                       "misnamed": int(should and flagged and p != r["label"])}
                rows_out.append(row)
                captures[(s, cap)].append(row)

    with open(out / "slot_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    with open(out / "confusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["truth \\ predicted"] + classes)
        for i, c in enumerate(classes):
            w.writerow([c] + list(conf_m[i]))

    def block_stats(block):
        caps = [v for v in captures.values() if v[0]["block"] == block]
        if block == "complete":
            fails = sum(any(r["false_alarm"] for r in v) for v in caps)
        else:
            fails = sum(any(r["miss"] for r in v) for v in caps)
        return {"captures": len(caps), "failures": fails,
                "upper_bound_95": round(upper_bound(fails, len(caps)), 4) if caps else None}

    # Accuracy by tag (imported sessions): swap crops, retake photos, lighting trials...
    by_tag = defaultdict(list)
    for r in rows_out:
        for t in filter(None, r["tags"].split(";")):
            by_tag[t].append(r["correct"])
    tag_acc = {t: {"crops": len(v), "accuracy": round(sum(v) / len(v), 4)} for t, v in sorted(by_tag.items())}

    # Would the lighting override have fired on the real retake photos, and was it right?
    over = {"photos": 0, "override_fired": 0, "fired_but_kit_not_complete": 0}
    for v in captures.values():
        if "retake_photo" not in v[0]["tags"]:
            continue
        over["photos"] += 1
        if ov.get("mode") == "all_confident":
            fire = all(float(r["confidence"]) >= ov["confidence"] for r in v)
        else:   # all_own; shape_ok is treated the same here (no Method A in evaluate; use replay)
            fire = all(r["predicted"] == r["slot"] and float(r["confidence"]) >= ov["confidence"] for r in v)
        if fire:
            over["override_fired"] += 1
            if any(r["should_flag"] for r in v):
                over["fired_but_kit_not_complete"] += 1

    slot_should = [r for r in rows_out if r["should_flag"]]
    slot_ok = [r for r in rows_out if not r["should_flag"]]
    swap_slots = [r for r in slot_should if r["truth"] != EMPTY]
    result = {
        "model": model_dir.name, "model_file": pred.model_file.name, "type": pred.kind,
        "sessions": sessions, "crops": len(rows_out),
        "crop_accuracy": round(float(np.trace(conf_m) / conf_m.sum()), 4),
        "slots": {  # flag = model alone (unsure counts as a flag)
            "should_flag": len(slot_should),
            "missed": sum(r["miss"] for r in slot_should),
            "misnamed": sum(r["misnamed"] for r in slot_should),
            "look_alike_slots": len(swap_slots),
            "look_alike_missed": sum(r["miss"] for r in swap_slots),
            "untouched": len(slot_ok),
            "false_alarms": sum(r["false_alarm"] for r in slot_ok)},
        "captures": {b: block_stats(b) for b in ("removal", "swap", "complete")},
        "accuracy_by_tag": tag_acc, "left_out_of_evaluation": left_out, "lighting_override_on_retake_photos": over,
        "decision_used": {"min_confidence": min_conf, "lighting_override": ov},
        "inference_s_per_capture": {
            "median": round(float(np.median(infer_times)), 4),
            "max": round(float(np.max(infer_times)), 4)},
        "note": "Inference time here is on the machine running the evaluation; "
                "measure the processor with `profile`."}
    save_json(out / "eval.json", result)

    print(f"\nEvaluation of {model_dir.name} ({pred.model_file.name}) on {sessions}")
    print(f"  crop accuracy {result['crop_accuracy']:.3f} over {len(rows_out)} crops")
    sl = result["slots"]
    print(f"  slots that should flag: {sl['should_flag']}, missed {sl['missed']}, "
          f"misnamed {sl['misnamed']} (look-alike slots {sl['look_alike_slots']}, "
          f"missed {sl['look_alike_missed']})")
    print(f"  untouched slots: {sl['untouched']}, false alarms {sl['false_alarms']}")
    for b, st in result["captures"].items():
        if st["captures"]:
            what = "with a false alarm" if b == "complete" else "missed"
            print(f"  {b:8s}: {st['failures']}/{st['captures']} {what}, "
                  f"95% upper bound {st['upper_bound_95']:.3f}")
    if left_out:
        print(f"  left out (label doubtful, reported, not scored): {left_out}")
    for t, v in tag_acc.items():
        print(f"  tag {t:15s}: accuracy {v['accuracy']:.3f} over {v['crops']} crops")
    if over["photos"]:
        print(f"  lighting override on {over['photos']} real retake photos: fired {over['override_fired']}, "
              f"fired when the kit was NOT complete {over['fired_but_kit_not_complete']} (must be 0)")
    print(f"  inference per capture: median {result['inference_s_per_capture']['median']*1000:.1f} ms")
    print(f"Saved {out}")
    return result
