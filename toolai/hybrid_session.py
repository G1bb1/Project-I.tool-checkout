"""Live bench session for the hybrid (Method A + model), same procedure as
`main.py run` so the results compare directly with sessions 1-3.

New compared with toolcheck's runner:
  * Checkout check: the model looks at every checkout photo first. If a slot does
    not hold its own tool (a previous swap not undone, as in session 3 trials 14
    and 16), you are told which slot and asked to fix it and retake the checkout.
  * Retake reasons are logged by category: lighting / many_slots, plus whether
    the lighting override or the "many slots, all named" rule cancelled a retake.
  * The five retake columns the Nov 17 patch asked for are always written:
    retakes, retake_reason, retake_exhausted, first_A_flagged, first_H_flagged.
  * TEST ONLY: with test_mode.allow_force_accept (or --allow-force), typing f at a
    retake prompt accepts the current photo. The trial is marked forced=1 and the
    summary reports forced trials separately.

Output: data/hybrid-<stamp>/ (session.json, trials.csv, slot_results.csv, images/)
Then: python ai_main.py hybrid-summary hybrid-<stamp>
"""
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import cv2

from .bridge import SOURCE, gap_mask
from .common import ROOT, expected_contents, save_json, slot_truth
from .hybrid import check_capture, checkout_issues, score, score_model_only

DATA = ROOT / "data"

TRIAL_FIELDS = [
    "session", "trial", "block", "timestamp", "instruction", "truth", "capture_s",
    "checkout_retakes", "checkout_issue_ignored",
    "A_flagged", "A_miss", "A_false_alarm",
    "AI_contents", "H_flagged", "H_labels", "H_cleared_by_ai",
    "H_miss", "H_false_alarm", "H_named_exactly", "H_named_correctly",
    "M_flagged", "M_miss", "M_false_alarm", "M_named_correctly",
    "A_process_s", "AI_process_s", "H_total_s",
    "retakes", "retake_reason", "retake_category", "retake_exhausted",
    "first_A_flagged", "first_H_flagged",
    "lighting_reason_final", "lighting_override", "many_slots_named", "forced",
    "void", "void_reason", "note",
]
SLOT_FIELDS = ["trial", "slot", "truth", "should_flag", "A_s", "A_c", "A_flag",
               "AI_content", "AI_conf", "H_flag", "H_label", "cleared_by_ai"]


def _plan(tools, kit, n_removal, n_swap, n_complete, rng, cycle_perturbations=False):
    plan = _base_plan(tools, kit, n_removal, n_swap, n_complete, rng)
    perts = list(kit.get("perturbations") or [])
    if cycle_perturbations and perts:
        # Every perturbation (incl. "Cast a hand shadow...") is used before any repeats,
        # instead of a random pick per trial (10 random picks from 6 skip one about half the time).
        rng.shuffle(perts)
        k = 0
        for step in plan:
            if step["block"] == "complete":
                p = perts[k % len(perts)].format(tool=rng.choice(tools))
                step["instruction"] = f"KIT STAYS COMPLETE. Perturbation: {p}"
                k += 1
    return plan


def _base_plan(tools, kit, n_removal, n_swap, n_complete, rng):
    try:
        from toolcheck.trials import build_plan          # identical plan to `main.py run`
        return build_plan(tools, kit, n_removal, n_swap, n_complete, rng)
    except ImportError:
        plan = [{"block": "removal", "truth": [t], "instruction": f"REMOVE: {t}"}
                for t in (tools * (n_removal // len(tools) + 1))[:n_removal]]
        pairs = [p for p in kit.get("lookalike_pairs", []) if p[0] in tools and p[1] in tools]
        plan += [{"block": "swap", "truth": list(pairs[i % len(pairs)]),
                  "instruction": "SWAP: put {0} in {1}'s slot and {1} in {0}'s slot".format(*pairs[i % len(pairs)])}
                 for i in range(n_swap if pairs else 0)]
        plan += [{"block": "complete", "truth": [], "instruction": "KIT STAYS COMPLETE. No change"}
                 for _ in range(n_complete)]
        return plan


def annotate(img, slots, res, title):
    out = img.copy()
    for s, r in zip(slots, res["rows"]):
        col = (0, 0, 255) if r["flag"] else ((255, 160, 0) if r["cleared_by_ai"] else (0, 200, 0))
        cv2.rectangle(out, (s["x"], s["y"]), (s["x"] + s["w"], s["y"] + s["h"]), col, 2)
        cv2.putText(out, f"{r['ai'][:14]} {r['conf']:.2f}", (s["x"] + 3, s["y"] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return out


def run_hybrid_session(cam, slots, thresholds, settings, kit, pred, ai_cfg,
                       n_removal, n_swap, n_complete, allow_force=False, ask=input,
                       conditions=None, seed=None):
    tools = [s["tool"] for s in slots]
    absent = list(ai_cfg.get("absent_tools") or [])
    unknown = [t for t in absent if t not in tools]
    if unknown:
        raise SystemExit(f"ai.json absent_tools has names that are not slots: {unknown}")
    expected = expected_contents(tools, absent)
    plan_tools = [t for t in tools if t not in absent]
    allow_force = allow_force or ai_cfg["test_mode"].get("allow_force_accept", False)
    rcfg = settings.get("retake", {})
    max_retakes = ai_cfg["retake"].get("max_retakes")
    if max_retakes is None:
        max_retakes = rcfg.get("max_retakes", 2)
    seed = seed if seed is not None else settings.get("trials", {}).get("seed") or int(time.time())
    plan = _plan(plan_tools, kit, n_removal, n_swap, n_complete, random.Random(seed),
                 ai_cfg.get("plan", {}).get("cycle_perturbations", False))

    sid = "hybrid-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    sdir = DATA / sid
    (sdir / "images").mkdir(parents=True, exist_ok=True)
    if conditions is None:
        print("\nTest conditions (saved in session.json):")
        conditions = {k: ask(f"  {q}: ").strip() for k, q in (
            ("lighting", "Lighting"), ("operator", "Operator"), ("logbook_entry", "Logbook entry"))}
    save_json(sdir / "session.json", {
        "session": sid, "started": datetime.now().isoformat(timespec="seconds"), "seed": seed,
        "resolution": list(cam.resolution()), "camera_locked": getattr(cam, "locked", {}),
        "conditions": conditions,
        "counts": {"removal": n_removal, "swap": n_swap, "complete": n_complete},
        "model": pred.dir.name, "model_file": pred.model_file.name,
        "model_train_sessions": pred.meta.get("train_sessions"),
        "absent_tools": absent, "ai_config": ai_cfg, "max_retakes": max_retakes, "force_accept_allowed": allow_force,
        "toolcheck_code_used": SOURCE, "thresholds": thresholds, "settings": settings,
        "slots": slots, "plan": plan})

    tf = open(sdir / "trials.csv", "w", newline="", encoding="utf-8")
    sf = open(sdir / "slot_results.csv", "w", newline="", encoding="utf-8")
    tw, sw = csv.DictWriter(tf, fieldnames=TRIAL_FIELDS), csv.DictWriter(sf, fieldnames=SLOT_FIELDS)
    tw.writeheader()
    sw.writeheader()
    gap = None
    print(f"\nHybrid session {sid}: {len(plan)} trials, model {pred.dir.name}. Data -> {sdir}")
    if absent:
        print(f"ABSENT this session (slot must stay EMPTY, never used in a trial): {', '.join(absent)}")
    if allow_force:
        print("TEST MODE: 'f' at a retake prompt force-accepts the photo (logged as forced).")
    try:
        for i, step in enumerate(plan, start=1):
            print(f"--- Trial {i}/{len(plan)}  [{step['block']}]")
            # 1. checkout photo, checked by the model
            co_retakes, co_ignored = 0, ""
            while True:
                if ask("  Kit complete? Enter = take checkout photo (q = stop): ").strip().lower() == "q":
                    raise KeyboardInterrupt
                checkout, _ = cam.grab()
                if not ai_cfg["checkout_check"].get("enabled", True):
                    break
                issues = checkout_issues(pred, checkout, slots, ai_cfg, expected)
                if not issues:
                    break
                txt = "; ".join(f"{t} slot: sees {n} ({c:.2f})" for t, n, c in issues)
                a = ask(f"  CHECKOUT NOT COMPLETE: {txt}\n"
                        "  Fix the kit, Enter = retake checkout | c = continue anyway: ").strip().lower()
                if a == "c":
                    co_ignored = txt
                    break
                co_retakes += 1
            cv2.imwrite(str(sdir / "images" / f"t{i:03d}_checkout.png"), checkout)
            if gap is None:
                gap = gap_mask(checkout.shape, slots)
                mm = pred.board.ref_mismatch(checkout) if pred.board is not None else None
                lim = ai_cfg["board"].get("tool_only", {}).get("max_ref_mismatch", 0.05)
                if mm is not None:
                    print(f"  Empty-board reference check: {mm:.1%} of the bare paper no longer matches "
                          f"(limit {lim:.0%}).")
                    if mm > lim and ask("  WARNING: the empty-board photo doesn't match the board now (moved, or "
                                        "different light/exposure). Stop and run `python main.py capture-refs` "
                                        "first? y = stop | Enter = continue anyway: ").strip().lower() == "y":
                        raise KeyboardInterrupt

            # 2. the change, then the sign-off photo with the retake loop
            if ask(f"  {step['instruction']}\n  Enter when done: ").strip().lower() == "q":
                break
            retakes, forced, first = 0, 0, None
            while True:
                signoff, cap_s = cam.grab()
                res = check_capture(signoff, checkout, slots, thresholds, settings, pred, ai_cfg, gap, expected)
                if not res["retake_reason"]:
                    break
                if first is None:
                    first = res
                    cv2.imwrite(str(sdir / "images" / f"t{i:03d}_signoff_try1.png"), signoff)
                if retakes >= max_retakes:
                    break
                opts = "Enter = retake" + (" | f = FORCE-ACCEPT this photo (test only)" if allow_force else "")
                a = ask(f"  RETAKE [{res['retake_category']}]: {res['retake_reason']}\n"
                        f"  Fix the light / kit, {opts}: ").strip().lower()
                if a == "f" and allow_force:
                    forced = 1
                    break
                retakes += 1
            exhausted = int(bool(res["retake_reason"]) and not forced and retakes >= max_retakes)

            content = slot_truth(tools, step["truth"], step["block"], absent)
            sh = score(content, res, expected)
            sm = score_model_only(content, res, expected)
            a_should = set(step["truth"])
            a_fl = set(res["A"]["flagged"])
            ann = annotate(signoff, slots, res, f"hybrid: {'OK' if not res['flagged'] else 'FLAG'}")
            cv2.imwrite(str(sdir / "images" / f"t{i:03d}_signoff.png"), signoff)
            cv2.imwrite(str(sdir / "images" / f"t{i:03d}_annotated.jpg"), ann)
            if settings.get("trials", {}).get("show_window"):
                cv2.imshow("hybrid", cv2.resize(ann, None, fx=0.7, fy=0.7))
                cv2.waitKey(1)

            verdict = ("MISSED " + str(sh["missed"])) if sh["miss"] else ("caught" if step["truth"] else "ok")
            print(f"  hybrid: {verdict}" + (f"  +false alarm {sh['extra']}" if sh["extra"] else "")
                  + (f"   flags: {res['labels']}" if res["labels"] else "")
                  + (f"   (AI cleared A's flag on {res['cleared_by_ai']})" if res["cleared_by_ai"] else "")
                  + ("   [lighting override]" if res["lighting_override"] else "")
                  + ("   [FORCED]" if forced else ""))
            ans = ask("  Enter = accept | v = void (procedural error) | n = add note: ").strip().lower()
            void, why, note = 0, "", ""
            if ans == "v":
                void, why = 1, ask("  Void reason: ").strip()
            elif ans == "n":
                note = ask("  Note: ").strip()

            tw.writerow({
                "session": sid, "trial": i, "block": step["block"],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "instruction": step["instruction"], "truth": json.dumps(step["truth"]),
                "capture_s": f"{cap_s:.4f}", "checkout_retakes": co_retakes,
                "checkout_issue_ignored": co_ignored,
                "A_flagged": json.dumps(res["A"]["flagged"]),
                "A_miss": int(bool(a_should - a_fl)), "A_false_alarm": int(bool(a_fl - a_should)),
                "AI_contents": json.dumps({t: f"{n}@{c:.2f}" for t, (n, c) in zip(tools, res["ai"])}),
                "H_flagged": json.dumps(res["flagged"]), "H_labels": json.dumps(res["labels"]),
                "H_cleared_by_ai": json.dumps(res["cleared_by_ai"]),
                "H_miss": sh["miss"], "H_false_alarm": sh["false_alarm"],
                "H_named_exactly": sh["named_exactly"], "H_named_correctly": sh["named_correctly"],
                "M_flagged": json.dumps(res["model_only"]["flagged"]), "M_miss": sm["miss"],
                "M_false_alarm": sm["false_alarm"], "M_named_correctly": sm["named_correctly"],
                "A_process_s": f"{res['timing']['A_s']:.4f}", "AI_process_s": f"{res['timing']['AI_s']:.4f}",
                "H_total_s": f"{cap_s + res['timing']['total_process_s']:.4f}",
                "retakes": retakes,
                "retake_reason": first["retake_reason"] if first else "",
                "retake_category": first["retake_category"] if first else "",
                "retake_exhausted": exhausted,
                "first_A_flagged": json.dumps(first["A"]["flagged"]) if first else "",
                "first_H_flagged": json.dumps(first["flagged"]) if first else "",
                "lighting_reason_final": res["lighting_reason"],
                "lighting_override": res["lighting_override"],
                "many_slots_named": res["many_slots_named"], "forced": forced,
                "void": void, "void_reason": why, "note": note})
            for r in res["rows"]:
                sc = res["A"]["scores"][r["slot"]]
                sw.writerow({"trial": i, "slot": r["slot"], "truth": content[r["slot"]],
                             "should_flag": int(content[r["slot"]] != expected[r["slot"]]),
                             "A_s": f"{sc['s']:.4f}", "A_c": f"{sc['c']:.3f}", "A_flag": int(r["a_flag"]),
                             "AI_content": r["ai"], "AI_conf": f"{r['conf']:.3f}", "H_flag": int(r["flag"]),
                             "H_label": r["label"], "cleared_by_ai": int(r["cleared_by_ai"])})
            tf.flush()
            sf.flush()
            if step["truth"]:
                ask("  Put the kit back to complete. Enter when done: ")
    except KeyboardInterrupt:
        print("  stopped")
    finally:
        tf.close()
        sf.close()
        if settings.get("trials", {}).get("show_window"):
            cv2.destroyAllWindows()
    print(f"\nSession saved: {sdir}\nNext: python ai_main.py hybrid-summary {sid}")
    return sdir
