"""Hybrid D: Method A says WHICH slots changed, the learned model says WHAT is in
every slot, and the two are combined slot by slot. Used the same way by the live
hybrid session, by replay of stored photos, and (later) by watch-serve.

Per slot (AI = the model's top answer and its confidence):
    AI unsure (conf < min_confidence)               -> FLAG "unsure"       (fail safe)
    AI: another tool / empty                         -> FLAG "wrong tool: X" / "missing"
    AI: own tool, A did not flag                     -> ok
    AI: own tool, A flagged, conf >= clear_confidence -> ok, "cleared by AI" (tool was reseated)
    AI: own tool, A flagged, conf <  clear_confidence -> FLAG "changed"    (fail safe)
    v3: AI own tool, A flagged on COLOUR only (shape score under its frozen limit),
        slot not in a look-alike pair, conf >= colour_only_clear_confidence -> ok, cleared
        (a removal or swap always moved A's shape score well past its limit in session 4;
        look-alike slots keep the strict clear_confidence because their margins were small)
So A can never hide a flag the AI raises, and the AI can only cancel A's flag
when it is sure the slot's own tool is there.

Per photo (retake check, from toolcheck's Nov 17 patch):
    lighting: shadow on the bare board, or room light shifted (limits frozen by calibrate)
    many slots: 3+ slots flagged by A at once
  Lighting override (your trial 27 idea). v3 mode "shape_ok": the light changed,
    A's SHAPE score is under its limit in EVERY slot (only colour moved = light), and
    the AI's top answer is the slot's own tool in every slot with >= confidence
    -> no retake, A's colour flags are cleared. Older modes all_own / all_confident
    need every slot >= confidence and ignore A's shape score.
    shape_ok also fires when 3+ slots are flagged and EVERY flag is colour only, even if
    the retake check saw no light change (trial 29 of hybrid-20261118-232906).
  Many slots, no lighting change, and the AI confidently names every flagged slot
    (your trial 14 note) -> no retake; the named problems are reported.
  Retake reasons are logged by category (lighting / many_slots), never lumped together.
"""
import time

from .bridge import gap_mask, lighting_change, method_a
from .common import EMPTY
from .infer import classify_slots


def _lookalike_slots(ai_cfg):
    return {t for pair in ai_cfg.get("lookalike_pairs", []) for t in pair}


def _shape_ok(a_scores, thr_a, tool):
    sc = (a_scores or {}).get(tool)
    return sc is not None and tool in thr_a and sc["s"] <= thr_a[tool]["s"]


def fmt_conf(c):
    """Confidence for screens and logs, rounded DOWN: 0.596 prints as 0.59, never as 0.60,
    so a printed value at the limit always means the limit was reached."""
    return f"{int(c * 100) / 100:.2f}"


def clear_bar(tool, dcfg, colour_only=False, lookalikes=()):
    """Confidence the AI needs (own tool named) to cancel Method A's flag on this slot.

    PoC version (Nov 20): per tool. decision.clear_confidence_by_tool lists the strict tools
    (e.g. Wire stripper / Wire cutter at 0.90); every other tool uses
    decision.clear_confidence_default, for shape AND colour flags. Without those keys the
    v3 rule is used: clear_confidence for every slot, lowered to colour_only_clear_confidence
    for colour-only flags on slots outside the look-alike pairs."""
    by_tool = dcfg.get("clear_confidence_by_tool")
    if by_tool is not None:
        return float(by_tool.get(tool, dcfg.get("clear_confidence_default", dcfg["clear_confidence"])))
    bar = dcfg["clear_confidence"]
    colour_clear = dcfg.get("colour_only_clear_confidence")
    if colour_only and colour_clear is not None and tool not in lookalikes:
        bar = min(bar, colour_clear)
    return bar


def slot_verdicts(tools, a_flagged, ai, dcfg, a_scores=None, thr_a=None, lookalikes=(), expected=None):
    """expected: slot -> what it should hold (own tool, or EMPTY for an absent tool)."""
    expected = expected or {t: t for t in tools}
    rows = []
    for tool, (content, conf) in zip(tools, ai):
        a = tool in a_flagged
        colour_only = a and _shape_ok(a_scores, thr_a or {}, tool)
        bar = clear_bar(tool, dcfg, colour_only, lookalikes)
        r = {"slot": tool, "ai": content, "conf": conf, "a_flag": a, "flag": False,
             "label": "", "cleared_by_ai": False}
        own = expected.get(tool, tool)
        if conf < dcfg["min_confidence"]:
            r.update(flag=True, label=f"unsure (AI {fmt_conf(conf)}: {content}?)")
        elif content != own:
            r.update(flag=True, label=(f"should be empty (tool absent this session): {content}" if own == EMPTY
                                       else "missing" if content == EMPTY else f"wrong tool: {content}"))
        elif a and not (dcfg.get("ai_can_clear_a_flags", True) and conf >= bar):
            r.update(flag=True, label=f"changed (AI {fmt_conf(conf)}: own tool, under its {bar:.2f} clear bar)")
        elif a:
            r["cleared_by_ai"] = True
        rows.append(r)
    return rows


def lighting_reason(light, limits):
    if not light or not limits:
        return ""
    out = []
    if light["dark"] > limits["dark"]:
        out.append(f"shadow on board ({light['dark']:.0%} of gap darker)")
    if light["shift"] > limits["shift"]:
        out.append(f"room light changed ({light['shift']:.0%})")
    return "; ".join(out)


def check_capture(signoff, checkout, slots, thresholds, settings, pred, ai_cfg, gap=None, expected=None):
    """One hybrid check of one sign-off photo against its checkout photo.
    expected: slot -> what it should hold (default: its own tool)."""
    t0 = time.perf_counter()
    tools = [s["tool"] for s in slots]
    expected = expected or {t: t for t in tools}
    feat = settings.get("features", {})
    rcfg = settings.get("retake", {})
    ra = method_a(signoff, checkout, slots, thresholds["A"], feat.get("blur_sigma", 2.0),
                  feat.get("shift_tolerance_px", 10))
    t1 = time.perf_counter()
    ai, _ = classify_slots(pred, signoff, slots)
    t2 = time.perf_counter()
    scores = ra.get("scores") or {}

    light, lreason = None, ""
    limits = thresholds.get("lighting")
    if limits and rcfg.get("enabled", False):
        gap = gap if gap is not None else gap_mask(signoff.shape, slots)
        light = lighting_change(signoff, checkout, gap, rcfg.get("blur_sigma", 8.0), rcfg.get("dark_tol", 0.15))
        lreason = lighting_reason(light, limits)

    ov = ai_cfg["lighting_override"]
    mode = ov.get("mode", "all_own")
    own_all = all(n == expected[t] and c >= ov["confidence"] for t, (n, c) in zip(tools, ai))
    if mode == "all_confident":
        ai_ok_for_override = all(c >= ov["confidence"] for _, c in ai)
    elif mode == "shape_ok":
        ai_ok_for_override = own_all and all(_shape_ok(scores, thresholds["A"], t) for t in tools)
    else:
        ai_ok_for_override = own_all
    max_flagged = rcfg.get("max_flagged", 3) if rcfg.get("enabled", False) else 0
    many = bool(max_flagged) and len(ra["flagged"]) >= max_flagged
    # shape_ok also treats "many slots flagged, every one on colour only" as a light change
    # (session hybrid-20261118-232906 trial 29: all 12 flagged, retake check saw no light change)
    colour_only_many = (mode == "shape_ok" and many
                        and all(_shape_ok(scores, thresholds["A"], t) for t in ra["flagged"]))
    override = (bool(lreason) or colour_only_many) and ov.get("enabled", False) and ai_ok_for_override

    # Under a shape_ok override every A flag was colour only (the light): the AI decides.
    a_used = [] if (override and mode == "shape_ok") else ra["flagged"]
    rows = slot_verdicts(tools, a_used, ai, ai_cfg["decision"], scores, thresholds["A"],
                         _lookalike_slots(ai_cfg), expected)
    if a_used is not ra["flagged"]:
        for r in rows:
            r["a_flag"] = r["slot"] in ra["flagged"]
            r["cleared_by_ai"] = r["a_flag"] and not r["flag"]

    named = all(r["conf"] >= clear_bar(r["slot"], ai_cfg["decision"], False, _lookalike_slots(ai_cfg))
                for r in rows if r["a_flag"])
    many_named = many and not lreason and named and ai_cfg["retake"].get("skip_many_slot_retake_if_ai_confident", False)

    reasons, cats = [], []
    if lreason and not override:
        reasons.append(lreason)
        cats.append("lighting")
    if many and not many_named and not override:    # under override, the many flags are the light
        reasons.append(f"{len(ra['flagged'])} slots flagged at once")
        cats.append("many_slots")

    flagged = [r["slot"] for r in rows if r["flag"]]
    # MODEL ONLY (for comparison, never used for the verdict): the AI alone on the same
    # photo, no Method A, no retake check. Flags a slot when the AI is unsure or names
    # something other than what the slot should hold.
    m_rows = slot_verdicts(tools, [], ai, ai_cfg["decision"], expected=expected)
    model_only = {"flagged": [r["slot"] for r in m_rows if r["flag"]],
                  "labels": {r["slot"]: r["label"] for r in m_rows if r["flag"]}}
    return {
        "model_only": model_only,
        "A": ra, "ai": ai, "rows": rows,
        "flagged": flagged, "labels": {r["slot"]: r["label"] for r in rows if r["flag"]},
        "contents": {r["slot"]: r["ai"] for r in rows},
        "cleared_by_ai": [r["slot"] for r in rows if r["cleared_by_ai"]],
        "light": light, "lighting_reason": lreason,
        "lighting_override": int(override), "many_slots_named": int(many_named),
        "retake_reason": "; ".join(reasons), "retake_category": "+".join(cats),
        "timing": {"A_s": t1 - t0, "AI_s": t2 - t1, "total_process_s": time.perf_counter() - t0},
    }


def checkout_issues(pred, checkout, slots, ai_cfg, expected=None):
    """Slots of the checkout photo that do not confidently hold what they should."""
    ai, _ = classify_slots(pred, checkout, slots)
    dmin = ai_cfg["decision"]["min_confidence"]
    expected = expected or {s["tool"]: s["tool"] for s in slots}
    return [(s["tool"], n, c) for s, (n, c) in zip(slots, ai) if n != expected[s["tool"]] or c < dmin]


def score_model_only(truth_content, result, expected=None):
    return score(truth_content, {"flagged": result["model_only"]["flagged"],
                                 "contents": result["contents"]}, expected)


def score(truth_content, result, expected=None):
    """Compare a hybrid result with what is really in each slot.
    should flag = slot does not hold what it should (its own tool, or EMPTY if absent)."""
    expected = expected or {t: t for t in truth_content}
    should = {t for t, c in truth_content.items() if c != expected[t]}
    flagged = set(result["flagged"])
    named_ok = all(result["contents"][t] == truth_content[t] for t in should & flagged)
    return {"missed": sorted(should - flagged), "extra": sorted(flagged - should),
            "miss": int(bool(should - flagged)), "false_alarm": int(bool(flagged - should)),
            "named_exactly": int(flagged == should),
            "named_correctly": int(flagged == should and named_ok)}
