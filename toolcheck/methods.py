"""The two concepts under comparison. Both check the same slots; they differ only
in what each slot is compared against.

Method A (before/after): each slot vs the CHECKOUT photo of the same job.
Method B (known shadow board): each slot vs references stored ONCE at setup
          (board full + board empty). The empty reference only labels a flagged
          slot as "missing" or "wrong tool"; it does not decide the flag.

A slot is flagged when either distance exceeds that slot's frozen threshold.
"""
import time

from .features import slot_scores, crop, structure_distance


def flag(scores, thr):
    return [t for t, sc in scores.items()
            if sc["s"] > thr[t]["s"] or sc["c"] > thr[t]["c"]]


def method_a(signoff, checkout, slots, thr_a, blur, shift=10):
    t0 = time.perf_counter()
    scores = slot_scores(signoff, checkout, slots, blur, shift)
    flagged = flag(scores, thr_a)
    labels = {t: "changed" for t in flagged}
    return {"flagged": flagged, "labels": labels, "scores": scores,
            "process_s": time.perf_counter() - t0}


def method_b(signoff, ref_full, ref_empty, slots, thr_b, blur, shift=10):
    t0 = time.perf_counter()
    scores = slot_scores(signoff, ref_full, slots, blur, shift)
    flagged = flag(scores, thr_b)
    by_tool = {s["tool"]: s for s in slots}
    labels = {}
    for t in flagged:
        c = crop(signoff, by_tool[t])
        d_empty = structure_distance(c, crop(ref_empty, by_tool[t]), blur, shift)
        labels[t] = "missing" if d_empty < scores[t]["s"] else "wrong tool"
    return {"flagged": flagged, "labels": labels, "scores": scores,
            "process_s": time.perf_counter() - t0}


def outcome(truth, flagged):
    """Score one check against ground truth (truth = slots that SHOULD be flagged)."""
    truth, flagged = set(truth), set(flagged)
    missed = sorted(truth - flagged)
    extra = sorted(flagged - truth)
    return {
        "missed": missed,                    # tools not caught
        "extra": extra,                      # slots flagged that were fine
        "miss": int(len(missed) > 0),        # REQ-TR-02 failure
        "false_alarm": int(len(extra) > 0),  # REQ-TR-06 failure
        "named_exactly": int(flagged == truth),  # REQ-TR-01
    }
