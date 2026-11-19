"""Threshold calibration on COMPLETE-kit captures only.

Thresholds are set from how much a slot's score varies when nothing is missing
(tools lifted and replaced by hand), then frozen to refs/<dir>/thresholds.json.
They are never tuned on trial results. Per slot and feature:

    threshold = max(floor, mean + k_sigma * std)
"""
import csv
import random
import statistics
from datetime import datetime

from .features import slot_scores, structure_distance, crop, gap_mask, lighting_change, lighting_limits_from_samples
from .paths import save_json


def thresholds_from_samples(samples, tools, k, s_floor, c_floor):
    """samples: list of {tool: {"s","c"}} -> {tool: {"s": thr, "c": thr}}"""
    thr = {}
    for t in tools:
        s_vals = [x[t]["s"] for x in samples]
        c_vals = [x[t]["c"] for x in samples]
        s_sd = statistics.pstdev(s_vals) if len(s_vals) > 1 else 0.0
        c_sd = statistics.pstdev(c_vals) if len(c_vals) > 1 else 0.0
        thr[t] = {"s": max(s_floor, statistics.mean(s_vals) + k * s_sd),
                  "c": max(c_floor, statistics.mean(c_vals) + k * c_sd)}
    return thr


def run_calibration(cam, slots, refs_dir, ref_full, ref_empty, settings, kit):
    cal = settings["calibration"]
    blur = settings["features"]["blur_sigma"]
    shift = settings["features"].get("shift_tolerance_px", 10)
    tools = [s["tool"] for s in slots]
    rng = random.Random()
    n = cal["captures"]
    print(f"\nCalibration: {n} captures of the COMPLETE kit.")
    print("Each time, lift the named tools and put them back roughly where they go.")
    print("Keep the lighting exactly as it will be for the removal trials.\n")

    input("Kit complete and untouched. Press Enter for the first capture...")
    prev, _ = cam.grab()
    rc = settings["retake"]
    gap = gap_mask(prev.shape, slots)
    light_samples = []
    samples_a, samples_b, rows = [], [], []
    for i in range(n):
        lifted = rng.sample(tools, min(cal["tools_lifted_per_capture"], len(tools)))
        input(f"[{i + 1}/{n}] Lift and replace: {', '.join(lifted)}. Enter when done...")
        img, _ = cam.grab()
        sa = slot_scores(img, prev, slots, blur, shift)      # A: vs the previous capture
        sb = slot_scores(img, ref_full, slots, blur, shift)  # B: vs stored reference
        samples_a.append(sa)
        samples_b.append(sb)
        for t in tools:
            rows.append([i + 1, t, int(t in lifted), sa[t]["s"], sa[t]["c"],
                         sb[t]["s"], sb[t]["c"]])
        light_samples.append(lighting_change(img, prev, gap, rc["blur_sigma"], rc["dark_tol"]))
        prev = img

    k, sf, cf = cal["k_sigma"], cal["structure_floor"], cal["colour_floor"]
    thr = {"A": thresholds_from_samples(samples_a, tools, k, sf, cf),
           "B": thresholds_from_samples(samples_b, tools, k, sf, cf),
           "created": datetime.now().isoformat(timespec="seconds"),
           "rule": f"max(floor, mean + {k} * std) over {n} complete-kit captures",
           "floors": {"s": sf, "c": cf}}
    thr["lighting"] = lighting_limits_from_samples(light_samples, k, rc["dark_floor"], rc["shift_floor"])
    save_json(refs_dir / "thresholds.json", thr)

    with open(refs_dir / "calibration_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["capture", "tool", "lifted", "A_s", "A_c", "B_s", "B_c"])
        w.writerows(rows)

    # Diagnostic only (uses the setup captures, not trial data): would removing
    # each tool clear its threshold? Reported, never used to change thresholds.
    print("\nSeparation check (full vs empty reference, structure distance):")
    print(f"{'tool':<24}{'removal':>9}{'thr A':>8}{'thr B':>8}  note")
    for s in slots:
        t = s["tool"]
        d = structure_distance(crop(ref_full, s), crop(ref_empty, s), blur, shift)
        note = ""
        if d <= thr["A"][t]["s"] or d <= thr["B"][t]["s"]:
            note = "WEAK: removal may not clear threshold (check colour too)"
        print(f"{t:<24}{d:9.3f}{thr['A'][t]['s']:8.3f}{thr['B'][t]['s']:8.3f}  {note}")
    print(f"\nThresholds frozen to {refs_dir / 'thresholds.json'}")
    return thr
