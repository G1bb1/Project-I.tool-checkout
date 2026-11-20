"""Measure what one AI check costs on the machine it runs on (R3, R4, REQ-TR-05, REQ-TR-07).

Run it on the candidate processor (e.g. the Pi Zero 2 W) with the kit complete.
It does n checks back to back and records, per stage: capture, crop+prepare,
inference, decide; plus peak memory, model file size, resolution and platform.

Writes models/<model>/profile-<host>-<stamp>.json
"""
import json
import os
import platform
import statistics
import time
from datetime import datetime

from .common import peak_memory_mb, save_json
from .infer import Predictor, check_frame


def _stats(xs):
    xs = sorted(xs)
    return {"median_ms": round(statistics.median(xs) * 1000, 1),
            "p95_ms": round(xs[int(0.95 * (len(xs) - 1))] * 1000, 1),
            "max_ms": round(xs[-1] * 1000, 1)}


def run_profile(cam, slots, model_dir, n=30, use_int8=False):
    mem_before_model = peak_memory_mb()
    t0 = time.perf_counter()
    from .board import current_refs_empty
    pred = Predictor(model_dir, use_int8).use_board(slots, current_refs_empty(), "current refs")
    load_s = time.perf_counter() - t0
    mem_after_model = peak_memory_mb()

    stages = {"capture_s": [], "crop_prepare_s": [], "inference_s": [], "decide_s": [], "total_s": []}
    flagged_counts = []
    print(f"Profiling {n} checks with {pred.model_file.name}. Keep the kit complete and still.")
    for i in range(n):
        img, cap_s = cam.grab()
        res, t = check_frame(pred, img, slots)
        stages["capture_s"].append(cap_s)
        for k, v in t.items():
            stages[k].append(v)
        stages["total_s"].append(cap_s + sum(t.values()))
        flagged_counts.append(sum(r["flag"] for r in res))
        print(f"  {i + 1}/{n}: {stages['total_s'][-1] * 1000:.0f} ms, flagged {flagged_counts[-1]}")

    h, w = img.shape[:2]
    result = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(), "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}", "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "model": model_dir.name, "model_file": pred.model_file.name,
        "model_size_kb": round(pred.model_size_kb(), 1), "model_load_s": round(load_s, 2),
        "resolution": [w, h], "slots": len(slots), "checks": n,
        "peak_memory_mb": {"before_model": mem_before_model, "after_model": mem_after_model,
                           "after_checks": peak_memory_mb()},
        "stages": {k: _stats(v) for k, v in stages.items()},
        "checks_over_3s": sum(t > 3.0 for t in stages["total_s"]),
        "flags_on_complete_kit": sum(1 for c in flagged_counts if c),
        "note": "Peak memory is for the whole Python process (interpreter, OpenCV, camera "
                "and model). End-to-end latency through the watch is measured by the watch "
                "sketch's latency_ms, not here."}
    out = model_dir / f"profile-{platform.node()}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    save_json(out, result)
    print("\n" + json.dumps({k: result[k] for k in
                             ("resolution", "model_size_kb", "peak_memory_mb", "stages",
                              "checks_over_3s", "flags_on_complete_kit")}, indent=2))
    print(f"Saved {out}")
    return result
