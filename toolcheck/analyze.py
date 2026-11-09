"""Summarise a session into numbers the Stage 1 report can quote.
Writes data/<session>/summary.md and prints the same text."""
import csv
import json
from collections import defaultdict

from .paths import DATA, load_json
from .stats import upper_bound, percentile


def _rows(sdir):
    with open(sdir / "trials.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarise(session_id):
    sdir = DATA / session_id
    meta = load_json(sdir / "session.json")
    rows = _rows(sdir)
    valid = [r for r in rows if r["void"] == "0"]
    voided = [r for r in rows if r["void"] != "0"]
    limit = meta["settings"]["trials"].get("time_limit_s", 3.0)

    L = []
    L.append(f"# Session {session_id} summary\n")
    c = meta["conditions"]
    L.append("**Conditions:** " + "; ".join(f"{k}: {v}" for k, v in c.items() if v)
             + f"; resolution {meta['resolution'][0]}x{meta['resolution'][1]}"
             + f"; checkout mode {meta['checkout_mode']}; seed {meta['seed']}\n")
    L.append(f"**Thresholds:** {meta['thresholds']['rule']} (frozen {meta['thresholds']['created']})\n")
    L.append(f"**Trials:** {len(rows)} run, {len(valid)} valid, {len(voided)} voided\n")
    if voided:
        L.append("Voided: " + "; ".join(f"t{r['trial']} ({r['void_reason']})" for r in voided) + "\n")

    L.append("\n## Detection and false alarms (95% one-sided exact upper bounds)\n")
    L.append("| Block | Method | n | Misses | Miss rate | Upper bound | False alarms | FA rate | Upper bound | Named exactly |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for block in ("removal", "swap", "complete"):
        b = [r for r in valid if r["block"] == block]
        if not b:
            continue
        for m in ("A", "B"):
            n = len(b)
            miss = sum(int(r[f"{m}_miss"]) for r in b)
            fa = sum(int(r[f"{m}_false_alarm"]) for r in b)
            named = sum(int(r[f"{m}_named_exactly"]) for r in b)
            ms = f"{miss}" if block != "complete" else "-"
            mr = f"{miss / n:.1%}" if block != "complete" else "-"
            mu = f"{upper_bound(miss, n):.1%}" if block != "complete" else "-"
            L.append(f"| {block} | {m} | {n} | {ms} | {mr} | {mu} | {fa} | {fa / n:.1%} | "
                     f"{upper_bound(fa, n):.1%} | {named}/{n} |")

    L.append("\n## Misses by tool (removal + swap)\n")
    L.append("| Tool | Trials as truth | A missed | B missed |")
    L.append("|---|---|---|---|")
    per = defaultdict(lambda: [0, 0, 0])
    for r in valid:
        truth = json.loads(r["truth"])
        for t in truth:
            per[t][0] += 1
        a_flag, b_flag = set(json.loads(r["A_flagged"])), set(json.loads(r["B_flagged"]))
        for t in truth:
            per[t][1] += int(t not in a_flag)
            per[t][2] += int(t not in b_flag)
    for t in sorted(per):
        L.append(f"| {t} | {per[t][0]} | {per[t][1]} | {per[t][2]} |")

    L.append(f"\n## Check time, capture + processing on this PC (limit {limit:.1f} s)\n")
    L.append("| Method | n | Median s | 95th pct s | Max s | Over limit |")
    L.append("|---|---|---|---|---|---|")
    for m in ("A", "B"):
        ts = [float(r[f"{m}_total_s"]) for r in valid]
        over = sum(t > limit for t in ts)
        L.append(f"| {m} | {len(ts)} | {percentile(ts, .5):.3f} | {percentile(ts, .95):.3f} | "
                 f"{max(ts) if ts else float('nan'):.3f} | {over} |")
    L.append("\nTimes are measured on a stand-in PC with a USB webcam, not the custom hardware.\n")

    text = "\n".join(L)
    (sdir / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return text
