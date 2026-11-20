"""summary.md for a hybrid session or a replay: Method A vs hybrid on the same photos,
retakes by category, lighting overrides, AI-cleared flags, and forced trials kept apart."""
import csv
import json
import statistics
from pathlib import Path

from .common import upper_bound


def _pct(k, n):
    return f"{k}/{n}" + (f" ({k / n:.0%})" if n else "")


def write_summary(out_dir, rows, title, replay=False, warning=""):
    out_dir = Path(out_dir)
    as_int = lambda r, k: int(r.get(k) or 0)        # noqa: E731
    main = [r for r in rows if not as_int(r, "void") and not as_int(r, "excluded")
            and not as_int(r, "forced") and r.get("photo", "signoff") == "signoff"]
    L = [f"# {title}", ""]
    if warning:
        L += [f"**{warning}**", ""]
    L += [f"Rows counted: {len(main)} final sign-off photos (voided, excluded and forced trials are listed "
          "separately below and are NOT in these tables).", "",
          "## Misses and false alarms (95% one-sided exact upper bounds)", "",
          "Three systems on the SAME final photos: A = Method A alone, Model = the AI alone "
          "(no Method A, no retake check), Hybrid = the verdict actually given.", "",
          "| Block | n | A misses | Model misses | Hybrid misses | Hybrid upper bound | A false alarms "
          "| Model false alarms | Hybrid false alarms | Hybrid FA upper bound | Model named correctly "
          "| Hybrid named correctly |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in ("removal", "swap", "complete"):
        rs = [r for r in main if r["block"] == b]
        n = len(rs)
        if not n:
            continue
        am, hm = sum(as_int(r, "A_miss") for r in rs), sum(as_int(r, "H_miss") for r in rs)
        af, hf = sum(as_int(r, "A_false_alarm") for r in rs), sum(as_int(r, "H_false_alarm") for r in rs)
        nc = sum(as_int(r, "H_named_correctly") for r in rs)
        has_m = all(r.get("M_miss", "") != "" for r in rs)
        mm = str(sum(as_int(r, "M_miss") for r in rs)) if has_m else "-"
        mf = str(sum(as_int(r, "M_false_alarm") for r in rs)) if has_m else "-"
        mnc = _pct(sum(as_int(r, "M_named_correctly") for r in rs), n) if has_m else "-"
        miss_cols = ("-", "-", "-", "-") if b == "complete" else (str(am), mm, str(hm), f"{upper_bound(hm, n):.1%}")
        L.append(f"| {b} | {n} | {miss_cols[0]} | {miss_cols[1]} | {miss_cols[2]} | {miss_cols[3]} | {af} | {mf} "
                 f"| {hf} | {upper_bound(hf, n):.1%} | {mnc} | {_pct(nc, n)} |")

    cleared = [r for r in main if json.loads(r.get("H_cleared_by_ai") or "[]")]
    L += ["", "## What the model changed", "",
          f"- Photos where the AI cleared one of Method A's flags (tool reseated, AI sure it is the own tool): "
          f"{len(cleared)}" + (": trials " + ", ".join(str(r["trial"]) for r in cleared) if cleared else ""),
          f"- Lighting override (light changed, AI named every tool, no retake): "
          f"{sum(as_int(r, 'lighting_override') for r in rows)} photos",
          f"- Many slots flagged but all named, no retake: {sum(as_int(r, 'many_slots_named') for r in rows)} photos"]

    L += ["", "## Retakes by category", ""]
    if replay:
        tries = [r for r in rows if r.get("photo") == "signoff_try1"]
        L += ["In a replay, `signoff_try1` rows are the real first photos that made toolcheck ask for a retake.",
              "| Trial | Category the hybrid would give | Lighting override | Hybrid flags on that photo | Truth |",
              "|---|---|---|---|---|"]
        for r in tries:
            L.append(f"| {r['trial']} | {r['retake_category'] or 'none (no retake)'} | {r['lighting_override']} "
                     f"| {', '.join(json.loads(r['H_flagged'])) or '-'} | {', '.join(json.loads(r['truth'])) or 'complete'} |")
        hidden = [r for r in tries if as_int(r, "lighting_override") and as_int(r, "H_miss")]
        L.append(f"\nOverride hid a real problem: {len(hidden)} (must be 0).")
    else:
        cats = {}
        for r in rows:
            if r.get("retake_category"):
                cats[r["retake_category"]] = cats.get(r["retake_category"], 0) + 1
        L += [f"- {k}: {v} trials" for k, v in sorted(cats.items())] or ["- none"]
        L.append(f"- Retakes ran out (last photo's verdict stands): "
                 f"{sum(as_int(r, 'retake_exhausted') for r in rows)}")
        co = [r for r in rows if as_int(r, "checkout_retakes") or r.get("checkout_issue_ignored")]
        L.append(f"- Checkout photo not complete (caught before the trial): {len(co)} trials"
                 + (": " + ", ".join(str(r["trial"]) for r in co) if co else ""))

    if replay:
        co = [r for r in rows if r.get("checkout_issues") and r.get("photo") == "signoff"]
        L += ["", "## Checkout photos the model says were not complete", ""]
        L += [f"- trial {r['trial']}: {r['checkout_issues']}" for r in co] or ["- none"]

    forced = [r for r in rows if as_int(r, "forced")]
    voided = [r for r in rows if as_int(r, "void") and r.get("photo", "signoff") == "signoff"]
    excl = [r for r in rows if as_int(r, "excluded") and r.get("photo", "signoff") == "signoff"]
    L += ["", "## Kept apart", "",
          f"- FORCED (test-only force-accept): {len(forced)}"
          + "".join(f"\n  - trial {r['trial']}: retake reason '{r['retake_reason']}', "
                    f"hybrid miss {r['H_miss']}, false alarm {r['H_false_alarm']}" for r in forced),
          f"- Voided: {', '.join(str(r['trial']) for r in voided) or 'none'}",
          f"- Excluded (ai_exclude.json): {', '.join(str(r['trial']) for r in excl) or 'none'}"]

    t_ai = [float(r["AI_process_s"]) for r in main if r.get("AI_process_s")]
    t_tot = [float(r["H_total_s"]) for r in main if r.get("H_total_s")]
    L += ["", "## Time", ""]
    if t_ai:
        L.append(f"- Model per photo (all 12 slots): median {statistics.median(t_ai) * 1000:.0f} ms, "
                 f"max {max(t_ai) * 1000:.0f} ms")
    if t_tot:
        L.append(f"- Capture + A + model: median {statistics.median(t_tot):.2f} s, max {max(t_tot):.2f} s, "
                 f"over 3 s: {sum(t > 3 for t in t_tot)}")
    L.append("- Measured on the machine that ran this, not the custom hardware.")
    (out_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nSaved {out_dir / 'summary.md'}")


def summarise_session(sdir):
    with open(Path(sdir) / "trials.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    write_summary(sdir, rows, f"Hybrid session {Path(sdir).name}")
