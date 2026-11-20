"""Watch checkout / sign-off link for the hybrid (Method A + model).

Put this file next to ai_main.py (project root) and toolai/watch_link.py in toolai/.
It does not change ai_main.py or any frozen setting.

    python watch_main.py serve                         # live: camera + model, watch talks to it
    python watch_main.py serve --replay data/<session> # stored photos, no camera (link test)
    python watch_main.py sim --jobs 10                 # fake watch on this PC (test before flashing)
    python watch_main.py sim --signoffs 60             # fake watch, server OFF: 60/60 timeout run
    python watch_main.py report data/watch-<...> --serial watch_serial.txt
                                                       # summary.md from the server log + watch log

The watch sketch is watch_firmware/toolcheck_watch_v2/toolcheck_watch_v2.ino.
Steps and what counts as a pass: WATCH_TEST.md.
"""
import argparse
import csv
import json
import socket
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from toolai.common import ROOT, load_ai_config, load_json, load_slots, load_toolcheck_settings

PORT_DEFAULT = 4211
TIMEOUT_S = 10.0


def _port(a):
    return a.port or int(load_toolcheck_settings().get("watch_server", {}).get("listen_port", PORT_DEFAULT))


def _current_thresholds():
    cur = ROOT / "refs" / "current.json"
    if not cur.exists():
        raise SystemExit("No refs/current.json. Run toolcheck's capture-refs and calibrate first.")
    p = ROOT / "refs" / load_json(cur)["dir"] / "thresholds.json"
    if not p.exists():
        raise SystemExit("No thresholds for the current references. Run: python main.py calibrate")
    return load_json(p)


def _in_training(pred, sid):
    return int(any(sid in s for s in (pred.meta.get("train_sessions") or [])))


def cmd_serve(a):
    from toolai.board import current_refs_empty, refs_empty_for
    from toolai.infer import Predictor, resolve_model_dir
    from toolai.watch_link import LiveSource, ReplaySource, WatchLink, new_out_dir, serve, write_session
    cfg = load_ai_config()
    if cfg.get("test_mode", {}).get("allow_force_accept"):
        raise SystemExit("ai.json test_mode.allow_force_accept is true. It must be false for the watch link.")
    pred = Predictor(resolve_model_dir(a.model), a.int8)

    if a.replay:
        src = ReplaySource(Path(a.replay))
        sj = src.sj
        slots, thr, settings = sj["slots"], sj["thresholds"], sj["settings"]
        if "retake" not in settings and "lighting" in thr:      # same fallback as `ai_main.py replay`
            settings["retake"] = {"enabled": True, "max_flagged": 3, "blur_sigma": 8.0, "dark_tol": 0.15}
        empty = refs_empty_for(sj.get("refs_dir")) or current_refs_empty()
        pred.use_board(slots, empty, f"{src.sid}: {empty}")
        in_tr = _in_training(pred, src.sid)
        out = new_out_dir("replay")
        info = {"source": "replay", "replay_session": src.sid, "in_training": in_tr}
        if in_tr:
            print("NOTE: the model was trained on this session. The link test is valid; "
                  "the right/wrong counts are NOT accuracy evidence.")
    else:
        from toolai.camera import open_camera
        from toolai.common import load_camera_settings
        slots, thr, settings = load_slots(), _current_thresholds(), load_toolcheck_settings()
        pred.use_board(slots, current_refs_empty(), "current refs")
        cam = open_camera(load_camera_settings())
        src = LiveSource(cam)
        img, _ = cam.grab()
        mm = pred.board.ref_mismatch(img) if pred.board is not None else None
        lim = cfg["board"].get("tool_only", {}).get("max_ref_mismatch", 0.05)
        if mm is not None:
            print(f"Empty-board reference check: {mm:.1%} of the bare paper no longer matches (limit {lim:.0%}).")
            if mm > lim and input("WARNING: run capture-refs + calibrate first? y = stop | Enter = continue: "
                                  ).strip().lower() == "y":
                cam.close()
                return
        in_tr = 0
        out = new_out_dir("live")
        info = {"source": "live"}
        conditions = {k: input(f"  {q}: ").strip() for k, q in (
            ("lighting", "Lighting"), ("operator", "Operator"), ("logbook_entry", "Logbook entry"))}
        info["conditions"] = conditions

    port = _port(a)
    write_session(out, **info, listen_port=port, model=pred.dir.name, model_file=pred.model_file.name,
                  model_train_sessions=pred.meta.get("train_sessions"), ai_config=cfg,
                  thresholds=thr, settings=settings, slots=slots)
    link = WatchLink(src, slots, thr, settings, pred, cfg, out, in_training=in_tr)
    print(f"Absent tools: {link.absent or 'none'}; {link.total} tools counted; max retakes {link.max_retakes}.")
    try:
        serve(link, port)
    finally:
        src.close()
    print(f"Next: python watch_main.py report {out} --serial <your watch log .txt>")


# ------------------------------------------------------------------ fake watch
class FakeWatch:
    """Speaks the same protocol as the watch sketch and prints the same LOG lines."""

    def __init__(self, pc, port, local_port, logf):
        self.pc = (pc, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", local_port))
        self.id = 0
        self.t0 = time.monotonic()
        self.logf = logf
        self.out("LOG,id,op,start_ms,outcome,latency_ms,detail")

    def out(self, line):
        print(line)
        self.logf.write(line + "\n")
        self.logf.flush()

    def ask(self, op):
        self.id += 1
        start = time.monotonic()
        start_ms = int((start - self.t0) * 1000)
        self.sock.sendto(json.dumps({"cmd": op, "id": self.id}).encode(), self.pc)
        reply = None
        while True:
            left = TIMEOUT_S - (time.monotonic() - start)
            if left <= 0:
                break
            self.sock.settimeout(left)
            try:
                data, _ = self.sock.recvfrom(2048)
            except socket.timeout:
                break
            except ConnectionResetError:        # Windows: port unreachable when the server is off
                continue
            try:
                r = json.loads(data.decode())
            except ValueError:
                continue
            if r.get("id") == self.id and r.get("op") == op:
                reply = r
                break
            self.out(f"# ignored stale reply id {r.get('id')}")
        lat = int((time.monotonic() - start) * 1000)
        outcome, detail = outcome_of(op, reply)
        self.out(f"LOG,{self.id},{op},{start_ms},{outcome},{lat},{detail}")
        return outcome


def outcome_of(op, r):
    """Same mapping as the sketch: only a matching 'complete' is a pass."""
    if r is None:
        return ("checkout_timeout" if op == "checkout" else "not_confirmed_timeout"), ""
    st, tools = r.get("status"), "; ".join(r.get("tools", []))
    n = f"{r.get('present', '?')}/{r.get('total', '?')}"
    table = {("checkout", "ready"): ("checkout_ok", n), ("checkout", "refused"): ("checkout_refused", tools),
             ("signoff", "complete"): ("confirmed", n), ("signoff", "missing"): ("missing", tools),
             ("signoff", "retake"): ("retake", r.get("reason", "")),
             ("signoff", "no_checkout"): ("no_checkout", ""), ("status", "job_open"): ("status_job_open", ""),
             ("status", "idle"): ("status_idle", "")}
    return table.get((op, st), ("not_confirmed_bad_reply", f"{st} {r.get('reason', '')}".strip()))


def cmd_sim(a):
    path = Path(a.log or f"watch_sim_{datetime.now():%Y%m%d-%H%M%S}.txt")
    with open(path, "w", encoding="utf-8") as f:
        w = FakeWatch(a.pc, _port(a), a.local_port, f)
        if a.signoffs:
            for _ in range(a.signoffs):
                w.ask("signoff")
                time.sleep(a.gap)
        for _ in range(a.jobs):
            w.ask("checkout")
            time.sleep(a.gap)
            for _ in range(4):                  # retake: sign off again (the server limits retakes)
                if w.ask("signoff") != "retake":
                    break
                time.sleep(a.gap)
            time.sleep(a.gap)
    print(f"Fake watch log: {path}")


# ------------------------------------------------------------------ report
def read_watch_log(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("LOG,") or line.startswith("LOG,id,"):
            continue
        p = line.split(",", 6)
        if len(p) < 6:
            continue
        rows.append({"id": p[1], "op": p[2], "start_ms": p[3], "outcome": p[4],
                     "latency_ms": int(p[5]) if p[5].lstrip("-").isdigit() else None,
                     "detail": p[6] if len(p) > 6 else ""})
    return rows


def _lat(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return "n/a"
    p95 = xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]
    return f"n={len(xs)}, median {statistics.median(xs):.0f} ms, 95th pct {p95} ms, max {xs[-1]} ms"


def cmd_report(a):
    if not a.out_dir and not a.serial:
        raise SystemExit("Give the data/watch-<...> folder, the --serial watch log, or both.")
    d = Path(a.out_dir) if a.out_dir else Path(a.serial).resolve().parent / (Path(a.serial).stem + "_report")
    d.mkdir(parents=True, exist_ok=True)
    server = list(csv.DictReader(open(d / "watch_log.csv", encoding="utf-8"))) if (d / "watch_log.csv").exists() else []
    info = load_json(d / "watch_session.json") if (d / "watch_session.json").exists() else {}
    watch = read_watch_log(a.serial) if a.serial else []
    L = [f"# Watch link test: {d.name}", ""]
    if info:
        L.append(f"Source: {info.get('source', '?')}"
             + (f" (replay of {info.get('replay_session')})" if info.get("source") == "replay" else "")
             + f"; model {info.get('model', '?')}; generated {datetime.now():%Y-%m-%d %H:%M}.")
    else:
        L.append(f"Watch log only ({a.serial}); no server log. Generated {datetime.now():%Y-%m-%d %H:%M}.")
    if info.get("in_training"):
        L.append("**The model was trained on the replayed session: right/wrong counts below are NOT accuracy "
                 "evidence. This run tests the link only.**")
    L.append("")

    if watch:
        L += ["## Watch side (end-to-end, measured on the watch)", ""]
        by = {}
        for r in watch:
            by.setdefault((r["op"], r["outcome"]), []).append(r)
        L += ["| command | outcome | count | latency |", "|---|---|---|---|"]
        for (op, oc), rs in sorted(by.items()):
            L.append(f"| {op} | {oc} | {len(rs)} | {_lat([r['latency_ms'] for r in rs])} |")
        so = [r for r in watch if r["op"] == "signoff"]
        to = [r for r in so if r["outcome"] == "not_confirmed_timeout"]
        L += ["", f"Sign-off checks: {len(so)}; NOT CONFIRMED by timeout: {len(to)}; "
              f"KIT CONFIRMED: {sum(r['outcome'] == 'confirmed' for r in so)}.",
              f"Answered sign-offs, end-to-end latency: "
              f"{_lat([r['latency_ms'] for r in so if r['outcome'] != 'not_confirmed_timeout'])}.", ""]
        if a.timeout_test:
            ok = len(so) > 0 and len(to) == len(so)
            L += [f"**Timeout test (server off): {len(to)}/{len(so)} NOT CONFIRMED after the 10 s limit -> "
                  f"{'PASS' if ok else 'FAIL'}.** Any other outcome with the server off is a fail-safe failure.", ""]

    if server:
        L += ["## Server side (what the PC/processor answered)", ""]
        cnt = {}
        for r in server:
            cnt[(r["cmd"], r["status"])] = cnt.get((r["cmd"], r["status"]), 0) + 1
        L += ["| command | status | count |", "|---|---|---|"] + \
             [f"| {c} | {s} | {n} |" for (c, s), n in sorted(cnt.items())]
        st = [float(r["server_total_s"]) * 1000 for r in server if r["cmd"] == "signoff" and r["server_total_s"]]
        L += ["", f"Server time per sign-off (capture + Method A + model + decision): "
              f"{_lat([round(x) for x in st])}.", ""]
        scored = [r for r in server if r.get("correct") not in (None, "")]
        if scored:
            miss = sum(int(r["miss"]) for r in scored)
            fa = sum(int(r["false_alarm"]) for r in scored)
            L += [f"Replay answers scored against the trial's truth: {len(scored)} final sign-offs, "
                  f"{miss} missed a changed slot, {fa} false alarms, "
                  f"{sum(int(r['correct']) for r in scored)} exactly right."
                  + (" (in_training: link test only.)" if info.get("in_training") else ""), ""]

    if watch and server:
        L += ["## Watch screen vs server answer (same check id)", ""]
        srv = {}
        for r in server:
            srv.setdefault(r["msg_id"], []).append(r)
        dup = [k for k, v in srv.items() if len(v) > 1]
        want = {"ready": "checkout_ok", "refused": "checkout_refused", "complete": "confirmed",
                "missing": "missing", "retake": "retake", "no_checkout": "no_checkout",
                "job_open": "status_job_open", "idle": "status_idle"}
        bad, lost, unsafe = [], [], []
        for r in watch:
            s = srv.get(r["id"])
            if not s:
                continue
            exp = want.get(s[-1]["status"], "not_confirmed_bad_reply")
            if r["outcome"] in ("not_confirmed_timeout", "checkout_timeout"):
                lost.append(r["id"])
            elif r["outcome"] != exp:
                bad.append((r["id"], s[-1]["status"], r["outcome"]))
            if r["outcome"] == "confirmed" and s[-1]["status"] != "complete":
                unsafe.append(r["id"])
        L += [f"- Server answered but the watch timed out (reply lost or later than 10 s): {len(lost)} {lost[:10]}",
              f"- Watch showed something other than the server's answer: {len(bad)} {bad[:10]}",
              f"- Watch showed KIT CONFIRMED without a 'complete' answer (must be 0): {len(unsafe)}"]
        if dup:
            L.append(f"- Note: ids seen more than once in the server log (watch restarted?): {dup[:10]}")
        L.append("")
    (d / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"-> {d / 'summary.md'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="answer the watch (live camera, or --replay a stored session)")
    s.add_argument("--replay", default=None, help="data/<session> with images/ and trials.csv")
    s.add_argument("--model", default=None)
    s.add_argument("--int8", action="store_true")
    s.add_argument("--port", type=int, default=None)
    s.set_defaults(fn=cmd_serve)
    m = sub.add_parser("sim", help="fake watch on this PC")
    m.add_argument("--pc", default="127.0.0.1")
    m.add_argument("--port", type=int, default=None)
    m.add_argument("--local-port", type=int, default=4210)
    m.add_argument("--jobs", type=int, default=0, help="checkout -> sign-off jobs")
    m.add_argument("--signoffs", type=int, default=0, help="sign-offs only (timeout test with the server off)")
    m.add_argument("--gap", type=float, default=1.0, help="seconds between messages")
    m.add_argument("--log", default=None)
    m.set_defaults(fn=cmd_sim)
    r = sub.add_parser("report", help="summary.md from data/watch-<...> and the watch's serial log")
    r.add_argument("out_dir", nargs="?", default=None,
                   help="data/watch-<...>; leave out for the server-OFF timeout run (watch log only)")
    r.add_argument("--serial", default=None, help="Serial Monitor text saved from the watch (or the sim log)")
    r.add_argument("--timeout-test", action="store_true", help="the serial log is the server-OFF 60/60 run")
    r.set_defaults(fn=cmd_report)
    argv = sys.argv[1:] or menu()
    a = p.parse_args(argv)
    a.fn(a)


def menu():
    steps = [("serve", "live: camera + model answer the watch"),
             ("serve --replay", "stored session photos answer the watch (no camera)"),
             ("sim", "fake watch on this PC"),
             ("report", "summary.md for a watch run")]
    print("watch link: choose a step")
    for i, (c, d) in enumerate(steps, 1):
        print(f"  {i}. {c:<15} {d}")
    c = input("Number: ").strip()
    if c == "1":
        return ["serve"]
    if c == "2":
        return ["serve", "--replay", input("  Session folder (data/<session>): ").strip()]
    if c == "3":
        n = input("  Jobs [10] (0 = timeout test): ").strip() or "10"
        return ["sim", "--jobs", n] if n != "0" else ["sim", "--signoffs", input("  Sign-offs [60]: ").strip() or "60"]
    if c == "4":
        return ["report", input("  data/watch-<...> folder: ").strip(), "--serial",
                input("  Watch log .txt: ").strip()]
    raise SystemExit("No valid choice.")


if __name__ == "__main__":
    main()
