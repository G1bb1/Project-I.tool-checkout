"""Synthetic test of the watch link: no camera, no watch, no real photos.

Uses the fake shadow board of tests/test_toolai.py, trains a small SVM on two fake
sessions, then:
  1. replays the third fake session through WatchLink (checkout -> sign-off), checks
     every answer against the trial's truth and the fail-safe rules;
  2. live mode with a fake camera: refused checkout (swap left in), retake on a
     shadow, missing tool, absent tool (Flathead) empty = fine / not empty = flagged;
  3. real UDP on this PC: the server in a thread + the fake watch, then `report`;
  4. server OFF: every sign-off must end NOT CONFIRMED (timeout), never a pass.

Run from the project root:  python -m tests.test_watch_link
Synthetic results only show the code runs. They are NOT evidence.
"""
import csv
import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import tests.test_toolai as T


def signoff(link, mid, tries=3):
    """Sign off; on a retake caused by the fake camera's random light noise, sign off again."""
    for k in range(tries):
        r = link.handle({"cmd": "signoff", "id": mid * 10 + k})
        if r["status"] != "retake":
            return r
    return r


def main():
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(T.SRC / "toolai", tmp / "toolai")
    shutil.copy(T.SRC / "watch_main.py", tmp / "watch_main.py")
    (tmp / "config").mkdir()
    shutil.copy(T.SRC / "config" / "ai.json", tmp / "config" / "ai.json")
    (tmp / "config" / "slots.json").write_text(json.dumps({"slots": T.SLOTS}))
    sys.path.insert(0, str(tmp))
    for m in [m for m in sys.modules if m == "toolai" or m.startswith("toolai.") or m == "watch_main"]:
        del sys.modules[m]
    from toolai.common import load_ai_config, load_slots
    from toolai.importer import import_trials
    from toolai.infer import Predictor
    from toolai.train import train
    from toolai.watch_link import LiveSource, ReplaySource, WatchLink, serve
    import watch_main as WM

    dirs = []
    for k, (sid, started) in enumerate((("20990101-000000", "2099-01-01T00:00:00"),
                                        ("20990102-000000", "2099-01-02T00:00:00"),
                                        ("20990103-000000", "2099-01-03T00:00:00"))):
        d, thr, settings = T.fake_toolcheck_session(tmp, sid, started, k, lighting=(k == 2),
                                                    gain=(0.8, 1.0, 1.1)[k])
        dirs.append(d)
    cfg, slots = load_ai_config(), load_slots()
    cfg["absent_tools"] = []
    names = import_trials([str(d) for d in dirs], T.TOOLS, cfg)
    cfg["augment"]["copies"] = 1
    mdir = train(slots, cfg, "svm", test_sessions=[names[2]])

    # 1. replay of the held-out session through the link logic
    src = ReplaySource(dirs[2])
    from toolai.board import refs_empty_for
    pred = Predictor(mdir).use_board(slots, refs_empty_for(src.sj["refs_dir"]))
    out = tmp / "data" / "watch-replay"
    out.mkdir(parents=True)
    link = WatchLink(src, src.sj["slots"], src.sj["thresholds"], src.sj["settings"], pred, cfg, out)
    mid = 0
    finals = []
    for _ in range(len(src.jobs)):
        mid += 1
        r = link.handle({"cmd": "checkout", "id": mid})
        assert r["id"] == mid and r["op"] == "checkout" and r["status"] in ("ready", "refused"), r
        for _ in range(4):
            mid += 1
            r = link.handle({"cmd": "signoff", "id": mid})
            if r["status"] != "retake":
                break
        finals.append(r["status"])
    r = link.handle({"cmd": "checkout", "id": 999})
    assert r["status"] == "error", r                    # replay finished -> error, never a pass
    r = link.handle({"cmd": "bogus", "id": 1000})
    assert r["status"] == "error", r
    link.close()
    rows = list(csv.DictReader(open(out / "watch_log.csv")))
    scored = [x for x in rows if x["correct"] != ""]
    misses = sum(int(x["miss"]) for x in scored)
    fas = sum(int(x["false_alarm"]) for x in scored)
    retakes = sum(x["status"] == "retake" for x in rows)
    print(f"replay: {len(scored)} final sign-offs, {misses} misses, {fas} false alarms, {retakes} retakes; "
          f"finals {dict((s, finals.count(s)) for s in set(finals))}")
    assert misses == 0, [x for x in scored if x["miss"] == "1"]
    for x in rows:                                     # a pass only when the truth is a complete kit
        if x["status"] == "complete" and x["truth"]:
            assert x["should_flag"] == "[]", x

    # 2. live mode with a fake camera
    cam = T.FakeCam(5)
    out2 = tmp / "data" / "watch-live"
    out2.mkdir(parents=True)
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["lighting_override"]["enabled"] = False
    thr2 = src.sj["thresholds"]
    link = WatchLink(LiveSource(cam), slots, thr2, src.sj["settings"], pred, cfg2, out2)
    assert link.handle({"cmd": "signoff", "id": 1})["status"] == "no_checkout"
    assert link.handle({"cmd": "status", "id": 2})["status"] == "idle"
    cam.content = T.content_for("swap", ["Wire stripper", "Wire cutter"])
    r = link.handle({"cmd": "checkout", "id": 3})
    assert r["status"] == "refused" and any("Wire stripper" in t for t in r["tools"]), r
    assert link.handle({"cmd": "signoff", "id": 4})["status"] == "no_checkout"
    cam.content = {t: t for t in T.TOOLS}
    r = link.handle({"cmd": "checkout", "id": 5})
    assert r["status"] == "ready" and r["job"] == 1 and r["total"] == 12, r
    assert link.handle({"cmd": "status", "id": 6})["status"] == "job_open"
    cam.content = T.content_for("removal", ["Pliers"])
    r = signoff(link, 7)
    assert r["status"] == "missing" and r["tools"] == ["Pliers: missing"] and r["present"] == 11, r
    cam.content = {t: t for t in T.TOOLS}
    cam.light, cam.light_shots = "shadow", 1
    r = link.handle({"cmd": "signoff", "id": 8})
    assert r["status"] == "retake" and "shadow" in r["reason"], r
    r = signoff(link, 9)
    assert r["status"] == "complete" and r["present"] == 12, r
    assert link.handle({"cmd": "signoff", "id": 10})["status"] == "no_checkout"   # job closed
    # absent tool: Flathead not on the board
    link.close()
    cfg3 = json.loads(json.dumps(cfg2))
    cfg3["absent_tools"] = ["Flathead screwdriver"]
    out3 = tmp / "data" / "watch-absent"
    out3.mkdir(parents=True)
    link = WatchLink(LiveSource(cam), slots, thr2, src.sj["settings"], pred, cfg3, out3)
    cam.content = dict({t: t for t in T.TOOLS}, **{"Flathead screwdriver": "empty"})
    r = link.handle({"cmd": "checkout", "id": 1})
    assert r["status"] == "ready" and r["total"] == 11 and r["absent"] == ["Flathead screwdriver"], r
    r = signoff(link, 2)
    assert r["status"] == "complete" and r["present"] == 11, r
    link.handle({"cmd": "checkout", "id": 3})
    cam.content = dict(cam.content, **{"Flathead screwdriver": "Knife", "Knife": "empty"})
    r = link.handle({"cmd": "signoff", "id": 4})
    assert r["status"] in ("missing", "retake") and any(t.startswith("Flathead") for t in r["tools"]), r
    cam.content = dict({t: t for t in T.TOOLS})            # screwdriver put on the board at checkout
    r = link.handle({"cmd": "checkout", "id": 5})
    assert r["status"] == "refused" and "Flathead screwdriver: not empty" in r["tools"], r
    link.close()
    print("live link logic OK (refused checkout, missing, retake, complete, absent tool)")

    # 3. real UDP on this PC: server thread + fake watch
    src = ReplaySource(dirs[2])
    out4 = tmp / "data" / "watch-udp"
    out4.mkdir(parents=True)
    link = WatchLink(src, src.sj["slots"], src.sj["thresholds"], src.sj["settings"], pred, cfg, out4,
                     in_training=0)
    th = threading.Thread(target=serve, args=(link, 47211, 25), daemon=True)
    th.start()
    time.sleep(0.5)
    log = tmp / "sim.txt"
    sys.argv = ["watch_main.py", "sim", "--pc", "127.0.0.1", "--port", "47211", "--local-port", "47210",
                "--jobs", "6", "--gap", "0.05", "--log", str(log)]
    WM.main()
    th.join()
    sys.argv = ["watch_main.py", "report", str(out4), "--serial", str(log)]
    WM.main()
    s = (out4 / "summary.md").read_text()
    assert "without a 'complete' answer (must be 0): 0" in s, s
    assert "Watch showed something other than the server's answer: 0" in s, s
    wl = WM.read_watch_log(log)
    assert sum(r["op"] == "checkout" for r in wl) == 6, wl
    assert all(r["outcome"] not in ("not_confirmed_timeout", "checkout_timeout") for r in wl), wl

    # 4. server OFF: 3 sign-offs must all time out (10 s each is real, so 3 is enough here)
    log2 = tmp / "sim_off.txt"
    sys.argv = ["watch_main.py", "sim", "--pc", "127.0.0.1", "--port", "47299", "--local-port", "47298",
                "--signoffs", "3", "--gap", "0", "--log", str(log2)]
    WM.main()
    sys.argv = ["watch_main.py", "report", "--serial", str(log2), "--timeout-test"]
    WM.main()
    assert "3/3 NOT CONFIRMED after the 10 s limit -> PASS" in (tmp / "sim_off_report" / "summary.md").read_text()
    lat = [r["latency_ms"] for r in WM.read_watch_log(log2)]
    assert all(10000 <= x < 10500 for x in lat), lat

    print("\nOK: replay link logic, live link logic, UDP round trip + report, server-off timeout all ran.")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
