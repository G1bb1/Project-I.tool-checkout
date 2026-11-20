"""Synthetic test of the PoC runner (no camera, no watch, no real photos).

Trains a small SVM on fake sessions, then drives PocLink through a short plan the way the
watch would (checkout -> sign-off(s), retakes), checks the plan bookkeeping, the per-tool
clear bar, resume after a stop mid-trial, the UDP loop, and the results table.

Run from the project root:  python -m tests.test_poc
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


def main():
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(T.SRC / "toolai", tmp / "toolai")
    (tmp / "config").mkdir()
    shutil.copy(T.SRC / "config" / "ai.json", tmp / "config" / "ai.json")
    (tmp / "config" / "slots.json").write_text(json.dumps({"slots": T.SLOTS}))
    sys.path.insert(0, str(tmp))
    for m in [m for m in sys.modules if m == "toolai" or m.startswith("toolai.")]:
        del sys.modules[m]
    from toolai.common import load_ai_config, load_slots
    from toolai.hybrid import clear_bar, fmt_conf
    from toolai.importer import import_trials
    from toolai.infer import Predictor
    from toolai.train import train
    from toolai.board import refs_empty_for
    from toolai import poc

    # per-tool clear bar
    cfg = load_ai_config()
    d = cfg["decision"]
    assert clear_bar("Wire stripper", d) == 0.90 and clear_bar("Wire cutter", d) == 0.90
    assert clear_bar("Measuring tape", d) == 0.60 and clear_bar("Nippers", d) == 0.60
    assert clear_bar("Painter's tape", d, colour_only=False) == 0.60
    old = {k: v for k, v in d.items() if k not in ("clear_confidence_by_tool", "clear_confidence_default")}
    assert clear_bar("Measuring tape", old, colour_only=False) == 0.90          # v3: shape flag -> 0.90
    assert clear_bar("Measuring tape", old, colour_only=True) == 0.60
    assert fmt_conf(0.596) == "0.59" and fmt_conf(0.6) == "0.60"
    print("per-tool clear bar OK")

    dirs = []
    for k, (sid, started) in enumerate((("20990101-000000", "2099-01-01T00:00:00"),
                                        ("20990102-000000", "2099-01-02T00:00:00"),
                                        ("20990103-000000", "2099-01-03T00:00:00"))):
        dd, thr, settings = T.fake_toolcheck_session(tmp, sid, started, k, lighting=(k == 2),
                                                     gain=(0.8, 1.0, 1.1)[k])
        dirs.append(dd)
    slots = load_slots()
    cfg["absent_tools"] = ["Flathead screwdriver"]
    names = import_trials([str(x) for x in dirs], T.TOOLS, cfg)
    cfg["augment"]["copies"] = 1
    mdir = train(slots, cfg, "svm", test_sessions=[names[2]])
    from toolai.replay import load_json
    sj = load_json(dirs[2] / "session.json")
    pred = Predictor(mdir).use_board(slots, refs_empty_for(sj["refs_dir"]))
    cfg["lighting_override"]["enabled"] = False

    kit = {"perturbations": ["Lift {tool} and put it back", "No change"]}
    plan = poc.build_plan(T.TOOLS, kit, cfg, n_removal=6, n_complete=5, n_swap=2, seed=1)
    assert len(plan["trials"]) == 13
    assert all(t["tool"] != "Flathead screwdriver" for t in plan["trials"])
    rem = [t["tool"] for t in plan["trials"] if t["block"] == "removal"]
    assert len(set(rem)) == 6                                       # one round, no repeats
    sdir = tmp / "data" / "poc-test"
    (sdir / "part1").mkdir(parents=True)
    json.dump(plan, open(sdir / "plan.json", "w"))
    json.dump({"model": "m", "freeze": {"refs_dir": "x"}}, open(sdir / "poc_session.json", "w"))

    cam = T.FakeCam(3)
    full = dict({t: t for t in T.TOOLS}, **{"Flathead screwdriver": "empty"})
    cam.content = dict(full)
    runner = poc.Runner(sdir, plan, T.TOOLS, plan["absent_tools"], 1)
    src = poc.PlanSource(cam, lambda: cam, runner)
    link = poc.PocLink(runner, src, slots, sj["thresholds"], sj["settings"], pred, cfg, sdir / "part1")
    mid = [0]

    def press(cmd):
        mid[0] += 1
        return link.handle({"cmd": cmd, "id": mid[0]})

    def signoff_until_final():
        for _ in range(4):
            r = press("signoff")
            if r["status"] != "retake":
                return r
        return r

    stop_at = 8           # stop mid-trial here to test resume
    while not runner.finished() and runner.trial()["n"] < stop_at:
        cam.content = dict(full)
        assert press("checkout")["status"] == "ready"
        t = runner.trial()                                     # the checkout may have skipped an unfinished trial
        for st in t["steps"][1:]:
            if runner.trial() is not t:
                break                                          # removal missed -> rest skipped
            cam.content = T.content_for(st["truth_block"], st["truth"]) if st["truth"] else dict(full)
            cam.content["Flathead screwdriver"] = "empty"
            if t["n"] == 2 and st["phase"] == "removed":       # one retake on a shadow
                cam.light, cam.light_shots = "shadow", 1
            if t["n"] == 3 and st["phase"] == "returned":
                break                                          # operator holds for a new checkout instead
            signoff_until_final()
        if t["n"] == 3:
            cam.content = dict(full)
    # trial 8: checkout done, then the run stops
    cam.content = dict(full)
    assert press("checkout")["status"] == "ready"
    link.close()
    runner.close()
    assert runner.trial()["n"] == 8 and runner.step_i == 1

    # resume: trial 8 is skipped (its job is gone), continue at 9
    (sdir / "part2").mkdir()
    runner = poc.Runner(sdir, plan, T.TOOLS, plan["absent_tools"], 2)
    assert runner.trial()["n"] == 9 and runner.step_i == 0
    src = poc.PlanSource(cam, lambda: cam, runner)
    link = poc.PocLink(runner, src, slots, sj["thresholds"], sj["settings"], pred, cfg, sdir / "part2")

    # the rest over real UDP with a fake watch speaking the same protocol
    th = threading.Thread(target=poc.serve_poc, args=(link, src, 47311), kwargs={"stop_after": 30}, daemon=True)
    th.start()
    time.sleep(0.5)
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(10)
    log = [("LOG,id,op,start_ms,outcome,latency_ms,detail")]
    want = {"ready": "checkout_ok", "complete": "confirmed", "missing": "missing", "retake": "retake",
            "no_checkout": "no_checkout", "error": "not_confirmed_bad_reply"}
    k = [1000]

    def ask(op):
        k[0] += 1
        t0 = time.monotonic()
        s.sendto(json.dumps({"cmd": op, "id": k[0]}).encode(), ("127.0.0.1", 47311))
        r = json.loads(s.recvfrom(2048)[0])
        log.append(f"LOG,{k[0]},{op},0,{want.get(r['status'], 'x')},{int((time.monotonic() - t0) * 1000)},")
        return r

    while not runner.finished():
        cam.content = dict(full)
        assert ask("checkout")["status"] == "ready"
        t = runner.trial()
        for st in t["steps"][1:]:
            if runner.trial() is not t:
                break
            cam.content = T.content_for(st["truth_block"], st["truth"]) if st["truth"] else dict(full)
            cam.content["Flathead screwdriver"] = "empty"
            for _ in range(4):
                if ask("signoff")["status"] != "retake":
                    break
    assert ask("signoff")["status"] in ("error",)               # plan finished: never a pass
    th.join()
    (tmp / "watch.txt").write_text("\n".join(log))

    rows = poc.read_checks(sdir)
    phases = [r["phase"] for r in rows if r["status"] != "skipped"]
    skipped = [r for r in rows if r["status"] == "skipped"]
    print(f"rows {len(rows)}: counted {len(phases)}, skipped {len(skipped)}: "
          f"{[(r['trial'], r['phase'], r['note'][:30]) for r in skipped]}")
    assert any(r["trial"] == "3" and r["phase"] == "returned" for r in skipped)
    assert any(r["trial"] == "8" for r in skipped)
    for r in rows:
        if r["status"] == "complete":
            assert r["truth_should_flag"] == "[]" or r["miss"] == "1", r
        if r["status"] != "skipped":
            assert r["criteria"] and r["server_total_s"], r
    rem_rows = [r for r in rows if r["phase"] == "removed" and r["status"] != "skipped"]
    assert all(r["criteria"] == "CL1+CL4+CL8" for r in rem_rows)
    assert sum(int(r["miss"]) for r in rem_rows) == 0, rem_rows
    out = poc.summarise(sdir, [tmp / "watch.txt"], None)
    txt = out.read_text()
    assert "| CL1 removal flagged |" in txt and "must be 0): 0" in txt, txt
    print("\nOK: plan, per-tool bar, retake, skipped steps, resume, UDP loop, results all ran (synthetic, not evidence).")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
