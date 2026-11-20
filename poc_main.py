"""PoC run, hybrid only (Nov 20). Press Run in PyCharm and pick a number.

Order (same as before):  1 capture-refs  2 calibrate  3 check  4 collect dataset
(5 import old trials, optional)  6 train  7 freeze  8 PoC run with the watch
9 timeout test 60/60 (server OFF)  10 results (90 / 300 / 60 table)

Command line works too:
    python poc_main.py run --removal 90 --complete 210 [--swap 0]
    python poc_main.py run --resume data/poc-<stamp>
    python poc_main.py results data/poc-<stamp> --serial watch_part1.txt watch_part2.txt --timeout watch_timeout.txt
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from toolai.common import CONFIG, ROOT, load_ai_config, load_json, load_slots, load_toolcheck_settings, save_json

FROZEN = ROOT / "data" / "poc_frozen.json"


def _run(args):
    print(f"\n> python {' '.join(args)}\n")
    r = subprocess.call([sys.executable] + args, cwd=ROOT)
    if r:
        print(f"(exit code {r})")


def _thresholds():
    cur = ROOT / "refs" / "current.json"
    if not cur.exists():
        raise SystemExit("No refs/current.json. Run 1 (capture-refs) and 2 (calibrate) first.")
    p = ROOT / "refs" / load_json(cur)["dir"] / "thresholds.json"
    if not p.exists():
        raise SystemExit("No thresholds for the current references. Run 2 (calibrate).")
    return load_json(p)


def cmd_freeze(a):
    from toolai.infer import resolve_model_dir
    from toolai.poc import freeze_state
    mdir = resolve_model_dir(a.model)
    st = freeze_state(mdir)
    cfg = load_ai_config()
    d = cfg["decision"]
    cam = load_toolcheck_settings().get("camera", {})
    print("\nFreezing these for the whole PoC run:")
    print(f"  model            {st['model']}  (trained on {load_json(Path(mdir) / 'meta.json').get('train_sessions') if (Path(mdir) / 'meta.json').exists() else '?'})")
    print(f"  refs             {st['refs_dir']}")
    print(f"  camera           exposure_mode {cam.get('exposure_mode')}, exposure {cam.get('exposure')}, "
          f"gain {cam.get('gain')}, wb_temperature {cam.get('wb_temperature')}")
    print(f"  min_confidence   {d['min_confidence']}")
    print(f"  clear bar        {d.get('clear_confidence_by_tool', {})}, every other tool "
          f"{d.get('clear_confidence_default', d['clear_confidence'])}")
    print(f"  lighting override {cfg['lighting_override']}")
    print(f"  absent tools     {cfg.get('absent_tools')}")
    if cam.get("exposure_mode") != "fixed" or cam.get("wb_temperature") is None:
        print("  WARNING: camera is not exposure_mode 'fixed' with a wb_temperature number (FROZEN_SETTINGS_Nov20.md).")
    if cfg.get("test_mode", {}).get("allow_force_accept"):
        raise SystemExit("ai.json test_mode.allow_force_accept is true. Set it to false first.")
    if input("Freeze? y/n [y]: ").strip().lower() in ("", "y"):
        save_json(FROZEN, {**st, "frozen_at": datetime.now().isoformat(timespec="seconds")})
        print(f"Frozen -> {FROZEN}. Write the time and these values in the logbook.")


def cmd_run(a):
    from toolai.board import current_refs_empty
    from toolai.camera import open_camera
    from toolai.common import load_camera_settings
    from toolai.infer import Predictor, resolve_model_dir
    from toolai.poc import (KeepAwake, PlanSource, PocLink, Runner, build_plan, freeze_diff, freeze_state,
                            serve_poc)
    cfg = load_ai_config()
    if cfg.get("test_mode", {}).get("allow_force_accept"):
        raise SystemExit("ai.json test_mode.allow_force_accept is true. It must be false for the PoC run.")
    if not FROZEN.exists():
        raise SystemExit("Nothing frozen yet. Run 7 (freeze) first.")
    frozen = load_json(FROZEN)
    slots, thr, settings = load_slots(), _thresholds(), load_toolcheck_settings()
    tools = [s["tool"] for s in slots]

    if a.resume:
        sdir = Path(a.resume)
        sdir = sdir if sdir.exists() else ROOT / "data" / a.resume
        info = load_json(sdir / "poc_session.json")
        plan = load_json(sdir / "plan.json")
        frozen = info["freeze"]
        model = frozen["model"]
    else:
        model = a.model or frozen["model"]
    pred = Predictor(resolve_model_dir(model), a.int8)
    now = freeze_state(pred.dir)
    changed = freeze_diff(frozen, now)
    if changed:
        print(f"\n!! Changed since freeze: {changed}.")
        print("   Rule: a change after freezing = dated logbook entry + capture-refs + calibrate + freeze + NEW run.")
        if input("   Type CONTINUE to run anyway (it is recorded in poc_session.json): ").strip() != "CONTINUE":
            return

    pred.use_board(slots, current_refs_empty(), "current refs")
    cam_cfg = load_camera_settings()
    cam = open_camera(cam_cfg)
    img, _ = cam.grab()
    mm = pred.board.ref_mismatch(img) if pred.board is not None else None
    lim = cfg["board"].get("tool_only", {}).get("max_ref_mismatch", 0.05)
    if mm is not None:
        print(f"Empty-board reference check: {mm:.1%} of the bare paper no longer matches (limit {lim:.0%}).")
        if mm > lim and input("WARNING: run capture-refs + calibrate first? y = stop | Enter = continue: "
                              ).strip().lower() == "y":
            cam.close()
            return

    if not a.resume:
        kit = load_json(CONFIG / "kit.json") if (CONFIG / "kit.json").exists() else {}
        plan = build_plan(tools, kit, cfg, a.removal, a.complete, a.swap, a.seed)
        sdir = ROOT / "data" / f"poc-{datetime.now():%Y%m%d-%H%M%S}"
        sdir.mkdir(parents=True)
        save_json(sdir / "plan.json", plan)
        print("\nTest conditions (saved in poc_session.json):")
        conditions = {k: input(f"  {q}: ").strip() for k, q in (
            ("lighting", "Lighting"), ("operator", "Operator"), ("logbook_entry", "Logbook entry"))}
        info = {"started": datetime.now().isoformat(timespec="seconds"), "conditions": conditions,
                "model": pred.dir.name, "model_file": pred.model_file.name,
                "model_train_sessions": pred.meta.get("train_sessions"), "freeze": frozen,
                "freeze_changed": changed, "ai_config": cfg, "thresholds": thr, "settings": settings,
                "slots": slots, "camera_locked": getattr(cam, "locked", {}), "parts": []}
    part = len(info.get("parts", [])) + 1
    info.setdefault("parts", []).append({"part": part, "started": datetime.now().isoformat(timespec="seconds"),
                                         "freeze_changed": changed, "camera_locked": getattr(cam, "locked", {})})
    if changed:
        info["freeze_changed"] = sorted(set(info.get("freeze_changed") or []) | set(changed))
    save_json(sdir / "poc_session.json", info)

    pdir = sdir / f"part{part}"
    pdir.mkdir(exist_ok=True)
    runner = Runner(sdir, plan, tools, plan.get("absent_tools", []), part)

    def reopen():
        c = open_camera(cam_cfg)
        return c

    src = PlanSource(cam, reopen, runner)

    def on_wake():
        try:
            src.cam.close()
        except Exception:
            pass
        src.cam = reopen()
        im, _ = src.cam.grab()
        m = pred.board.ref_mismatch(im) if pred.board is not None else None
        if m is not None:
            print(f"   After wake: {m:.1%} of the bare paper differs from the empty-board photo (limit {lim:.0%})."
                  + ("  ** Camera settings may have reset: STOP, recapture refs + calibrate. **" if m > lim else ""))

    link = PocLink(runner, src, slots, thr, settings, pred, cfg, pdir)
    port = a.port or int(settings.get("watch_server", {}).get("listen_port", 4211))
    print(f"\nPoC run {sdir.name}, part {part}: {len(plan['trials'])} trials, model {pred.dir.name}.")
    print(f"Absent tools: {link.absent or 'none'}; {link.total} tools counted; max retakes {link.max_retakes}.")
    print("Start the watch's Serial Monitor now and save its text at the end as "
          f"watch_{sdir.name}_part{part}.txt (menu 10 reads it).")
    runner.show_next()
    with KeepAwake():
        try:
            serve_poc(link, src, port, on_wake=on_wake)
        finally:
            runner.close()
            src.close()
    done = sum(1 for _ in open(sdir / "poc_checks.csv", encoding="utf-8")) - 1
    print(f"\nStopped. {done} rows in {sdir / 'poc_checks.csv'}. Menu 8 + 'r' resumes; menu 10 for results.")


def cmd_results(a):
    from toolai.poc import summarise
    d = Path(a.session)
    summarise(d if d.exists() else ROOT / "data" / a.session, a.serial or (), a.timeout)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--model", default=None)
    f.set_defaults(fn=cmd_freeze)
    r = sub.add_parser("run")
    r.add_argument("--removal", type=int, default=90)
    r.add_argument("--complete", type=int, default=210, help="complete-only trials (the 90 returns add 90 more)")
    r.add_argument("--swap", type=int, default=0)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--resume", default=None, help="data/poc-<stamp> to continue")
    r.add_argument("--model", default=None)
    r.add_argument("--int8", action="store_true")
    r.add_argument("--port", type=int, default=None)
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("results")
    s.add_argument("session")
    s.add_argument("--serial", nargs="*", default=None, help="watch Serial Monitor text of each part")
    s.add_argument("--timeout", default=None, help="watch Serial Monitor text of the a60 run (server off)")
    s.set_defaults(fn=cmd_results)
    argv = sys.argv[1:] or menu()
    if not argv:
        return
    a = p.parse_args(argv)
    a.fn(a)


def _latest_poc():
    d = ROOT / "data"
    ps = sorted(x.name for x in d.iterdir() if x.name.startswith("poc-")) if d.exists() else []
    return ps[-1] if ps else ""


def menu():
    steps = [
        ("capture-refs", "toolcheck: empty board + full board photos (after ANY camera change)"),
        ("calibrate", "toolcheck: freeze Method A thresholds"),
        ("check", "one live check, every slot (camera + refs + model look right?)"),
        ("collect", "get dataset: labelled captures at today's camera settings"),
        ("import-trials", "optional: add old toolcheck sessions to the dataset"),
        ("train", "train the CNN on ALL dataset sessions (the PoC run is the held-out test)"),
        ("freeze", "record model + refs + settings; the run refuses silent changes"),
        ("run", "PoC run with the watch (new, or resume)"),
        ("timeout", "60/60 timeout test with the link cut (server OFF)"),
        ("results", "90 / 300 / 60 table -> summary.md"),
    ]
    print("PoC run, hybrid only: choose a step")
    for i, (c, d) in enumerate(steps, 1):
        print(f"  {i:2d}. {c:<14} {d}")
    ch = input("Number: ").strip()
    if not ch.isdigit() or not 1 <= int(ch) <= len(steps):
        raise SystemExit("No valid choice.")
    c = steps[int(ch) - 1][0]
    if c in ("capture-refs", "calibrate"):
        _run(["main.py", c])
        return []
    if c == "check":
        _run(["ai_main.py", "check"])
        return []
    if c == "collect":
        _run(["ai_main.py", "collect", "--n", input("  Captures [60]: ").strip() or "60"])
        return []
    if c == "import-trials":
        _run(["ai_main.py", "import-trials"] + input("  data/<session> folders (space-separated, Enter = all): ").split())
        return []
    if c == "train":
        e = input("  Epochs [ai.json value]: ").strip()
        _run(["ai_main.py", "train", "--type", "cnn", "--test"] + (["--epochs", e] if e else []))
        return []
    if c == "freeze":
        return ["freeze"]
    if c == "run":
        last = _latest_poc()
        if last and input(f"  Resume {last}? r = resume | Enter = new run: ").strip().lower() == "r":
            return ["run", "--resume", last]
        rm = input("  Removal trials [90]: ").strip() or "90"
        co = input("  Complete-only trials [210] (+ one 'returned' check per removal): ").strip() or "210"
        sw = input("  Look-alike swap trials [0]: ").strip() or "0"
        return ["run", "--removal", rm, "--complete", co, "--swap", sw]
    if c == "timeout":
        print("\n  1. Stop the server (Ctrl+C in the run) or switch the PC's Wi-Fi off. The watch stays on Wi-Fi.\n"
              "  2. Serial Monitor: type a60 + Enter (60 sign-offs, one every 12 s, about 12 min).\n"
              "  3. Save the Serial Monitor text as watch_timeout_<date>.txt next to poc_main.py.\n"
              "  4. Menu 10 (results) asks for that file.")
        return []
    if c == "results":
        last = _latest_poc()
        s = input(f"  PoC session [{last}]: ").strip() or last
        ser = input("  Watch logs of the run parts (space-separated .txt, Enter = none): ").split()
        to = input("  Watch log of the 60/60 timeout run (Enter = none): ").strip()
        return ["results", s] + (["--serial"] + ser if ser else []) + (["--timeout", to] if to else [])
    return []


if __name__ == "__main__":
    main()
