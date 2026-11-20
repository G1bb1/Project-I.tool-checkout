"""AI tool recognition on the shadow board: collect -> train -> evaluate -> profile.

Run from the project root (same folder as main.py). Uses config/slots.json and the
camera settings in config/settings.json from toolcheck; AI settings are in config/ai.json.

    python ai_main.py pink                           # measure the pink shadows (once, FIRST)
    python ai_main.py import-trials data/<session> ...   # your existing trial photos -> dataset
    python ai_main.py collect --n 60                 # one labelled collection session
    python ai_main.py train --type cnn               # newest session held out for testing
    python ai_main.py train --type svm               # classical baseline to compare against
    python ai_main.py evaluate [model]               # scores the held-out session(s)
    python ai_main.py check [model]                  # one live check, prints each slot
    python ai_main.py profile [model] --n 30         # timing + memory on this machine
    python ai_main.py replay data/<session> [--model M]  # hybrid vs A on stored photos, no camera
    python ai_main.py hybrid --removal 12 --swap 10 --complete 10 [--allow-force]
    python ai_main.py hybrid-summary hybrid-<stamp>
"""
import argparse
import sys

from pathlib import Path

from toolai.common import (CONFIG, ROOT, list_sessions, load_ai_config, load_camera_settings,
                           load_json, load_slots, load_toolcheck_settings)


def cmd_import(a):
    from toolai.importer import import_trials
    dirs = a.sessions
    if not dirs:
        data = ROOT / "data"
        dirs = sorted(str(d) for d in data.iterdir() if (d / "session.json").exists()
                      and not d.name.startswith("hybrid-")) if data.exists() else []
    import_trials(dirs, [s["tool"] for s in load_slots()], load_ai_config(), a.include_void)


def cmd_pink(a):
    from toolai.pink import find_refs, run_pink
    empty, full = find_refs(ROOT)
    empty = a.empty or empty
    full = a.full or full
    if not empty:
        raise SystemExit("No empty-board photo. Give one: python ai_main.py pink --empty refs/<dir>/empty.png")
    run_pink(load_slots(), load_ai_config(), empty, full)


def cmd_replay(a):
    from toolai.infer import Predictor, resolve_model_dir
    from toolai.replay import replay
    pred = Predictor(resolve_model_dir(a.model), a.int8)
    for d in a.sessions:
        replay(d, pred, load_ai_config())


def _current_thresholds():
    cur = ROOT / "refs" / "current.json"
    if not cur.exists():
        raise SystemExit("No refs/current.json. Run toolcheck's capture-refs and calibrate first.")
    p = ROOT / "refs" / load_json(cur)["dir"] / "thresholds.json"
    if not p.exists():
        raise SystemExit("No thresholds for the current references. Run: python main.py calibrate")
    thr = load_json(p)
    if "lighting" not in thr:
        print("WARNING: thresholds.json has no 'lighting' entry, so the lighting retake check is OFF. "
              "Recalibrate with the Nov 17 patch applied.")
    return thr


def cmd_hybrid(a):
    from toolai.camera import open_camera
    from toolai.hybrid_session import run_hybrid_session
    from toolai.infer import Predictor, resolve_model_dir
    settings = load_toolcheck_settings()
    if not settings.get("retake", {}).get("enabled"):
        print("WARNING: settings.json has no enabled 'retake' block, so no retakes will be asked for.")
    kit = load_json(CONFIG / "kit.json")
    from toolai.board import current_refs_empty
    pred = Predictor(resolve_model_dir(a.model), a.int8).use_board(load_slots(), current_refs_empty(), "current refs")
    cam = open_camera(load_camera_settings())
    try:
        run_hybrid_session(cam, load_slots(), _current_thresholds(), settings, kit, pred,
                           load_ai_config(), a.removal, a.swap, a.complete, allow_force=a.allow_force)
    finally:
        cam.close()


def cmd_hybrid_summary(a):
    from toolai.summary import summarise_session
    d = Path(a.session)
    summarise_session(d if d.exists() else ROOT / "data" / a.session)


def cmd_collect(a):
    from toolai.camera import open_camera
    from toolai.collect import collect_session
    cam = open_camera(load_camera_settings())
    try:
        collect_session(cam, load_slots(), load_ai_config(), a.n, seed=a.seed)
    finally:
        cam.close()


def cmd_train(a):
    from toolai.train import train
    train(load_slots(), load_ai_config(), a.type, test_sessions=a.test,
          epochs=a.epochs, base=a.base)


def cmd_evaluate(a):
    from toolai.evaluate import evaluate
    from toolai.infer import resolve_model_dir
    evaluate(resolve_model_dir(a.model), a.sessions, a.int8)


def cmd_check(a):
    from toolai.camera import open_camera
    from toolai.infer import Predictor, check_frame, resolve_model_dir
    from toolai.board import current_refs_empty
    pred = Predictor(resolve_model_dir(a.model), a.int8).use_board(load_slots(), current_refs_empty(), "current refs")
    cam = open_camera(load_camera_settings())
    try:
        img, cap_s = cam.grab()
    finally:
        cam.close()
    res, t = check_frame(pred, img, load_slots())
    for r in res:
        unsure = r["confidence"] < load_ai_config()["decision"]["min_confidence"]
        r["flag"] = r["flag"] or unsure
        if unsure and not r["reason"]:
            r["reason"] = "unsure"
        print(f"  {'FLAG' if r['flag'] else ' ok '}  {r['slot']:<22} sees: {r['content']}  ({r['confidence']:.2f})"
              + ("  UNSURE" if unsure else ""))
    flagged = [r for r in res if r["flag"]]
    print("KIT CONFIRMED" if not flagged else
          "MISSING/WRONG: " + "; ".join(f"{r['slot']} ({r['reason']})" for r in flagged))
    print(f"capture {cap_s*1000:.0f} ms, " + ", ".join(f"{k} {v*1000:.0f} ms" for k, v in t.items()))


def cmd_profile(a):
    from toolai.camera import open_camera
    from toolai.infer import resolve_model_dir
    from toolai.profile import run_profile
    cam = open_camera(load_camera_settings())
    try:
        run_profile(cam, load_slots(), resolve_model_dir(a.model), a.n, a.int8)
    finally:
        cam.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    im = sub.add_parser("import-trials", help="turn toolcheck trial photos into labelled crops")
    im.add_argument("sessions", nargs="*", help="toolcheck session folders; default: every data/<session>")
    im.add_argument("--include-void", action="store_true")
    im.set_defaults(fn=cmd_import)

    pk = sub.add_parser("pink", help="measure the pink shadow colour from the empty-board photo")
    pk.add_argument("--empty", default=None, help="default: refs/<current>/empty.png")
    pk.add_argument("--full", default=None, help="default: refs/<current>/full.png (to check no tool is erased)")
    pk.set_defaults(fn=cmd_pink)

    rp = sub.add_parser("replay", help="run the hybrid on a stored toolcheck session (no camera)")
    rp.add_argument("sessions", nargs="+")
    rp.add_argument("--model", default=None)
    rp.add_argument("--int8", action="store_true")
    rp.set_defaults(fn=cmd_replay)

    hy = sub.add_parser("hybrid", help="live bench session: Method A + model, retake, checkout check")
    hy.add_argument("--removal", type=int, default=12)
    hy.add_argument("--swap", type=int, default=10)
    hy.add_argument("--complete", type=int, default=10)
    hy.add_argument("--model", default=None)
    hy.add_argument("--int8", action="store_true")
    hy.add_argument("--allow-force", action="store_true",
                    help="TEST ONLY: 'f' at a retake prompt force-accepts the photo (logged forced=1)")
    hy.set_defaults(fn=cmd_hybrid)

    hs = sub.add_parser("hybrid-summary", help="summary.md for a hybrid session")
    hs.add_argument("session")
    hs.set_defaults(fn=cmd_hybrid_summary)

    c = sub.add_parser("collect", help="capture one labelled dataset session")
    c.add_argument("--n", type=int, default=60, help="number of captures")
    c.add_argument("--seed", type=int, default=None)
    c.set_defaults(fn=cmd_collect)

    t = sub.add_parser("train", help="train on every session except the held-out ones")
    t.add_argument("--type", choices=["cnn", "svm"], default="cnn")
    t.add_argument("--test", nargs="*", default=None,
                   help=f"held-out session(s); default: the newest. Sessions: {', '.join(list_sessions()) or 'none yet'}")
    t.add_argument("--base", choices=["small", "mobilenet"], default="small",
                   help="cnn only: small = from scratch; mobilenet = pretrained (downloads weights)")
    t.add_argument("--epochs", type=int, default=None)
    t.set_defaults(fn=cmd_train)

    for name, fn, hlp in (("evaluate", cmd_evaluate, "score the held-out sessions"),
                          ("check", cmd_check, "one live check"),
                          ("profile", cmd_profile, "timing and memory on this machine")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("model", nargs="?", default=None, help="model folder name; default: newest")
        s.add_argument("--int8", action="store_true", help="cnn only: use model_int8.tflite")
        if name == "evaluate":
            s.add_argument("--sessions", nargs="*", default=None)
        if name == "profile":
            s.add_argument("--n", type=int, default=30)
        s.set_defaults(fn=fn)

    argv = sys.argv[1:] or menu()
    a = p.parse_args(argv)
    a.fn(a)


def menu():
    """Shown when ai_main.py is run with no command (e.g. PyCharm's Run button)."""
    steps = [("pink", "measure the pink shadows (once, FIRST)"),
             ("import-trials", "turn your toolcheck trial photos into training crops"),
             ("train", "train a model (asks cnn/svm and the held-out session)"),
             ("evaluate", "score the newest model on its held-out session"),
             ("replay", "A vs hybrid on a stored toolcheck session (no camera)"),
             ("hybrid", "live bench session (asks trial counts)"),
             ("hybrid-summary", "summary.md for a hybrid session"),
             ("check", "one live check, prints every slot"),
             ("collect", "capture one labelled dataset session"),
             ("profile", "timing and memory on this machine")]
    print("toolai: choose a step")
    for i, (cmd, desc) in enumerate(steps, 1):
        print(f"  {i:2d}. {cmd:<15} {desc}")
    choice = input("Number: ").strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(steps):
        raise SystemExit("No valid choice.")
    cmd = steps[int(choice) - 1][0]
    if cmd == "train":
        kind = input("  Model type cnn/svm [cnn]: ").strip() or "cnn"
        sess = list_sessions()
        print("  Sessions: " + (", ".join(sess) or "none yet (run import-trials first)"))
        test = input(f"  Held-out session [{sess[-1] if sess else ''}]: ").strip()
        return ["train", "--type", kind] + (["--test", test] if test else [])
    if cmd == "replay":
        data = ROOT / "data"
        tc = sorted(d.name for d in data.iterdir() if (d / "session.json").exists()
                    and not d.name.startswith("hybrid-")) if data.exists() else []
        print("  toolcheck sessions: " + (", ".join(tc) or "none in data/"))
        s = input(f"  Session [{tc[-1] if tc else ''}]: ").strip() or (tc[-1] if tc else "")
        return ["replay", str(data / s)]
    if cmd == "hybrid":
        r = input("  Removal trials [12]: ").strip() or "12"
        sw = input("  Swap trials [10]: ").strip() or "10"
        c = input("  Complete-kit trials [10]: ").strip() or "10"
        f = input("  Allow force-accept at retakes (TESTS ONLY) y/n [n]: ").strip().lower() == "y"
        return ["hybrid", "--removal", r, "--swap", sw, "--complete", c] + (["--allow-force"] if f else [])
    if cmd == "hybrid-summary":
        data = ROOT / "data"
        hs = sorted(d.name for d in data.iterdir() if d.name.startswith("hybrid-")) if data.exists() else []
        s = input(f"  Session [{hs[-1] if hs else ''}]: ").strip() or (hs[-1] if hs else "")
        return ["hybrid-summary", s]
    if cmd == "collect":
        return ["collect", "--n", input("  Captures [60]: ").strip() or "60"]
    return [cmd]


if __name__ == "__main__":
    main()
