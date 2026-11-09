"""Tool return confirmation: bench PoC.

Order of use (each step saves its output, so you can stop between steps):
  python main.py preview                  live view with slot boxes, check framing
  python main.py define-slots             draw one box per tool, in kit.json order
  python main.py capture-refs             stored references for Method B (full + empty)
  python main.py calibrate                freeze thresholds from complete-kit captures
  python main.py run --removal 90 --swap 30 --complete 60
  python main.py analyze <session_id>     summary tables for the report
"""
import argparse
from datetime import datetime

import cv2

from toolcheck.camera import Camera
from toolcheck.paths import (REFS, SLOTS_FILE, CURRENT_REFS_FILE, load_settings, load_kit,
                             load_slots, load_json, save_json, current_refs_dir)


def cmd_preview(args):
    s = load_settings()
    cam = Camera(s["camera"])
    slots = load_slots() if SLOTS_FILE.exists() else []
    print(f"Resolution {cam.resolution()}. Press q in the window to quit.")
    while True:
        f = cam.live_frame()
        if f is None:
            continue
        for sl in slots:
            cv2.rectangle(f, (sl["x"], sl["y"]), (sl["x"] + sl["w"], sl["y"] + sl["h"]), (0, 200, 0), 2)
            cv2.putText(f, sl["tool"][:14], (sl["x"] + 3, sl["y"] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        cv2.imshow("preview", f)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.close()
    cv2.destroyAllWindows()


def cmd_define_slots(args):
    s, kit = load_settings(), load_kit()
    cam = Camera(s["camera"])
    input("Place the COMPLETE kit on the board. Enter to capture a frame for drawing slots...")
    img, _ = cam.grab()
    cam.close()
    slots = []
    for tool in kit["tools"]:
        view = img.copy()
        for sl in slots:
            cv2.rectangle(view, (sl["x"], sl["y"]), (sl["x"] + sl["w"], sl["y"] + sl["h"]), (0, 200, 0), 2)
        cv2.putText(view, f"Draw slot for: {tool}  (drag, then Enter/Space; c = cancel)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        x, y, w, h = cv2.selectROI("define slots", view, showCrosshair=False)
        if w == 0 or h == 0:
            raise SystemExit(f"No box drawn for {tool}; nothing saved.")
        slots.append({"tool": tool, "x": int(x), "y": int(y), "w": int(w), "h": int(h)})
    cv2.destroyAllWindows()
    save_json(SLOTS_FILE, {"created": datetime.now().isoformat(timespec="seconds"),
                           "resolution": list(img.shape[1::-1]), "slots": slots})
    cv2.imwrite(str(SLOTS_FILE.with_suffix(".png")), img)
    print(f"Saved {len(slots)} slots to {SLOTS_FILE}")


def cmd_capture_refs(args):
    s = load_settings()
    cam = Camera(s["camera"])
    d = datetime.now().strftime("%Y%m%d-%H%M%S")
    rdir = REFS / d
    rdir.mkdir(parents=True, exist_ok=True)
    input("Board FULL (every tool in its slot). Enter to capture...")
    full, _ = cam.grab()
    input("Now remove EVERY tool (board empty, camera and light untouched). Enter to capture...")
    empty, _ = cam.grab()
    cam.close()
    cv2.imwrite(str(rdir / "full.png"), full)
    cv2.imwrite(str(rdir / "empty.png"), empty)
    save_json(CURRENT_REFS_FILE, {"dir": d})
    print(f"References saved in {rdir}. Put all tools back, then run: python main.py calibrate")


def _load_refs():
    rdir = current_refs_dir()
    return {"dir": rdir, "full": cv2.imread(str(rdir / "full.png")),
            "empty": cv2.imread(str(rdir / "empty.png"))}


def cmd_calibrate(args):
    from toolcheck.calibrate import run_calibration
    s, kit, slots = load_settings(), load_kit(), load_slots()
    refs = _load_refs()
    cam = Camera(s["camera"])
    try:
        run_calibration(cam, slots, refs["dir"], refs["full"], refs["empty"], s, kit)
    finally:
        cam.close()


def cmd_run(args):
    from toolcheck.trials import run_session
    s, kit, slots = load_settings(), load_kit(), load_slots()
    refs = _load_refs()
    thr_file = refs["dir"] / "thresholds.json"
    if not thr_file.exists():
        raise SystemExit("No thresholds for the current references. Run: python main.py calibrate")
    thresholds = load_json(thr_file)
    cam = Camera(s["camera"])
    try:
        run_session(cam, slots, refs, thresholds, s, kit, args.removal, args.swap, args.complete)
    finally:
        cam.close()


def cmd_analyze(args):
    from toolcheck.analyze import summarise
    summarise(args.session)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preview").set_defaults(fn=cmd_preview)
    sub.add_parser("define-slots").set_defaults(fn=cmd_define_slots)
    sub.add_parser("capture-refs").set_defaults(fn=cmd_capture_refs)
    sub.add_parser("calibrate").set_defaults(fn=cmd_calibrate)
    r = sub.add_parser("run")
    r.add_argument("--removal", type=int, default=90)
    r.add_argument("--swap", type=int, default=30)
    r.add_argument("--complete", type=int, default=60)
    r.set_defaults(fn=cmd_run)
    a = sub.add_parser("analyze")
    a.add_argument("session")
    a.set_defaults(fn=cmd_analyze)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
