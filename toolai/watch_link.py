"""Watch link for the hybrid (Method A + model): the watch starts a CHECKOUT and a
SIGN-OFF, the PC/processor answers with the status of the board.

Messages (UDP, one flat JSON object each way, port watch_server.listen_port, default 4211):

  watch -> PC   {"cmd": "checkout", "id": N}
                {"cmd": "signoff",  "id": N}
                {"cmd": "status",   "id": N}       (no photo: is a job open?)

  PC -> watch   {"id": N, "op": "<cmd>", "status": ..., "tools": [...], "reason": "...",
                 "present": P, "total": T, "job": J}

  checkout  status "ready"     photo stored as this job's checkout (the Method A baseline)
            status "refused"   a slot does not hold its own tool; tools = which, nothing stored
  signoff   status "complete"  every slot holds its own tool -> job closed
            status "missing"   tools = "<slot>: missing" / "<slot>: has <tool>" / "<slot>: unsure"
                               / "<slot>: changed"; job stays open (return the tool, sign off again)
            status "retake"    the retake check fired (lighting / many slots); reason says why;
                               job stays open, sign off again
            status "no_checkout"  no open job: check out first
  status    status "job_open" | "idle"
  anything else / any error -> status "error" (the watch shows NOT CONFIRMED, never a pass)

Fail-safe rules (same as the hybrid session):
  * only "complete" is a pass; the watch shows NOT CONFIRMED for a timeout (10 s),
    an unknown status, or a reply for another check id;
  * the checkout photo is only kept when the model sees every slot's own tool,
    so a swap left in from before cannot become the baseline (session 3, trials 14/16);
  * retakes follow the hybrid's retake rule and the same max_retakes; after the last
    retake the result stands, as in `ai_main.py hybrid`.

Absent tools (ai.json "absent_tools", e.g. the Flathead screwdriver while it is at school):
the slot must be EMPTY at checkout and sign-off. Empty = expected and not counted in
present/total. Anything in it = flagged.

Sources:
  live    the camera (toolcheck's Camera when importable): what the bench test uses
  replay  stored photos of a toolcheck or hybrid session, served job by job in trial
          order (checkout -> sign-off try1 -> sign-off final). Tests the LINK without the
          bench. Every replayed answer is scored against that trial's truth, but a
          session the model was trained on is not accuracy evidence (in_training=1).
"""
import csv
import inspect
import json
import socket
import time
from datetime import datetime
from pathlib import Path

import cv2

from .common import EMPTY, ROOT, load_json, save_json, slot_truth
from .hybrid import check_capture
from .infer import classify_slots

DATA = ROOT / "data"

LOG_FIELDS = [
    "timestamp", "msg_id", "watch_ip", "cmd", "job", "try", "photo",
    "capture_s", "A_s", "AI_s", "server_total_s",
    "status", "tools", "reason", "present", "total",
    "A_flagged", "A_scores", "H_flagged", "H_labels", "AI_contents",
    "retake_category", "retake_exhausted", "lighting_override",
    "truth", "should_flag", "miss", "false_alarm", "correct", "in_training", "note",
]
MAX_TOOLS_IN_REPLY = 8      # keeps a reply well under the watch's 1024-byte buffer


def short_label(slot, label):
    """Hybrid label -> one short line for the watch screen."""
    if label == "missing":
        return f"{slot}: missing"
    if label.startswith("wrong tool:"):
        return f"{slot}: has {label.split(':', 1)[1].strip()}"
    if label.startswith("unsure"):
        return f"{slot}: unsure"
    if label.startswith("changed"):
        return f"{slot}: changed"
    if label.startswith("absent slot"):
        return f"{slot}: not empty"
    return f"{slot}: {label}"


# ---------------------------------------------------------------- photo sources
class LiveSource:
    kind = "live"

    def __init__(self, cam):
        self.cam = cam

    def checkout(self):
        img, cap_s = self.cam.grab()
        return img, cap_s, "live"

    def signoff(self):
        img, cap_s = self.cam.grab()
        return img, cap_s, "live"

    def truth(self):
        return None

    def close(self):
        self.cam.close()


class ReplaySource:
    """Stored session photos in trial order. checkout() moves to the next trial."""
    kind = "replay"

    def __init__(self, session_dir):
        self.dir = Path(session_dir)
        self.sj = load_json(self.dir / "session.json")
        self.sid = self.sj.get("session", self.dir.name)
        self.tools = [s["tool"] for s in self.sj["slots"]]
        with open(self.dir / "trials.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.jobs = []
        for r in rows:
            n = int(r["trial"])
            co = self._img_path(f"t{n:03d}_checkout.png") or self._img_path("checkout_session.png")
            so = [p for p in (self._img_path(f"t{n:03d}_signoff_try1.png"),
                              self._img_path(f"t{n:03d}_signoff.png")) if p]
            if not co or not so:
                continue
            self.jobs.append({"trial": n, "block": r["block"], "truth": json.loads(r["truth"]),
                              "void": r.get("void", "0"), "checkout": co, "signoffs": so})
        self.i = -1
        self.k = 0
        print(f"Replay source {self.sid}: {len(self.jobs)} trials with photos.")

    def _img_path(self, name):
        p = self.dir / "images" / name
        return p if p.exists() else None

    def _read(self, p):
        t0 = time.perf_counter()
        img = cv2.imread(str(p))
        return img, time.perf_counter() - t0, p.name

    def checkout(self):
        self.i += 1
        self.k = 0
        if self.i >= len(self.jobs):
            return None, 0.0, "replay finished"
        return self._read(self.jobs[self.i]["checkout"])

    def signoff(self):
        if not 0 <= self.i < len(self.jobs):
            return None, 0.0, "replay finished"
        so = self.jobs[self.i]["signoffs"]
        p = so[min(self.k, len(so) - 1)]
        self.k += 1
        return self._read(p)

    def truth(self):
        if not 0 <= self.i < len(self.jobs):
            return None
        j = self.jobs[self.i]
        return {"trial": j["trial"], "block": j["block"], "void": j["void"],
                "content": slot_truth(self.tools, j["truth"], j["block"])}

    def close(self):
        pass


# ---------------------------------------------------------------- the link logic
class WatchLink:
    """Turns watch messages into board checks. No sockets here, so it can be tested alone."""

    def __init__(self, source, slots, thresholds, settings, pred, ai_cfg, out_dir,
                 save_images=True, in_training=0):
        self.src, self.slots, self.thr, self.settings = source, slots, thresholds, settings
        self.pred, self.cfg = pred, ai_cfg
        self.tools = [s["tool"] for s in slots]
        self.absent = [t for t in ai_cfg.get("absent_tools", []) if t in self.tools]
        self.total = len(self.tools) - len(self.absent)
        # toolai Nov 20+ takes `expected` (absent slot should be empty); older versions don't,
        # and _apply_absent below gives the same result either way.
        self.expected = {t: (EMPTY if t in self.absent else t) for t in self.tools}
        self._pass_expected = "expected" in inspect.signature(check_capture).parameters
        mr = ai_cfg.get("retake", {}).get("max_retakes")
        self.max_retakes = mr if mr is not None else settings.get("retake", {}).get("max_retakes", 2)
        self.out = Path(out_dir)
        self.save_images = save_images and source.kind == "live"
        if self.save_images:
            (self.out / "images").mkdir(parents=True, exist_ok=True)
        self.in_training = in_training
        self.job = 0
        self.checkout_img = None      # None = no open job
        self.tries = 0
        self.gap = None
        self.f = open(self.out / "watch_log.csv", "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=LOG_FIELDS)
        self.w.writeheader()

    def close(self):
        self.f.close()

    # -- helpers
    def _min_conf(self):
        return self.cfg["decision"]["min_confidence"]

    def _checkout_problems(self, img):
        """Slots of the checkout photo that are not as they should be (own tool; absent = empty)."""
        ai, _ = classify_slots(self.pred, img, self.slots)
        out = []
        for t, (n, c) in zip(self.tools, ai):
            if t in self.absent:
                if not (n == EMPTY and c >= self._min_conf()):
                    out.append(f"{t}: not empty" if n != EMPTY else f"{t}: unsure")
            elif n == t and c >= self._min_conf():
                continue
            elif c < self._min_conf():
                out.append(f"{t}: unsure")
            elif n == EMPTY:
                out.append(f"{t}: missing")
            else:
                out.append(f"{t}: has {n}")
        return out

    def _apply_absent(self, res):
        """Absent slots: empty = expected (unflag), anything else = flagged."""
        flagged = list(res["flagged"])
        labels = dict(res["labels"])
        for t in self.absent:
            c = res["contents"].get(t)
            conf = dict(zip(self.tools, res["ai"]))[t][1] if res.get("ai") else 1.0
            if c == EMPTY and conf >= self._min_conf():
                if t in flagged:
                    flagged.remove(t)
                labels.pop(t, None)
            else:
                if t not in flagged:
                    flagged.append(t)
                labels[t] = f"absent slot holds {c}"
        return flagged, labels

    def _score(self, flagged, truth):
        if not truth:
            return {}
        content = truth["content"]
        should = {t for t, c in content.items() if c != t and not (t in self.absent and c == EMPTY)}
        fl = set(flagged)
        return {"truth": f"trial {truth['trial']} {truth['block']}" + (" VOID" if str(truth["void"]) == "1" else ""),
                "should_flag": json.dumps(sorted(should)),
                "miss": int(bool(should - fl)), "false_alarm": int(bool(fl - should)),
                "correct": int(fl == should)}

    def _save(self, img, name):
        if self.save_images and img is not None:
            cv2.imwrite(str(self.out / "images" / name), img)

    # -- commands
    def handle(self, msg, watch_ip=""):
        t0 = time.perf_counter()
        cmd = str(msg.get("cmd", ""))
        mid = msg.get("id")
        log = {"timestamp": datetime.now().isoformat(timespec="milliseconds"), "msg_id": mid,
               "watch_ip": watch_ip, "cmd": cmd, "in_training": self.in_training}
        try:
            if cmd == "checkout":
                reply = self._checkout(log)
            elif cmd == "signoff":
                reply = self._signoff(log)
            elif cmd == "status":
                reply = {"status": "job_open" if self.checkout_img is not None else "idle"}
            else:
                reply = {"status": "error", "reason": f"unknown command '{cmd}'"}
        except Exception as e:                      # never crash the server; the watch shows NOT CONFIRMED
            reply = {"status": "error", "reason": f"server error: {type(e).__name__}"}
            log["note"] = repr(e)
            print(f"  ERROR on {cmd} {mid}: {e!r}")
        reply = {"id": mid, "op": cmd, **reply, "job": self.job}
        tools = reply.get("tools", [])
        if len(tools) > MAX_TOOLS_IN_REPLY:
            reply["tools"] = tools[:MAX_TOOLS_IN_REPLY - 1] + [f"+{len(tools) - MAX_TOOLS_IN_REPLY + 1} more"]
        log.update({"job": self.job, "status": reply["status"], "tools": json.dumps(tools),
                    "reason": reply.get("reason", ""), "present": reply.get("present", ""),
                    "total": reply.get("total", ""),
                    "server_total_s": f"{time.perf_counter() - t0:.4f}"})
        self.w.writerow(log)
        self.f.flush()
        return reply

    def _checkout(self, log):
        img, cap_s, photo = self.src.checkout()
        log.update({"photo": photo, "capture_s": f"{cap_s:.4f}"})
        if img is None:
            self.checkout_img = None
            return {"status": "error", "reason": photo}
        t1 = time.perf_counter()
        problems = self._checkout_problems(img) if self.cfg.get("checkout_check", {}).get("enabled", True) else []
        log["AI_s"] = f"{time.perf_counter() - t1:.4f}"
        truth = self.src.truth()
        if truth:
            log["truth"] = f"trial {truth['trial']} {truth['block']} (checkout)"
        if problems:
            self.checkout_img = None            # never keep a bad baseline; an open job is cancelled too
            log["note"] = "checkout refused; any open job closed"
            self._save(img, f"refused_{datetime.now():%H%M%S}_checkout.png")
            return {"status": "refused", "tools": problems, "total": self.total,
                    "present": self.total - sum(1 for p in problems if not p.endswith("not empty"))}
        if self.checkout_img is not None:
            log["note"] = f"replaced open job {self.job}"
        self.job += 1
        self.checkout_img = img
        self.tries = 0
        self.gap = None
        self._save(img, f"j{self.job:03d}_checkout.png")
        return {"status": "ready", "present": self.total, "total": self.total,
                "absent": self.absent}

    def _signoff(self, log):
        if self.checkout_img is None:
            return {"status": "no_checkout", "reason": "check out first"}
        img, cap_s, photo = self.src.signoff()
        self.tries += 1
        log.update({"photo": photo, "capture_s": f"{cap_s:.4f}", "try": self.tries})
        if img is None:
            return {"status": "error", "reason": photo}
        kw = {"expected": self.expected} if self._pass_expected else {}
        res = check_capture(img, self.checkout_img, self.slots, self.thr, self.settings,
                            self.pred, self.cfg, self.gap, **kw)
        self._save(img, f"j{self.job:03d}_signoff_try{self.tries}.png")
        flagged, labels = self._apply_absent(res)
        log.update({"A_s": f"{res['timing']['A_s']:.4f}", "AI_s": f"{res['timing']['AI_s']:.4f}",
                    "A_flagged": json.dumps(res["A"]["flagged"]),
                    "A_scores": json.dumps({t: f"s{v['s']:.3f}/c{v['c']:.3f}"
                                            for t, v in (res["A"].get("scores") or {}).items()}),
                    "H_flagged": json.dumps(flagged),
                    "H_labels": json.dumps(labels),
                    "AI_contents": json.dumps({t: f"{n}@{c:.3f}" for t, (n, c) in zip(self.tools, res["ai"])}),
                    "retake_category": res["retake_category"],
                    "lighting_override": res["lighting_override"]})
        if res["retake_reason"] and self.tries <= self.max_retakes:
            return {"status": "retake", "reason": res["retake_reason"][:60],
                    "tools": [short_label(t, labels[t]) for t in flagged]}
        log["retake_exhausted"] = int(bool(res["retake_reason"]))
        log.update(self._score(flagged, self.src.truth()))
        present = self.total - sum(1 for t in flagged if t not in self.absent)
        if flagged:
            self.tries = 0                      # next press = a new sign-off attempt, same retake rule
            return {"status": "missing", "tools": [short_label(t, labels[t]) for t in flagged],
                    "present": present, "total": self.total}
        self.checkout_img = None                # job closed
        return {"status": "complete", "present": self.total, "total": self.total}


# ---------------------------------------------------------------- the UDP server
def serve(link, port, stop_after=None):
    """Answer watch messages until Ctrl+C (or `stop_after` seconds, used by the tests)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.5)
    t_end = None if stop_after is None else time.time() + stop_after
    print(f"Watch link listening on UDP {port}. Ctrl+C to stop.  Log -> {link.out / 'watch_log.csv'}")
    try:
        while t_end is None or time.time() < t_end:
            try:
                data, addr = sock.recvfrom(1024)
            except socket.timeout:
                continue
            try:
                msg = json.loads(data.decode("utf-8"))
                if not isinstance(msg, dict):
                    raise ValueError
            except ValueError:
                print(f"  ignored non-JSON packet from {addr[0]}")
                continue
            reply = link.handle(msg, addr[0])
            sock.sendto(json.dumps(reply, separators=(",", ":")).encode("utf-8"), addr)
            print(f"  {reply['op']:<8} id {reply['id']} job {reply['job']}: {reply['status']}"
                  + (f" {reply['tools']}" if reply.get("tools") else "")
                  + (f" ({reply['reason']})" if reply.get("reason") else ""))
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        link.close()
        print(f"Watch link stopped. Log: {link.out / 'watch_log.csv'}")


def new_out_dir(tag):
    d = DATA / f"watch-{tag}-{datetime.now():%Y%m%d-%H%M%S}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_session(out, **info):
    save_json(Path(out) / "watch_session.json",
              {"started": datetime.now().isoformat(timespec="seconds"), **info})
