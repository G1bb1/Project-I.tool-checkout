"""PoC run (Nov 20): the hybrid only, driven by the watch, with a trial plan so every
sign-off is scored against what is really on the board.

One TRIAL = one watch job:

  removal  (CL1)  1. checkout, full kit        -> CHECKOUT OK
                  2. remove the named tool, sign off   -> counted as a REMOVAL check (CL1)
                  3. put it back, sign off again       -> counted as a COMPLETE-KIT check (CL3)
  complete (CL3)  1. checkout, full kit
                  2. do the perturbation (lift / nudge / rotate / hand shadow / lamp / no change),
                     sign off                          -> counted as a COMPLETE-KIT check (CL3)
  swap (CL2, optional) 1. checkout  2. swap two look-alikes, sign off -> counted as a SWAP check

Every counted sign-off is also timed (CL4) and goes through the watch (CL8).
A retake is not counted: sign off again; the final answer of that step is what counts
(after the last retake the result stands, as in `ai_main.py hybrid`).

Default 90 removals + 210 complete trials = 90 removal checks + 300 complete-kit checks
(90 "tool returned" + 210 perturbation), 390 timed checks through the watch.
The 60/60 timeout test (link cut) is separate: the server is OFF for it.

Files: data/poc-<stamp>/
  plan.json         the frozen plan (seed, every trial)
  poc_session.json  model, frozen settings, thresholds, conditions, freeze hashes
  poc_checks.csv    ONE ROW PER COUNTED CHECK with every criterion it covers
  progress.json     where to resume
  part<k>/          one folder per run (resume = new part): watch_log.csv + photos
"""
import csv
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

from .common import EMPTY, ROOT, load_json, save_json, slot_truth
from .watch_link import LiveSource, WatchLink, short_label

DATA = ROOT / "data"
TIME_LIMIT_S = 3.0

CHECK_FIELDS = [
    "timestamp", "part", "trial", "step", "block", "phase", "tool", "perturbation", "criteria",
    "job", "msg_id", "presses", "status", "truth_should_flag", "H_flagged", "H_labels",
    "miss", "false_alarm", "named_correctly", "A_flagged", "A_miss", "A_false_alarm",
    "retakes", "retake_category", "retake_exhausted", "lighting_override",
    "capture_s", "A_s", "AI_s", "server_total_s", "within_3s", "AI_contents", "note",
]


# ------------------------------------------------------------------ keep the PC awake
class KeepAwake:
    """Stops the PC sleeping while the run is open (sleep = the watch gets no answer).
    Windows: SetThreadExecutionState. macOS: caffeinate. Linux: nothing (a note is printed)."""

    def __init__(self):
        self.proc = None
        self.on = False

    def __enter__(self):
        sysname = platform.system()
        try:
            if sysname == "Windows":
                import ctypes
                ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x1, 0x2
                self.on = bool(ctypes.windll.kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED))
            elif sysname == "Darwin":
                self.proc = subprocess.Popen(["caffeinate", "-dims", "-w", str(os.getpid())])
                self.on = True
        except Exception as e:                       # never stop the run for this
            print(f"  (keep-awake not available: {e!r})")
        print("Keep-awake: " + ("ON, the PC will not sleep while this run is open." if self.on else
                                "NOT available here. Set Power & sleep to Never for this run."))
        print("  Closing the laptop lid can still sleep it; leave it open or set 'lid close = do nothing'.")
        return self

    def __exit__(self, *exc):
        try:
            if platform.system() == "Windows" and self.on:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            if self.proc:
                self.proc.terminate()
        except Exception:
            pass
        return False


def local_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        ips.update(i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    return sorted(i for i in ips if not i.startswith("127."))


# ------------------------------------------------------------------ freeze
def file_hash(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def freeze_state(model_dir):
    """What must not change between `freeze` and the end of the run."""
    from .common import CONFIG
    cur = ROOT / "refs" / "current.json"
    refs_dir = load_json(cur)["dir"] if cur.exists() else None
    return {
        "ai.json": file_hash(CONFIG / "ai.json"),
        "settings.json": file_hash(CONFIG / "settings.json"),
        "slots.json": file_hash(CONFIG / "slots.json"),
        "refs_dir": refs_dir,
        "thresholds.json": file_hash(ROOT / "refs" / refs_dir / "thresholds.json") if refs_dir else None,
        "model": Path(model_dir).name,
    }


def freeze_diff(frozen, now):
    return [k for k in frozen if k not in ("frozen_at",) and frozen.get(k) != now.get(k)]


# ------------------------------------------------------------------ the plan
def build_plan(tools, kit, ai_cfg, n_removal, n_complete, n_swap=0, seed=None):
    absent = set(ai_cfg.get("absent_tools", []))
    use = [t for t in tools if t not in absent]
    seed = seed if seed is not None else random.randrange(10 ** 6)
    rng = random.Random(seed)

    removal_tools = []
    while len(removal_tools) < n_removal:            # every tool once per round, shuffled
        r = use[:]
        rng.shuffle(r)
        removal_tools += r
    removal_tools = removal_tools[:n_removal]

    perts = list(kit.get("perturbations") or ["No change (repeat check of an untouched complete kit)"])
    pert_list = []
    while len(pert_list) < n_complete:               # every perturbation before any repeats
        r = perts[:]
        rng.shuffle(r)
        pert_list += r
    pert_list = pert_list[:n_complete]

    pairs = [p for p in (ai_cfg.get("lookalike_pairs") or kit.get("lookalike_pairs") or [])
             if p[0] in use and p[1] in use]

    trials = []
    for t in removal_tools:
        trials.append({"block": "removal", "tool": t, "steps": [
            {"kind": "checkout"},
            {"kind": "signoff", "phase": "removed", "truth_block": "removal", "truth": [t],
             "criteria": "CL1+CL4+CL8", "do": f"REMOVE {t}"},
            {"kind": "signoff", "phase": "returned", "truth_block": "complete", "truth": [],
             "criteria": "CL3+CL4+CL8", "do": f"PUT {t} BACK in its slot"}]})
    for i in range(n_swap if pairs else 0):
        a, b = pairs[i % len(pairs)]
        trials.append({"block": "swap", "tool": f"{a} <-> {b}", "steps": [
            {"kind": "checkout"},
            {"kind": "signoff", "phase": "swapped", "truth_block": "swap", "truth": [a, b],
             "criteria": "CL2+CL4+CL8", "do": f"SWAP {a} and {b}"}]})
    for p in pert_list:
        tool = rng.choice(use)
        pt = p.format(tool=tool)
        trials.append({"block": "complete", "tool": tool if "{tool}" in p else "", "perturbation": pt, "steps": [
            {"kind": "checkout"},
            {"kind": "signoff", "phase": "perturbed", "truth_block": "complete", "truth": [],
             "criteria": "CL3+CL4+CL8", "do": pt}]})
    for i, t in enumerate(trials, 1):
        t["n"] = i
    return {"seed": seed, "counts": {"removal": n_removal, "complete_only": n_complete, "swap": n_swap},
            "absent_tools": sorted(absent), "trials": trials}


# ------------------------------------------------------------------ the source with a camera that survives sleep
class PlanSource(LiveSource):
    """Live camera. Truth = the current plan step. Reopens the camera once if a grab fails
    (after the PC slept, DirectShow often loses the webcam)."""

    def __init__(self, cam, reopen, runner):
        super().__init__(cam)
        self.reopen = reopen
        self.runner = runner

    def _grab(self):
        try:
            return self.cam.grab()
        except BaseException as e:                     # toolai's simple camera raises SystemExit
            if isinstance(e, KeyboardInterrupt):
                raise
            print(f"  camera grab failed ({e!r}); reopening the camera once")
            try:
                self.cam.close()
            except Exception:
                pass
            self.cam = self.reopen()
            return self.cam.grab()

    def checkout(self):
        img, cap_s = self._grab()
        return img, cap_s, "live"

    def signoff(self):
        img, cap_s = self._grab()
        return img, cap_s, "live"

    def truth(self):
        return self.runner.truth()


# ------------------------------------------------------------------ the link with the plan on top
class PocLink(WatchLink):
    def __init__(self, runner, *a, **kw):
        super().__init__(*a, **kw)
        self.r = runner
        self._last = None
        orig = self.w.writerow

        def keep(row):
            self._last = dict(row)
            return orig(row)
        self.w.writerow = keep
        self.presses = 0
        self.retakes = 0

    def handle(self, msg, watch_ip=""):
        r = self.r
        cmd = str(msg.get("cmd", ""))
        if r.finished():
            reply = {"id": msg.get("id"), "op": cmd, "status": "error", "reason": "PoC plan finished", "job": self.job}
            print("  plan finished: answered 'error' (watch shows NOT CONFIRMED). Ctrl+C to stop.")
            return reply
        if cmd == "checkout" and r.step()["kind"] != "checkout":
            r.skip_rest("new checkout pressed before this trial's last sign-off")
            self.checkout_img = None
        reply = super().handle(msg, watch_ip)
        st = reply["status"]
        step = r.step()
        if cmd == "checkout" and step["kind"] == "checkout":
            if st == "ready":
                self.presses = self.retakes = 0
                r.advance()
        elif cmd == "signoff" and step["kind"] == "signoff" and st in ("retake", "missing", "complete"):
            self.presses += 1
            if st == "retake":
                self.retakes += 1
            else:
                r.record(step, reply, self._last or {}, self.presses, self.retakes)
                self.presses = self.retakes = 0
                last_step = r.step_i == len(r.trial()["steps"]) - 1
                if st == "complete" and not last_step:
                    # the removed tool was NOT flagged (a miss): the job closed, nothing to return to
                    r.advance()
                    r.skip_rest("job closed by KIT CONFIRMED on the removal step (miss)")
                else:
                    r.advance()
                    if last_step:
                        self.checkout_img = None        # one trial = one job
        r.show_next(st)
        return reply


# ------------------------------------------------------------------ run state
class Runner:
    def __init__(self, sdir, plan, tools, absent, part):
        self.dir = Path(sdir)
        self.plan = plan
        self.tools = tools
        self.absent = absent
        self.part = part
        prog = load_json(self.dir / "progress.json") if (self.dir / "progress.json").exists() else {}
        self.trial_i = prog.get("trial_i", 0)
        self.step_i = prog.get("step_i", 0)
        new = not (self.dir / "poc_checks.csv").exists()
        self.f = open(self.dir / "poc_checks.csv", "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=CHECK_FIELDS)
        if new:
            self.w.writeheader()
        if self.step_i:                                  # resumed mid-trial: that job is gone
            self.skip_rest(f"run stopped mid-trial; resumed in part {part}")

    # where we are
    def finished(self):
        return self.trial_i >= len(self.plan["trials"])

    def trial(self):
        return self.plan["trials"][self.trial_i]

    def step(self):
        return self.trial()["steps"][self.step_i] if not self.finished() else {"kind": "done"}

    def _save(self):
        save_json(self.dir / "progress.json", {"trial_i": self.trial_i, "step_i": self.step_i,
                                               "updated": datetime.now().isoformat(timespec="seconds")})

    def advance(self):
        self.step_i += 1
        if self.step_i >= len(self.trial()["steps"]):
            self.trial_i += 1
            self.step_i = 0
        self._save()

    def skip_rest(self, why):
        if self.finished():
            return
        t = self.trial()
        for k in range(self.step_i, len(t["steps"])):
            s = t["steps"][k]
            if s["kind"] == "signoff":
                self._row({"trial": t["n"], "step": k, "block": t["block"], "phase": s["phase"],
                           "tool": t.get("tool", ""), "perturbation": t.get("perturbation", ""),
                           "criteria": "", "status": "skipped", "note": why})
        self.trial_i += 1
        self.step_i = 0
        self._save()
        print(f"  (trial {t['n']}: remaining step(s) not counted: {why})")

    def truth(self):
        if self.finished():
            return None
        t, s = self.trial(), self.step()
        if s["kind"] != "signoff":
            return {"trial": t["n"], "block": t["block"], "void": "0",
                    "content": slot_truth(self.tools, [], "complete", self.absent)}
        return {"trial": t["n"], "block": s["truth_block"], "void": "0",
                "content": slot_truth(self.tools, s["truth"], s["truth_block"], self.absent)}

    # one counted check
    def record(self, step, reply, log, presses, retakes):
        t = self.trial()
        content = self.truth()["content"]
        should = {x for x, c in content.items() if c != x and not (x in self.absent and c == EMPTY)}
        flagged = set(json.loads(log.get("H_flagged") or "[]"))
        labels = json.loads(log.get("H_labels") or "{}")
        a_fl = set(json.loads(log.get("A_flagged") or "[]")) - set(self.absent)
        named = int(flagged == should and all(
            (labels.get(x) == "missing") if content[x] == EMPTY else (content[x] in labels.get(x, ""))
            for x in should))
        try:
            total = float(log.get("server_total_s") or "nan")
        except ValueError:
            total = float("nan")
        self._row({
            "trial": t["n"], "step": self.step_i, "block": t["block"], "phase": step["phase"],
            "tool": t.get("tool", ""), "perturbation": t.get("perturbation", ""), "criteria": step["criteria"],
            "job": log.get("job", ""), "msg_id": log.get("msg_id", ""), "presses": presses,
            "status": reply["status"], "truth_should_flag": json.dumps(sorted(should)),
            "H_flagged": json.dumps(sorted(flagged)), "H_labels": json.dumps(labels),
            "miss": int(bool(should - flagged)), "false_alarm": int(bool(flagged - should)),
            "named_correctly": named,
            "A_flagged": json.dumps(sorted(a_fl)), "A_miss": int(bool(should - a_fl)),
            "A_false_alarm": int(bool(a_fl - should)),
            "retakes": retakes, "retake_category": log.get("retake_category", ""),
            "retake_exhausted": log.get("retake_exhausted", ""),
            "lighting_override": log.get("lighting_override", ""),
            "capture_s": log.get("capture_s", ""), "A_s": log.get("A_s", ""), "AI_s": log.get("AI_s", ""),
            "server_total_s": log.get("server_total_s", ""),
            "within_3s": int(total <= TIME_LIMIT_S) if total == total else "",
            "AI_contents": log.get("AI_contents", ""),
        })
        verdict = ("MISS" if should - flagged else "") + (" FALSE ALARM" if flagged - should else "")
        print(f"  -> counted: trial {t['n']} {step['phase']} ({step['criteria']}): {reply['status']}"
              f"{'  ** ' + verdict.strip() + ' **' if verdict else '  ok'}  [{total:.2f} s]")

    def _row(self, d):
        d = {"timestamp": datetime.now().isoformat(timespec="seconds"), "part": self.part, **d}
        self.w.writerow({k: d.get(k, "") for k in CHECK_FIELDS})
        self.f.flush()

    def close(self):
        self.f.close()

    # what the operator does next
    def counts(self):
        rows = read_checks(self.dir)
        c = {"removed": 0, "complete": 0, "swapped": 0}
        for r in rows:
            if r["status"] in ("missing", "complete"):
                c["removed" if r["phase"] == "removed" else "swapped" if r["phase"] == "swapped" else "complete"] += 1
        return c

    def show_next(self, last_status=None):
        if self.finished():
            print("\n=== PLAN FINISHED. Ctrl+C to stop the server, then menu 10 (results). ===\n")
            return
        t, s = self.trial(), self.step()
        c = self.counts()
        n = len(self.plan["trials"])
        k = self.plan["counts"]
        head = (f"=== Trial {t['n']}/{n}  {t['block'].upper()}   counted so far: removals {c['removed']}/"
                f"{k['removal']}, complete-kit {c['complete']}/{k['removal'] + k['complete_only']}"
                + (f", swaps {c['swapped']}/{k['swap']}" if k.get("swap") else "") + " ===")
        print("\n" + head)
        for i, st in enumerate(t["steps"]):
            arrow = "->" if i == self.step_i else "  "
            done = "done" if i < self.step_i else ""
            if st["kind"] == "checkout":
                txt = "Full kit on the board. CHECKOUT on the watch"
            else:
                txt = f"{st['do']}, then SIGN OFF on the watch"
            print(f"  {arrow} {i + 1}. {txt}  {done}")
        if s["kind"] == "checkout":
            if last_status in ("ready", "missing", "retake"):
                print("     The watch still thinks a job is open: HOLD the button 2 s for the checkout.")
            else:
                print("     Short press = checkout (or hold 2 s).")
        elif last_status == "retake":
            print("     RETAKE: change nothing, sign off again (short press).")


def read_checks(sdir):
    p = Path(sdir) / "poc_checks.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ server loop that survives sleep
def serve_poc(link, src, port, on_wake=None, stop_after=None):
    def bind():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("0.0.0.0", port))
        s.settimeout(0.5)
        return s

    sock = bind()
    print(f"Listening on UDP {port}. This PC: {', '.join(local_ips()) or '?'} (the sketch's PC_IP must be one "
          f"of these). Ctrl+C to stop; menu 8 resumes.")
    t_end = None if stop_after is None else time.time() + stop_after
    last = time.time()
    try:
        while t_end is None or time.time() < t_end:
            now = time.time()
            if now - last > 20:                     # the loop ticks every 0.5 s: a gap = the PC slept
                print(f"\n!! The PC was asleep or frozen for {now - last:.0f} s. Rebinding the socket"
                      f" and reopening the camera. Watch checks during that time showed NOT CONFIRMED.")
                try:
                    sock.close()
                except OSError:
                    pass
                sock = bind()
                if on_wake:
                    on_wake()
                print(f"   This PC now: {', '.join(local_ips()) or '?'} (if it changed, the watch can't reach it).")
            last = time.time()
            try:
                data, addr = sock.recvfrom(1024)
            except socket.timeout:
                continue
            except ConnectionResetError:             # Windows: ICMP port-unreachable from an old send
                continue
            except OSError as e:
                print(f"  socket error {e!r}; rebinding in 2 s")
                time.sleep(2)
                try:
                    sock.close()
                except OSError:
                    pass
                sock = bind()
                continue
            try:
                msg = json.loads(data.decode("utf-8"))
                if not isinstance(msg, dict):
                    raise ValueError
            except ValueError:
                continue
            reply = link.handle(msg, addr[0])
            try:
                sock.sendto(json.dumps(reply, separators=(",", ":")).encode("utf-8"), addr)
            except OSError as e:
                print(f"  could not send the answer ({e!r}); the watch will show NOT CONFIRMED")
            print(f"  [{reply['op']} id {reply['id']} job {reply['job']}: {reply['status']}"
                  + (f" {reply['tools']}" if reply.get("tools") else "")
                  + (f" ({reply['reason']})" if reply.get("reason") else "") + "]")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        link.close()


# ------------------------------------------------------------------ results
def _bound(fails, n):
    if n == 0:
        return "n/a"
    try:
        from .common import upper_bound
        return f"{upper_bound(fails, n):.1%}"
    except Exception:
        return f"{3 / n:.1%} (rule of three)" if fails == 0 else "n/a"


def _lat(xs):
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return "n/a"
    import statistics
    p95 = xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]
    return f"median {statistics.median(xs):.2f} s, 95th pct {p95:.2f} s, max {xs[-1]:.2f} s"


def summarise(sdir, serial_logs=(), timeout_log=None):
    sdir = Path(sdir)
    info = load_json(sdir / "poc_session.json")
    plan = load_json(sdir / "plan.json")
    rows = read_checks(sdir)
    counted = [r for r in rows if r["status"] in ("missing", "complete")]
    skipped = [r for r in rows if r["status"] == "skipped"]
    rem = [r for r in counted if r["phase"] == "removed"]
    comp = [r for r in counted if r["phase"] in ("returned", "perturbed")]
    swp = [r for r in counted if r["phase"] == "swapped"]
    k = plan["counts"]
    L = [f"# PoC run {sdir.name}", "",
         f"Model {info.get('model')} (trained on {info.get('model_train_sessions')}); refs {info.get('freeze', {}).get('refs_dir')}; "
         f"generated {datetime.now():%Y-%m-%d %H:%M}. Plan: {k['removal']} removal trials + {k['complete_only']} "
         f"complete-only trials" + (f" + {k['swap']} swaps" if k.get("swap") else "") + f", seed {plan['seed']}.",
         f"Frozen settings unchanged during the run: {'yes' if not info.get('freeze_changed') else 'NO: ' + str(info['freeze_changed'])}.",
         "", "Measured on the bench; the model never saw these photos in training.", "",
         "| Claim | Requirement | Checks | Failures | 95% upper bound | Pass rule | Result |", "|---|---|---|---|---|---|---|"]
    m = sum(int(r["miss"]) for r in rem)
    L.append(f"| CL1 removal flagged | REQ-TR-02 | {len(rem)}/{k['removal']} | {m} misses | {_bound(m, len(rem))} | "
             f"0 misses in 90 | {'PASS' if m == 0 and len(rem) >= 90 else 'FAIL' if m else 'not enough checks yet'} |")
    fa = sum(int(r["false_alarm"]) for r in comp)
    need = k["removal"] + k["complete_only"]
    L.append(f"| CL3 complete kit not flagged | REQ-TR-06 | {len(comp)}/{need} | {fa} false alarms | {_bound(fa, len(comp))} | "
             f"0 in 300 | {'PASS' if fa == 0 and len(comp) >= 300 else 'FAIL' if fa else 'not enough checks yet'} |")
    if swp:
        sm = sum(int(r["miss"]) for r in swp)
        sn = sum(int(r["named_correctly"]) for r in swp)
        L.append(f"| CL2 look-alike swap | REQ-TR-02/01 | {len(swp)} | {sm} not detected; {len(swp) - sn} not named | "
                 f"{_bound(sm, len(swp))} | report | - |")
    tim = []
    for r in counted:
        try:
            tim.append(float(r["server_total_s"]))
        except ValueError:
            pass
    over = sum(1 for x in tim if x > TIME_LIMIT_S)
    L.append(f"| CL4 time <= 3 s (PC, capture to result) | REQ-TR-07 | {len(tim)} | {over} over 3 s | - | "
             f"none over 3 s in 90 | {'PASS' if over == 0 and len(tim) >= 90 else 'FAIL' if over else 'not enough checks yet'} |")
    L += ["", f"Skipped steps (not counted, listed so nothing disappears): {len(skipped)}"
          + "".join(f"\n- trial {r['trial']} {r['phase']}: {r['note']}" for r in skipped[:30]), ""]

    L += ["## Detail", "",
          f"- Removal checks named correctly ('<tool>: missing' and nothing else): "
          f"{sum(int(r['named_correctly']) for r in rem)}/{len(rem)}; extra slots flagged on a removal: "
          f"{sum(int(r['false_alarm']) for r in rem)}.",
          f"- Complete-kit checks: {sum(r['phase'] == 'returned' for r in comp)} after returning the removed tool "
          f"({sum(int(r['false_alarm']) for r in comp if r['phase'] == 'returned')} false alarms), "
          f"{sum(r['phase'] == 'perturbed' for r in comp)} perturbation trials "
          f"({sum(int(r['false_alarm']) for r in comp if r['phase'] == 'perturbed')} false alarms)."]
    by = {}
    for r in comp:
        if r["phase"] == "perturbed":
            key = r["perturbation"].split(" (")[0]
            for t in plan_tools(plan):
                key = key.replace(t, "<tool>")
            by.setdefault(key, []).append(r)
    for key, rs in sorted(by.items()):
        L.append(f"  - {key}: {len(rs)} checks, {sum(int(x['false_alarm']) for x in rs)} false alarms, "
                 f"{sum(int(x['retakes'] or 0) > 0 for x in rs)} needed a retake, "
                 f"{sum(x['lighting_override'] == '1' for x in rs)} lighting overrides")
    rt = [r for r in counted if int(r["retakes"] or 0) > 0]
    cats = {}
    for r in counted:
        for c in filter(None, (r["retake_category"] or "").split("+")):
            cats[c] = cats.get(c, 0) + 1
    L += [f"- Retakes (CL5): {sum(int(r['retakes'] or 0) for r in counted)} retake answers over {len(rt)} checks; "
          f"retakes used up (result stood): {sum(r['retake_exhausted'] == '1' for r in counted)}; final-answer "
          f"retake categories {cats or 'none'}.",
          f"- Method A alone on the same photos (for comparison only): {sum(int(r['A_miss']) for r in rem)} removal "
          f"misses, {sum(int(r['A_false_alarm']) for r in comp)} complete-kit false alarms.",
          f"- Time per counted check (server: capture + Method A + model + decision): {_lat(tim)}."]
    fl = {}
    for r in comp:
        if r["false_alarm"] == "1":
            for s, lab in json.loads(r["H_labels"] or "{}").items():
                fl.setdefault(s, []).append(lab.split(" (")[0])
    if fl:
        L.append("- False alarms by slot: " + "; ".join(f"{s} {len(v)} ({', '.join(sorted(set(v)))})"
                                                        for s, v in sorted(fl.items(), key=lambda x: -len(x[1]))))
    L.append("")

    # watch side
    if serial_logs:
        srv = {}
        for wl in sorted(sdir.glob("part*/watch_log.csv")):
            with open(wl, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    srv[(wl.parent.name, r["msg_id"])] = r
        latest = {}
        for (part, mid), r in sorted(srv.items()):
            latest[mid] = r
        watch = []
        for p in serial_logs:
            watch += read_serial(p)
        want = {"ready": "checkout_ok", "refused": "checkout_refused", "complete": "confirmed",
                "missing": "missing", "retake": "retake", "no_checkout": "no_checkout"}
        so = [w for w in watch if w["op"] == "signoff"]
        lost = [w for w in watch if w["outcome"] in ("not_confirmed_timeout", "checkout_timeout")]
        bad = [w for w in watch if w["id"] in latest and w["outcome"] not in ("not_confirmed_timeout", "checkout_timeout")
               and w["outcome"] != want.get(latest[w["id"]]["status"], "not_confirmed_bad_reply")]
        unsafe = [w for w in watch if w["outcome"] == "confirmed"
                  and latest.get(w["id"], {}).get("status") != "complete"]
        ans = [w["latency_ms"] / 1000 for w in so if w["outcome"] not in ("not_confirmed_timeout",)
               and w["latency_ms"] is not None]
        L += ["## Watch (CL8), from the watch's own log", "",
              f"- Messages logged by the watch: {len(watch)} ({len(so)} sign-offs).",
              f"- Watch screen = PC answer: {len(watch) - len(lost) - len(bad)}/{len(watch) - len(lost)} answered; "
              f"mismatches {len(bad)} {[(w['id'], w['outcome']) for w in bad[:8]]}.",
              f"- KIT CONFIRMED without a 'complete' answer (must be 0): {len(unsafe)}.",
              f"- No answer within 10 s -> NOT CONFIRMED shown: {len(lost)} (safe; count them, e.g. PC asleep).",
              f"- End-to-end sign-off time, button to screen: {_lat(ans)}; over 3 s: {sum(1 for x in ans if x > 3)}.",
              "- Ids matched across parts by number: if the watch was restarted mid-run, its ids restart too; "
              "use each part's `python watch_main.py report` for an exact match.", ""]
    if timeout_log:
        w = [x for x in read_serial(timeout_log) if x["op"] == "signoff"]
        to = [x for x in w if x["outcome"] == "not_confirmed_timeout"]
        ok = len(w) >= 60 and len(to) == len(w)
        L += ["## Link cut, 60/60 timeout (CL8, REQ-TR-04)", "",
              f"- {len(to)}/{len(w)} sign-offs showed NOT CONFIRMED after the 10 s limit; other outcomes "
              f"{len(w) - len(to)} -> **{'PASS' if ok else 'FAIL' if len(to) != len(w) else 'not enough checks yet'}**.", ""]
    else:
        L += ["## Link cut, 60/60 timeout", "", "- Not included: give the watch log of the a60 run (menu 9).", ""]
    (sdir / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"-> {sdir / 'summary.md'}")
    return sdir / "summary.md"


def plan_tools(plan):
    return sorted({t["tool"] for t in plan["trials"] if t.get("tool") and "<->" not in t["tool"]}, key=len, reverse=True)


def read_serial(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("LOG,") or line.startswith("LOG,id,"):
            continue
        p = line.split(",", 6)
        if len(p) < 6:
            continue
        rows.append({"id": p[1], "op": p[2], "outcome": p[4],
                     "latency_ms": int(p[5]) if p[5].lstrip("-").isdigit() else None,
                     "detail": p[6] if len(p) > 6 else ""})
    return rows
