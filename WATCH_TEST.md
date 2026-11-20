# Watch checkout / sign-off test (CL8, CL4; REQ-TR-01, -04, -07, -08)

Written 2026-11-20. Code drafted with Claude; the test plan choices and all numbers are yours.

## What this tests

The watch starts the two steps of a job and gets the board status back:

1. **Checkout** (press once): the processor takes the checkout photo and the model checks that
   every slot holds its own tool. Watch: **CHECKOUT OK** (job number, 11/11 tools in place) or
   **NOT CHECKED OUT** + which slots are wrong. A bad checkout is never kept as the baseline.
2. **Sign-off** (press again): hybrid check against that checkout photo (Method A + model, same
   code and frozen settings as `ai_main.py hybrid`). Watch: **KIT CONFIRMED**, **MISSING** + the
   tools (`Pliers: missing`, `Wire cutter: has Nippers`, `Knife: unsure`), **RETAKE** + the reason,
   or **NO CHECKOUT**.
3. **No answer in 10 s, or any answer it can't read: NOT CONFIRMED** (white screen). Only a
   `complete` answer for the same check id shows KIT CONFIRMED.

Hold the BOOT button 2 s to force a new checkout. The Flathead screwdriver: if `ai.json` has
`"absent_tools": ["Flathead screwdriver"]` (the AI thread's new version), its slot must be empty at
checkout and sign-off, it isn't counted (11/11), and anything put in it is flagged.

**Difference from the proposal text:** the proposal says the watch compares the result with the
checkout list. Here the processor compares (the hybrid needs the checkout *photo*), and the watch
only shows the answer. That fits the Oct 16 direction (watch = indicator only). Worth one line in the
logbook.

## Files (put them in your PyCharm project)

| File | Goes to |
|---|---|
| `watch_main.py` | project root, next to `ai_main.py` |
| `toolai/watch_link.py` | `toolai/` |
| `tests/test_watch_link.py` | `tests/` (needs `tests/test_toolai.py`, already there) |
| `watch_firmware/toolcheck_watch_v2/toolcheck_watch_v2.ino` | open in Arduino IDE |

No existing file is changed, and `ai_main.py`, `ai.json` and `settings.json` stay as they are.
Works with toolai v3 and the next version from the AI thread.

## Before the first run

- `settings.json` → `"watch_server": {"listen_port": 4211}` (the old `"method"` key is not used here).
- Windows firewall: allow Python **inbound UDP 4211** (Windows asks the first time `serve` runs;
  tick *Private networks*).
- PC and watch on the **same 2.4 GHz network**. Find the PC's IP with `ipconfig`.
- In the sketch, set `WIFI_SSID`, `WIFI_PASS`, `PC_IP`. Board settings are in the sketch header.
  **The sketch has not been compiled.** I couldn't install the ESP32 toolchain in my session.
  If it fails to compile, paste me the first error.

## Step 0: synthetic self-test (1–2 min, no hardware)

    python -m tests.test_watch_link

It ends with `OK: replay link logic, live link logic, UDP round trip + report, server-off timeout all ran.`
It only shows the code runs. It is not evidence.

## Step 1: PC only, fake watch (5 min)

Terminal 1: `python watch_main.py serve --replay data/<a session with photos>`
Terminal 2: `python watch_main.py sim --jobs 10`
Then: `python watch_main.py report data/watch-replay-<stamp> --serial watch_sim_<stamp>.txt`

## Step 2: real watch, stored photos (link test, ~10 min)

1. Flash the sketch. In the Serial Monitor, check `Wi-Fi OK, watch IP ...`.
2. `python watch_main.py serve --replay data/<session>`
3. Serial Monitor: `j10`. Each job sends a checkout, then a sign-off, and repeats the sign-off on RETAKE.
   Also press the BOOT button a few times by hand and check the screens are readable.
4. Save the whole Serial Monitor text as `watch_replay_<date>.txt`, stop the server (Ctrl+C), then run
   `python watch_main.py report data/watch-replay-<stamp> --serial watch_replay_<date>.txt`

This gives the **link** latency without the camera, as the Stage 1 plan says. Replay time does not
include camera capture, so report capture time separately (from the live run or the trial sessions).
Every v3 model was trained on all sessions, so the right/wrong counts in a replay are **not** accuracy
evidence. The report says so at the top.

## Step 3: real watch, live camera on the bench (~20 min)

1. As always: `python main.py capture-refs` + `python main.py calibrate` first (same session rule as hybrid).
2. `python watch_main.py serve`. It checks the empty-board reference and asks for lighting, operator and logbook entry.
3. Do the jobs with the **watch button only**. Write the truth of each job in your logbook (the live
   server doesn't know what you removed). Suggested 10 jobs:
   - 4 × complete kit → expect KIT CONFIRMED
   - 2 × one tool removed → MISSING + that tool; return it, press again → KIT CONFIRMED
   - 1 × **two tools removed → the watch names both** (REQ-TR-01 demonstration)
   - 1 × look-alike swap (e.g. Wire stripper ↔ Wire cutter) → MISSING with `has ...`
   - 1 × checkout with a swap left in → NOT CHECKED OUT; fix, press again
   - 1 × hand shadow over the board at sign-off → RETAKE or KIT CONFIRMED (lighting override);
     log which one happened
4. Save the Serial Monitor text, Ctrl+C the server, then `python watch_main.py report data/watch-live-<stamp> --serial <file>`.

## Step 4: link cut, 60/60 timeout (REQ-TR-04, ~12 min)

1. **Stop the server** (or switch off the PC's Wi-Fi). The watch stays on Wi-Fi.
2. Serial Monitor: `a60`. That's 60 sign-offs, one every 12 s.
3. Save the text, then `python watch_main.py report --serial watch_timeout_<date>.txt --timeout-test`

## What counts as a pass (decide and freeze before running; these are suggestions)

| Check | Pass | From |
|---|---|---|
| Timeout, link cut | 60/60 NOT CONFIRMED, each ≥ 10 s | REQ-TR-04 acceptance |
| Fail-safe | 0 × KIT CONFIRMED without a `complete` answer | report "must be 0" line |
| Screen = answer | watch outcome matches the server answer for every answered check | report |
| Two tools removed | the watch names both | REQ-TR-01 acceptance |
| End-to-end time | live sign-off, tap → screen ≤ 3 s | R3 / REQ-TR-07 |
| Lost answers | count and report them (each one shows NOT CONFIRMED, which is safe) | — |

## What gets logged

- Watch: `LOG,<id>,<op>,<start_ms>,<outcome>,<latency_ms>,<detail>` in the Serial Monitor (end to end).
- Server: `data/watch-<live|replay>-<stamp>/watch_log.csv`: one row per message, with capture,
  Method A, model and total server time, the answer, A's and the hybrid's flags, the model's reading
  of each slot, retake reason, and (replay only) the trial's truth. Also `watch_session.json`
  (model, frozen thresholds, ai.json, conditions) and, live, the photos in `images/`.
- `report` writes `summary.md` in the same folder.
