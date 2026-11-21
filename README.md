# toolcheck + toolai: bench PoC for tool return confirmation

Camera check that every tool in a 12-tool kit is back on its pink shadow board at sign-off.

- **Method A** (`toolcheck`) compares each slot of the sign-off photo with the checkout photo
  and says which slots *changed*. Thresholds are frozen by calibration before any trial.
- **The learned slot classifier** (`toolai`) says *what* is in each slot: one of the 12 tools,
  or "empty", with a confidence from 0 to 1. It is trained on the PC and exported as
  `.tflite` for the processor (REQ-TR-05).
- **The hybrid (concept D)** combines both, slot by slot. This is the chosen design.
- **The watch link** (`watch_main.py`) lets the watch start checkout and sign-off. The
  processor (the PC as stand-in) runs the hybrid and sends back the verdict; the watch only
  displays it, and shows NOT CONFIRMED if no answer arrives within 10 s.
- **The PoC runner** (`poc_main.py`) is one menu from references to results for the full
  hybrid-only run, with the watch starting every check (section 10).
- **Method B** (stored full/empty references) stays in the code for comparison only. It is
  rejected as a design choice because it drifts after a tool is reseated (CL6 refuted).

---

## 1. Setup (PyCharm, Windows)

1. Open this folder as a project in PyCharm.
2. Settings > Project > Python Interpreter > add a new virtual environment.
3. In the PyCharm terminal:
   ```
   pip install -r requirements.txt
   pip install "opencv-python>=4.8,<5" tensorflow scikit-learn joblib scipy
   ```
   Use OpenCV 4.x: OpenCV 5 removed the HOG function the SVM baseline uses.
4. Check that everything runs without the webcam:
   - `python -m tests.test_pipeline` (should end with `session: run + analyze OK`)
   - `python -m tests.test_toolai` (about 10 min; ends with `OK: import -> ... -> summary all ran`)
   - `python -m tests.test_poc` (synthetic PoC run: plan, retake, resume, UDP loop, results)
   - `python -m tests.test_watch_link` (ends with `OK: replay link logic, live link logic, UDP round trip + report, server-off timeout all ran.`)

`python main.py`, `python ai_main.py` and `python poc_main.py` with no command (or PyCharm's Run button) show a
numbered menu of the steps below.

The synthetic tests prove only that the code runs. Their accuracy is not evidence.

---

## 2. Settings

The full list of frozen values for the Nov 20 run, with blanks for what calibration writes on
the day, is in `FROZEN_SETTINGS_Nov20.md`.

### `config/settings.json` (toolcheck)

```json
"camera": {
  "index": 0, "width": 1280, "height": 720,
  "warmup_frames": 30, "flush_frames": 5, "average_frames": 5,
  "lock_exposure": true,
  "exposure_mode": "fixed",
  "exposure": -4,
  "gain": null,
  "disable_autofocus": true
},
"calibration": { "colour_floor": 0.1, ... },
"retake": {
  "enabled": true, "max_flagged": 3, "max_retakes": 2,
  "dark_tol": 0.15, "blur_sigma": 8.0, "dark_floor": 0.02, "shift_floor": 0.05
}
```

**Camera.** DirectShow exposure is log2 seconds: −4 is about 1/16 s and −6 about 1/64 s.
- At −6 the camera makes up brightness with gain. On Nov 19 it locked at gain 190, and the
  image was noisy and the results worse. Use **−4**. (The Nov 20 PoC run used −5, with its own
  references and calibration; logbook L18.)
- `exposure_mode`:
  - `"fixed"` sets `exposure`, and also `gain` and `wb_temperature` if you give them.
  - `"settle_lock"` leaves exposure, gain and white balance on auto for `settle_s` seconds
    (default 3), then locks all three at what auto chose.
- The locked values are printed and saved in each hybrid session's `session.json`
  (`camera_locked`). After the first good session at −4, copy that `gain` into `"gain"` so
  every later session matches.
- This webcam can't report white balance (it reads back −1).

**Any camera change means new references and a new calibration first** (steps 4 and 5 below).
Without them, every slot reads as noise (Nov 19 informal check: all slots misread at 0.2–0.4).

**Retake.** The retake check runs only when the `retake` block has `"enabled": true` **and** the
current `thresholds.json` has a `"lighting"` entry (written by `calibrate`). Without either,
trials run with no retake check and no warning.

### `config/ai.json` (toolai)

Freeze every value **before** a held-out evaluation or a hybrid session. If you change one after
seeing results, write a dated logbook entry, then run a new session.

| Key | Current | Meaning |
|---|---|---|
| `pink.enabled`, `board.footprint.enabled` | false | No-pink version: the tool is found by comparing with the empty-board photo |
| `board.exposure` | on, target 215 | Each colour channel scaled so the bare paper matches in every session |
| `board.tool_only` | on | The model sees only pixels that differ from `refs/<dir>/empty.png` |
| `decision.min_confidence` | 0.60 | Below this a slot is flagged "unsure" |
| `decision.clear_confidence_by_tool` | Wire stripper, Wire cutter: 0.90 | PoC version: confidence needed to cancel A's flag for these tools (shape or colour flag) |
| `decision.clear_confidence_default` | 0.60 | PoC version: the same for every other tool. Replaces the two keys below when present |
| `decision.clear_confidence` | 0.90 | v3: confidence needed to cancel A's flag (shape flags and look-alike slots) |
| `decision.colour_only_clear_confidence` | 0.60 | v3: cancel A's flag when A flagged on colour only (non-look-alike slots) |
| `lighting_override` | `shape_ok`, 0.60 | See "How the hybrid decides" |
| `absent_tools` | `["Flathead screwdriver"]` | Tool not in the kit this session (left at school) |
| `plan.cycle_perturbations` | true | Complete-kit trials use every perturbation, including the hand shadow, before repeating one |
| `test_mode.allow_force_accept` | false | Tests only, see section 7 |

Also check that the names in `lookalike_pairs` match `config/slots.json` exactly.

---

## 3. Bench session, in order

| Step | Command | What you do |
|---|---|---|
| 1 | `python main.py preview` | Fix the camera and light. Whole board in frame. `q` to quit. If it opens the laptop camera, set `camera.index` to 1. |
| 2 | `python main.py exposure` | Bare board. `[` / `]` change exposure until it says GOOD (median 200–230, < 0.5 % clipped white). Put the value in `settings.json` (−4 on this bench). |
| 3 | `python main.py define-slots` | Only when the board or camera moved. Drag a box around each slot, in the order of `config/kit.json`. |
| 4 | `python main.py capture-refs` | Full board, then empty board. Don't touch the camera between them. An absent tool's slot stays empty in both. |
| 5 | `python main.py calibrate` | 12 complete-kit captures, lifting and replacing 3 tools each time. Read the separation check and `calibration_scores.csv` (see below) **before** trials. |
| 6 | `python ai_main.py check` | One look at the board as the model sees it. Every slot should read its own tool (an absent slot reads "empty"). If many slots read as random tools, the references don't match the camera: go back to step 4. |
| 7 | `python ai_main.py hybrid --removal 11 --swap 10 --complete 10` | The live hybrid session (section 5). Enter = accept, `v` = void (procedural mistake only), `n` = note, `q` = stop. |
| 8 | `python ai_main.py hybrid-summary hybrid-<stamp>` | Prints and saves `summary.md`: A, model alone and hybrid side by side. |

For a toolcheck-only session (A and B, no model), use `python main.py run --removal 12 --swap 10 --complete 10`
in step 7, then `python main.py analyze <session>`.

**Trial counts.**
- Use a removal count that is a multiple of the number of tools in use (11 while the
  screwdriver is absent), so every tool is removed equally often.
- Full run (PoC pass mark and REQ-TR-06): 90 removals first, then 300 complete-kit checks.
- A short run (12/10/10) still gives honest numbers with wider bounds. Report n and the bound.

**Reading the calibration.**
- The separation table's columns are: removal score, A's limit, B's limit. Only A matters.
- `WEAK` on A means a removal may not cross A's shape limit, so A then relies on colour.
- Calibration compares each capture with the one before it. So a lifted tool that lands
  differently raises that slot's limit (Box cutter reached 0.53 on Nov 19).
- If one capture scores like a removal, redo it before freezing.

If the camera, light or board moves, redo steps 4–6 before more trials.

### Before trial 1: check the setup is live
1. `refs/<stamp>/thresholds.json` has a `"lighting"` entry with `dark` and `shift`.
2. The session's `session.json` shows `colour_floor` 0.1, a `retake` block and `camera_locked`.
3. After the first trial, `trials.csv` has the five retake columns.
4. At the first checkout, the hybrid prints the empty-board reference check. If it warns that
   the board no longer matches, stop and redo step 4.

---

## 4. Training the model (only when the data or `ai.json` changes)

```
python ai_main.py import-trials          # every data/<session> -> dataset/tc-<session>
                                         # open dataset/tc-*/preview.png and suspects.png
python ai_main.py train --type cnn --test                  # train on ALL sessions, hold none out
python ai_main.py train --type cnn --test tc-<session>     # or hold one session out
python ai_main.py evaluate               # model alone on the held-out session (if any)
python ai_main.py replay data/<session>  # A vs hybrid on a stored session's photos, no camera
```

- **Labels** come from the trial plan (`truth` and `block` in `trials.csv`), never from a
  person.
- **What import skips:** voided trials, the trials in `config/ai_exclude.json` (session 3,
  trials 14 and 16) and sessions with a different kit (such as the 16-tool dry run).
- **What the model sees:** only the pixels that differ from that session's empty-board photo,
  after the exposure is evened out on the paper, placed in a same-size grey window. No pink,
  no shadow outline, no slot shape and no floor. A changed patch whose centre is nearer
  another slot is dropped. An empty slot (changed area under 1 % of its box) is "empty" by
  rule. Import needs each session's `refs/<dir>/empty.png`, and warns and falls back to plain
  boxes without it.
- **Cleaning:**
  - Swap crops are left out of training (the pink shows), but stay in the held-out
    evaluation.
  - Crops whose label disagrees with the photo are listed in `suspects.png` and skipped.
  - Put any other bad crop in `config/ai_bad_crops.txt`, one per line.
- **Augmentation:** each training crop gets 4 extra copies with random brightness, gamma,
  colour temperature, a soft shadow, glare (a blob of the tool greyed out), a small rotation
  and a shift. Evaluation crops are never augmented.
- **`--test` with nothing after it** holds out nothing. Session 3 is already used up as a
  test. The next hybrid session is then that model's held-out test.
- **Check before training:** `meta.json` must list the sessions you meant to train on
  (`train_sessions`).
- **Validation numbers:** the accuracy `train` prints comes from the training sessions and is
  optimistic. Report `evaluate`, `replay` and the hybrid sessions instead.

---

## 5. How the hybrid decides

**Per slot** (expected content is the slot's own tool, or empty for an absent tool):

| The model says | Method A | Result |
|---|---|---|
| confidence < 0.60 | either | **FLAG** "unsure" |
| not the expected content | either | **FLAG** "wrong tool: X", "missing", or "should be empty" |
| expected content | not flagged | ok |
| expected content, ≥ the tool's clear bar | flagged | ok, "cleared by AI" (tool reseated) |
| otherwise | flagged | **FLAG** "changed" |

The clear bar (PoC version) is 0.90 for the Wire stripper and Wire cutter, which really look
alike, and 0.60 for every other tool, for shape and colour flags. In v3 it was 0.90 for any shape
flag and for look-alike slots, and 0.60 only for colour-only flags on other slots; that kept the
Nippers, both tapes and the Measuring tape flagged even when the model named them. Confidences
print rounded down (0.596 shows as 0.59) and are logged with 3 decimals.

The model can raise a flag that A missed. It can cancel A's flag only when it is sure. When
either side is unsure, the slot is flagged.

**Per photo:**
- **Retake:** shadow on the board or room light shifted (toolcheck's frozen `lighting`
  limits), or 3+ slots flagged by A. Retake reasons are logged by category (`lighting`,
  `many_slots`).
- **Lighting override (`shape_ok`).** The photo is accepted with no retake when all of these
  hold:
  - the light changed, or 3+ slots are flagged and every flag is colour only;
  - A's **shape** score is under its limit in every slot;
  - the model names every slot's expected content at ≥ 0.60.

  Any removal or swap moves A's shape score; a light change does not.
- **Many slots, light unchanged:** if the model names every flagged slot at ≥ 0.90, the
  problems are reported without a retake.
- **Retakes run out:** the last photo's verdict stands (`retake_exhausted = 1`). A missing
  tool is never turned into a pass, because it flags again on every retake.
- **Checkout check:** the model also reads every checkout photo. If a slot doesn't hold what it
  should, you are told before the trial (`c` = continue anyway, logged). This covers A's blind
  spot: a board already wrong at checkout.

**Known limits (measured on Nov 19):**
- A local hand shadow differs from the empty-board photo, so it shows up as part of a tool, and
  the model may misread that slot. A room-wide light change mostly cancels out.
- Painter's tape sits near 0.60.
- The stripper in the cutter's slot reads as an unsure "screwdriver", because only part of it
  fits the slot. It is still flagged.

---

## 6. Absent tool (`absent_tools`)

The Flathead screwdriver is at school. Its slot:
- must stay **empty**;
- is never used in a removal or swap;
- is expected empty at checkout and sign-off;
- is flagged "should be empty" if anything is put in it.

The model keeps all 12 classes. **Do not edit `slots.json`**, because that would drop a class.
Leave the slot empty for capture-refs and calibrate. Log it as missing equipment, not as a
dropped tool. Set `absent_tools` back to `[]` when the screwdriver is back.

For code that calls toolai (for example the watch link), use:
`check_capture(..., gap, expected)`, `checkout_issues(..., expected)` and
`score(content, result, expected)`, with
`expected = common.expected_contents(tools, ai_cfg["absent_tools"])`.

---

## 7. Force-accept (tests only)

`python ai_main.py hybrid ... --allow-force`, or set `"test_mode": {"allow_force_accept": true}`.
Every retake prompt then adds **f = FORCE-ACCEPT this photo**. The trial is marked
`forced = 1`, and the summary lists it separately, outside the miss and false-alarm tables. It
exists only in the bench runner, never in `watch-serve`, `watch_main.py serve` or a real sign-off.

---

## 8. Outputs and what to report

```
data/<session>/              toolcheck session: session.json, trials.csv, slot_scores.csv, images/, summary.md
data/hybrid-<stamp>/         hybrid session: session.json (incl. ai_config, thresholds, camera_locked,
                             absent_tools), trials.csv, slot_results.csv, images/, summary.md
data/<session>/replay-<model>/   trials.csv, summary.md, replay.json
data/poc-<stamp>/            PoC run: poc_session.json, plan.json, progress.json, poc_checks.csv (one row per counted
                             check), part<k>/ (watch_log.csv, images/), summary.md
data/poc_frozen.json         hashes recorded by the freeze step
data/watch-<live|replay>-<stamp>/  watch link: watch_session.json, watch_log.csv (one row per message), images/ (live), summary.md
dataset/tc-<session>/        imported crops: session.json, manifest.csv (with tags), crops/, preview.png, suspects.png
models/<stamp>-<type>/       meta.json, train_report.json, model.tflite, model_int8.tflite, model.keras | model.joblib
    eval-<stamp>/            eval.json, confusion.csv, slot_results.csv
refs/<stamp>/                full.png, empty.png, thresholds.json, calibration_scores.csv
```

**Columns in `trials.csv`:**
- Hybrid: `A_*` is Method A alone, `M_*` is the model alone (no A, no retake check, on the same
  final photo) and `H_*` is the hybrid verdict.
- Retake columns: `retakes`, `retake_reason`, `retake_category`, `retake_exhausted`,
  `first_A_flagged` and `first_H_flagged`.

**What to report:**
- Lighting retakes, retakes on non-lighting trials, misses and false alarms, each separately.
- Forced, voided and excluded trials, each on its own line. Never drop them.

**Rules that keep the result honest:**
- Split by session, never by image.
- Freeze `ai.json` and the thresholds before a session.
- A change made after seeing results needs a logbook entry and a new session, and the old
  results are kept.

---

## 9. Watch link (CL8)

The watch starts the two steps of a job; the processor answers with the board status.

1. **Checkout** (press once). The processor takes the checkout photo and the model checks that
   every slot holds its own tool. Watch: **CHECKOUT OK** (job number, 11/11 tools in place) or
   **NOT CHECKED OUT** and the slots that are wrong. A bad checkout is never kept as the baseline.
2. **Sign-off** (press again). The hybrid check against that checkout photo, with the same frozen
   settings as `ai_main.py hybrid`. Watch: **KIT CONFIRMED**, **MISSING** and the tools
   (`Pliers: missing`, `Wire cutter: has Nippers`, `Knife: unsure`), **RETAKE** and the reason
   with the tools seen, or **NO CHECKOUT**.
3. **No answer in 10 s, or an answer it can't read: NOT CONFIRMED.** Only a `complete` answer
   for the same check id shows KIT CONFIRMED.

Hold the BOOT button 2 s to force a new checkout (it replaces an open job's baseline). The absent
screwdriver follows section 6: its slot must be empty and isn't counted (11/11).

The processor compares, not the watch, because the hybrid needs the checkout *photo*. The
proposal said the watch compares with the checkout list (report change 12, L17).

**Protocol.** UDP JSON on port 4211. Watch → PC `{"cmd": "checkout" | "signoff" | "status", "id"}`;
PC → watch `{"id", "op", "status", "tools", "present", "total", "job"}`.

**Setup.**
- `config/settings.json` → `"watch_server": {"listen_port": 4211}`.
- Windows firewall: allow Python **inbound UDP 4211**, network set to *Private*.
- PC and watch on the **same 2.4 GHz network**. Find the PC's IP with `ipconfig`.
- Sketch `watch_firmware/toolcheck_watch_v2/toolcheck_watch_v2.ino`: set `WIFI_SSID`,
  `WIFI_PASS`, `PC_IP`. Same Arduino IDE board settings as the Oct 30 sketch (below).
- If the PC gets a new IP after sleeping, the watch can't reach it; a DHCP reservation avoids this.
- Serial commands on the watch: `c` checkout, `s` sign-off, `q` status, `j10` ten jobs,
  `a60` the 60/60 timeout run.

**Running it.**

| Step | Command | What it gives |
|---|---|---|
| 1 | `python watch_main.py serve --replay data/<session>` + `python watch_main.py sim --jobs 10` | PC only, fake watch; checks the logic |
| 2 | `python watch_main.py serve --replay data/<session>`, then `j10` on the watch | Link latency without the camera. Not accuracy evidence (the models were trained on those sessions) |
| 3 | steps 4–6 of section 3, then `python watch_main.py serve` | Live jobs with the camera and the watch button only. Write each job's truth in the logbook |
| 4 | stop the server, then `a60` on the watch | 60/60 link-cut test (REQ-TR-04) |

After each run, save the Serial Monitor text and run
`python watch_main.py report data/watch-<live|replay>-<stamp> --serial <file>`
(for step 4: `python watch_main.py report --serial watch_timeout_<date>.txt --timeout-test`).

**Pass criteria (frozen before the runs).** 60/60 NOT CONFIRMED with the link cut, each ≥ 10 s;
0 × KIT CONFIRMED without a `complete` answer; the watch screen matches the server answer; with
2 tools removed the watch names both; sign-off ≤ 3 s press to screen.

**Measured.**
- Live runs, night of Nov 19–20 (L17): checkout, sign-off and status worked; server answer
  1.0–2.0 s. The only link failure was the PC going to sleep. Auto white balance drift caused
  many-slot retakes (a camera issue, fixed with `wb_temperature`).
- PoC run, Nov 20 (L18): 842 server answers, median 1.10 s, max 2.50 s; the watch displayed every
  result.
- Link cut, Nov 20: 60/60 NOT CONFIRMED after 10.000–10.001 s, **pass**.
- All times are on the PC stand-in and may not hold on the custom processor (CL7, Gate 2).

**Logs.** The watch prints `LOG,<id>,<op>,<start_ms>,<outcome>,<latency_ms>,<detail>`. The server
writes `watch_log.csv` (one row per message: capture, A, model and total time, the answer, A's and
the hybrid's flags, A's per-slot scores, the model's reading of each slot, retake reason) and
`watch_session.json`.

---

## 10. PoC run (`poc_main.py`)

One menu from references to results. Only the hybrid is used, and the watch starts every check.
Steps 1–2 run `main.py`, steps 3–6 run `ai_main.py`, step 8 runs the watch server (section 9).

```
 1. capture-refs   toolcheck: empty board + full board photos (after ANY camera change)
 2. calibrate      toolcheck: freeze Method A thresholds
 3. check          one live check, every slot (camera + refs + model look right?)
 4. collect        get dataset: labelled captures at today's camera settings
 5. import-trials  optional: add old toolcheck sessions to the dataset
 6. train          CNN on ALL dataset sessions (the PoC run is the held-out test)
 7. freeze         record model + refs + settings; the run refuses silent changes
 8. run            PoC run with the watch (new, or resume)
 9. timeout        60/60 timeout test with the link cut (server OFF)
10. results        90 / 300 / 60 table -> summary.md
```

Before step 1: camera `"exposure_mode": "fixed"`, `"exposure": -4` (the Nov 20 run used −5,
logged as L18), `"wb_temperature"` set to a number (section 2, `FROZEN_SETTINGS_Nov20.md`). Any camera change after step 1 means doing steps 1 and 2 again.
Step 7 refuses to freeze with force-accept on, and step 8 stops if ai.json, settings.json,
slots.json, the refs, the thresholds or the model changed since the freeze (unless you type
CONTINUE, which is recorded in `poc_session.json`).

### Step 8: what you do

The terminal shows each trial and an arrow on the step you're on, for example:

```
=== Trial 12/300  REMOVAL   counted so far: removals 11/90, complete-kit 11/300 ===
     1. Full kit on the board. CHECKOUT on the watch  done
  -> 2. REMOVE Pliers, then SIGN OFF on the watch
     3. PUT Pliers BACK in its slot, then SIGN OFF on the watch
```

- **Removal trial (90):** checkout → remove the named tool → sign off (that is the removal check)
  → put it back → sign off again (that is a complete-kit check).
- **Complete trial (210):** checkout → the perturbation shown (lift, nudge, rotate, hand shadow,
  second lamp, no change; each one used before any repeats) → sign off.
- **RETAKE:** change nothing and sign off again. Only the final answer of a step counts.
- **Between trials** the terminal tells you to press or **hold 2 s** for the next checkout (the PC
  closes the job at the end of each trial).
- The tool to remove comes from a shuffled round of the 11 tools on the board, so each tool is
  removed 8 or 9 times. The Flathead screwdriver (absent) is never used.
- Keep the watch's Serial Monitor open the whole time and save its text at the end of each part as
  `watch_<session>_part<k>.txt`. Menu 10 reads it for the watch side of CL8.
- **Stopping and resuming:** Ctrl+C any time. Menu 8 offers `r = resume` for the newest run. A trial
  that was stopped halfway is listed as *skipped*, not counted, and not hidden. Each resume is a new
  `part<k>/` folder with its own watch log and photos.

Time: about 30–45 s per trial, so 300 trials take roughly 3 to 3.5 hours without breaks (an estimate).

### The PC going to sleep

- While menu 8 runs, the PC is kept awake (Windows `SetThreadExecutionState`, which also keeps the
  screen on). Closing the laptop lid can still sleep it, so leave the lid open or set
  "lid close = do nothing".
- If the PC still sleeps or freezes for more than 20 s, the server notices when it wakes up. It
  re-opens the UDP socket and the camera, re-checks the board against the empty-board photo,
  and prints the PC's IP address. The watch has the PC's IP built into the sketch (`PC_IP`), so
  if Wi-Fi gives the PC a new address after waking, the watch can't reach it until you change it.
  A DHCP reservation on the router avoids that.
- A camera grab that fails is retried once with the camera re-opened. Before, it stopped the server.
- On Windows, a UDP "port unreachable" left over from an earlier send no longer stops the server.
- While the PC is asleep the watch shows NOT CONFIRMED after 10 s. That is the fail-safe working,
  and those presses are counted in the results as lost answers.

### 90 / 300 / 60: what can overlap

- **The 90 removals can't be counted as complete-kit checks.** A removal photo has a tool missing,
  so it tests the opposite of CL3 ("a complete kit is not flagged").
- **But each removal trial can give a complete-kit check:** put the tool back and sign off again. That
  photo really is a complete kit, much like the "lift and put back" perturbation. So **90 removal trials
  + 210 complete trials = 90 removal checks + 300 complete-kit checks**, in 300 trials. That is the default.
  Report it as "90 after returning the removed tool + 210 with perturbations". The results table
  shows the two groups separately, so it's visible if false alarms cluster in one.
- **Timing (CL4, "90 timed checks")** comes free: every counted sign-off is timed, so about 390.
- **The 60 can't be inside the 390.** The 60/60 test needs the link cut (server off), and in the live
  run the link is up. The 390 live sign-offs cover the other half of CL8 (the watch shows the PC's answer,
  and never KIT CONFIRMED without a `complete` answer). The 60/60 is the `a60` serial command,
  automatic, about 12 minutes (menu 9).

Every row of `data/poc-<stamp>/poc_checks.csv` is one counted check. Its `criteria` column says what
it counts toward (`CL1+CL4+CL8`, `CL3+CL4+CL8`, `CL2+CL4+CL8`), and the row has the miss, false alarm,
name check, Method A alone (for comparison), retakes, override and timings.

### Results (step 10)

`summary.md` in the run folder has this table, with a 95 % upper bound for each rate:

| Claim | Checks | Failures | Pass rule |
|---|---|---|---|
| CL1 removal flagged | n/90 | misses | 0 misses in 90 |
| CL3 complete kit not flagged | n/300 | false alarms | 0 in 300 |
| CL4 ≤ 3 s (PC) | n | over 3 s | none over 3 s in 90 |
| CL8 watch | from the watch log | screen ≠ answer, unsafe confirms, lost | 0 unsafe |
| CL8 link cut | 60 | not NOT CONFIRMED | 60/60 |

It says "not enough checks yet" until the counts are reached, so a partial run can go in the report
as it is.

### Results of the Nov 20 run (measured)

Run `poc-20261120-151301`, logbook L18. Model `20261120-150449-cnn`, trained on sessions 1–3; refs
`20261120-145249`; exposure −5. Everything frozen at 15:08 and unchanged during the run (checked by
file hashes). The model never saw these photos. Nothing was skipped or voided.

| Claim | Requirement | Result | 95 % upper bound | Pass rule | |
|---|---|---|---|---|---|
| CL1 removal flagged | REQ-TR-02 | 90/90, all named | 3.3 % | 0 misses in 90 | **pass** |
| CL2 look-alike swap | REQ-TR-02/01 | 29/30 detected, 28/30 named | 14.9 % | reported | – |
| CL3 complete kit not flagged | REQ-TR-06 | 5 false alarms in 300 | 3.5 % | 0 in 300 | **fail** |
| CL4 ≤ 3 s, capture to result (PC) | REQ-TR-07 | 0 of 420 over 3 s; median 1.13 s, max 1.27 s | – | none over 3 s | **pass on PC** |
| CL8 link cut | REQ-TR-04 | 60/60 NOT CONFIRMED | – | 60/60 | **pass** |

- **Complete-kit checks:** 90 after returning the removed tool (2 false alarms) and 210 with a disturbance (3 false alarms). Hand shadow and room-light change gave 0 false alarms in 35 each. Shadows were retaken (33/35), and light changes were retaken (15) or passed by the lighting override (20). No check ran out of retakes.
- **False alarms by slot:** Measuring tape 2 (unsure), Wire stripper 2 (changed), Tweezers 1 (missing). All came from tools that had been put back, rotated or lifted, not from shadows or light.
- **Method A alone** on the same photos: 0 removal misses, 112 complete-kit false alarms.
- **Swaps:** trial 97 (stripper ↔ nippers) left the nippers' slot unflagged, but the kit was still reported incomplete. In trial 103 (stripper ↔ cutter), both slots were flagged but neither was named.
- **Timings:** these are on the PC standing in for the custom processor. They may not hold on the real hardware (CL7, Gate 2).
- **Operator notes:** they are in the logbook (L18) and the report. They are observations only, and no check was recounted because of them.

---

## 11. Older sessions and other tools

**Recovering retake columns** for session `20261117-220424`, which ran before the columns
existed:

```
python recover_retakes.py 20261117-220424 --count 26=2 28=2 32=1
```

- `--count` takes counts from your own record (labelled operator-reported).
- The script first checks that it reproduces `trials.csv`, then writes
  `trials_retakes_recovered.csv` and `retakes_recovery.md`.
- **`trials.csv` is never edited.**
- 1 vs 2 retakes on a clean final photo can't be recovered, because only the first photo is
  saved.

**On the Pi (CL7):**

```
pip install opencv-python-headless numpy scipy ai-edge-litert
python ai_main.py replay data/<session> --model <model folder>   # no camera needed
python ai_main.py profile --n 30                                   # with the camera
```

Use `ai-edge-litert`, not TensorFlow: TensorFlow alone needs more memory than the Pi Zero
2 W's 512 MB.

**Watch test sketch:** `wifi_scan_select.ino` (Oct 30) is a standalone test of the Waveshare
ESP32-S3 Touch AMOLED 2.06. It shows a Wi-Fi scan and a touch-select screen, drawn with
"GFX Library for Arduino". Arduino IDE settings: board ESP32S3 Dev Module, USB CDC On Boot
enabled, Flash 32MB, PSRAM OPI PSRAM. It does not talk to `toolcheck`. The watch link to the
PC is section 9, and the PoC runner section 10.

---

## Version history (toolai)

- **Nov 13:** first `toolai_setup`.
- **Nov 18 (v2):**
  - `import-trials`, checkout check and retakes by category.
  - Force-accept for tests and the lighting override (`all_own`, 0.90).
  - Tool-only crops, exposure evening and swap crops left out of training.
  - No-pink version: pink measurement failed and the outlines were too faint.
- **Nov 19 (v3, after `hybrid-20261118-232906`):**
  - `shape_ok` override at 0.60, because three tools never reached 0.90 in normal light.
  - Colour-only clear at 0.60, and glare augmentation.
- **Nov 20 (v3 update):**
  - `absent_tools`, model-only columns and perturbation cycling.
  - Camera settle/fixed lock with gain, exposure −4, and a planner fix.
- **Nov 20 (watch link):** `watch_main.py`, `toolai/watch_link.py` and the
  `toolcheck_watch_v2` sketch. Checkout and sign-off from the watch, 10 s NOT CONFIRMED.
- **Nov 20 (PoC runner):** `poc_main.py` and `toolai/poc.py` (plan, freeze, resume, keep-awake,
  results); clear bar set per tool. Used for run `poc-20261120-151301` (L18).

---

## AI disclosure

An AI assistant (Claude) was used to (1) write and revise the code in this repository to my
specifications: the Method A check and retake check (`toolcheck`), the slot classifier's import,
training, evaluation and hybrid decision (`toolai`), the watch link and watch sketch, and the PoC
runner; and (2) write the synthetic tests. The test design, the trials on the bench,
the frozen settings, the logbook and every decision are mine. I ran every session myself, and
every number reported here comes from my own session files, checked against my logbook. The
synthetic tests show only that the code runs; they are not evidence.
