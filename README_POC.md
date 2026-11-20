# PoC run, hybrid only (Nov 20)

Code drafted with Claude. The test plan choices, the frozen values and every number are yours.

This is toolai v3 no-pink (Nov 20) plus the watch link (Nov 20), with one new menu, `poc_main.py`,
that takes you from references to results. Only the hybrid is used. The watch starts every check.

## Put it in your PyCharm project

| File | What changed |
|---|---|
| `poc_main.py` | **new**, project root (next to `main.py` and `ai_main.py`) |
| `toolai/poc.py` | **new**: plan, scoring, keep-awake, server that survives sleep, results |
| `toolai/hybrid.py` | per-tool clear bar; confidences printed rounded down |
| `toolai/watch_link.py` | logs confidences with 3 decimals (otherwise as in watch_link_Nov20) |
| `ai_main.py` | `check` prints 3 decimals (nothing else) |
| `config/ai.json` | **only add** the 3 new keys in `"decision"` (below) to your own file, so your `absent_tools` etc. stay |
| `tests/test_poc.py` | **new** synthetic test: `python -m tests.test_poc` (it is not evidence) |

`watch_main.py`, the watch sketch and everything else are unchanged from watch_link_Nov20.

The 3 keys for `config/ai.json` → `"decision"`:

```json
"clear_confidence_by_tool": {"Wire stripper": 0.90, "Wire cutter": 0.90},
"clear_confidence_default": 0.60,
"_note_poc": "..."
```

## The menu (press Run on `poc_main.py`)

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

Before 1: camera `"exposure_mode": "fixed"`, `"exposure": -4`, `"wb_temperature"` set to a number
(FROZEN_SETTINGS_Nov20.md). Any camera change after step 1 means doing steps 1 and 2 again.
Step 7 refuses to freeze with force-accept on, and step 8 stops if ai.json, settings.json,
slots.json, the refs, the thresholds or the model changed since the freeze (unless you type
CONTINUE, which is recorded in `poc_session.json`).

## Step 8: what you do

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

## The PC going to sleep

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

## Why the Measuring tape at 0.60 was never cleared

It wasn't the look-alike rule: the Measuring tape isn't in any look-alike pair. In v3 the AI can
clear a slot that Method A flagged in two ways:

1. A flagged on **colour only** (shape score under its limit): cleared at 0.60, except look-alike slots (0.90).
2. A flagged on **shape**: cleared only at **0.90, for every tool**.

The tape is big and shiny, so a reseat or a light change moves its shape score: its calibrated shape
limit in the Nov 20 00:18 watch run was already 0.30, against 0.05 for most slots, and it went to 0.36
once. On a shape flag the tape needed 0.90, which it never reaches. A second, smaller cause: anything
under the 0.60 minimum is flagged "unsure", and a 0.596 was printed as "0.60". Your label tells you
which one it was. `changed (AI 0.6x ...)` is the 0.90 bar, and `unsure (AI 0.60 ...)` is the rounding.

What this version does:
- **Per-tool clear bar**, for shape and colour flags: Wire stripper and Wire cutter stay at 0.90. Every
  other tool (Nippers, both tapes, Measuring tape and the rest) uses 0.60, the same as `min_confidence`.
  Add a tool to `clear_confidence_by_tool` to make it strict too (Box cutter vs Knife, for example).
- Confidences print **rounded down** (0.596 shows as 0.59) and are logged with 3 decimals.
- Unchanged: under 0.60 is still "unsure" and flagged, and a wrong tool or empty is always flagged.
  So the AI can still only cancel A's flag when it names the slot's own tool.

This is a rule change after seeing results. It needs a logbook entry dated before the run, then
capture-refs, calibrate and freeze. The PoC run is a new session, so it counts; the earlier sessions
keep their old rule and are reported as they were.

## 90 / 300 / 60: what can overlap

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

## Results (menu 10)

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

## Tested

`python -m tests.test_poc` (synthetic board, SVM, real UDP on one PC). It covers the plan, the
per-tool bar, a retake, a skipped step, resuming mid-trial, the UDP loop and the results table.
`tests.test_watch_link` still passes. In the cloud, `tests.test_toolai` passes up to CNN training,
which needs TensorFlow (not installed there). Not tested: the real webcam, the real watch, and
Windows keep-awake (Windows-only).
