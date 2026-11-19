# toolcheck: bench PoC for tool return confirmation

Camera check that every tool in a 12-tool kit is back on its pink shadow board at sign-off.
Method A compares the sign-off photo with the checkout photo; Method B compares it with stored
full/empty references (kept for comparison only, rejected as a design choice because it drifts
after a tool is reseated). Thresholds are frozen by calibration before any trial.

## Setup (PyCharm, Windows)
1. Open this folder as a project in PyCharm.
2. Settings > Project > Python Interpreter > add a new virtual environment.
3. In the PyCharm terminal: `pip install -r requirements.txt`
4. Check everything works without the webcam: `python -m tests.test_pipeline`
   (should end with `session: run + analyze OK`).

Running `python main.py` with no command (or PyCharm's Run button) shows a numbered menu of the steps below.

## Settings to check in `config/settings.json`
- `calibration.colour_floor`: `0.1`
- a top-level `"retake"` block:

```json
"retake": {
  "enabled": true,
  "max_flagged": 3,
  "max_retakes": 2,
  "dark_tol": 0.15,
  "blur_sigma": 8.0,
  "dark_floor": 0.02,
  "shift_floor": 0.05
}
```

The retake check only runs when this block has `"enabled": true` **and** the current
`thresholds.json` has a `"lighting"` entry (written by `calibrate`). Without either, trials run
with no retake check and no warning.

## Bench night, in order
| Step | Command | What you do |
|---|---|---|
| 1 | `python main.py preview` | Fix camera and light. Whole board in frame. `q` to quit. If it opens the laptop camera, set `camera.index` to 1. |
| 2 | `python main.py exposure` | Bare board. `[` / `]` change exposure until it says GOOD (median 200–230, < 0.5 % clipped white). Put the value in `settings.json`. |
| 3 | `python main.py define-slots` | Drag a box around each slot, in the order of `config/kit.json`. Leave a margin around each tool. |
| 4 | `python main.py capture-refs` | Full board, then empty board. Don't touch the camera between. |
| 5 | `python main.py calibrate` | 12 complete-kit captures, lifting and replacing 3 tools each time. Read the separation check and `calibration_scores.csv`: redo a capture if a `WEAK` slot or a lifted tool scores like a removal, *before* trials. |
| 6 | `python main.py run --removal 90 --swap 30 --complete 60` | Follow the prompts. Enter = accept, `v` = void (procedural mistake only), `n` = note, `q` = stop. |
| 7 | `python main.py analyze <session>` | Prints and saves `data/<session>/summary.md`. |

Short on time? `run --removal 12 --swap 10 --complete 10` still gives honest numbers
with wider bounds. Use removal counts in multiples of 12 so every tool is removed. Report n and the bound.

If the camera, light or board moves, redo steps 4–5 before more trials.

## Before trial 1: check the retake check is live
1. The new `refs/<stamp>/thresholds.json` has a `"lighting"` entry with `dark` and `shift`.
2. `data/<session>/session.json` shows `colour_floor` 0.1 and a `retake` block.
3. After the first trial, `trials.csv` has the five retake columns.

If any of these fails, stop and fix it before running more trials.

## Retakes during a session
If the sign-off photo has a shadow, a brightness shift, or 3+ slots flagged, the script asks for a
retake (up to 2). Move the hand or shadow away and press Enter, as a technician would. The first
photo is kept as `images/t<NN>_signoff_try1.png`. The final verdict is always the last photo;
if retakes run out it still stands and `retake_exhausted = 1`. A missing tool is never turned
into a pass: it flags again on every retake.

`trials.csv` records `retakes, retake_reason, retake_exhausted, first_A_flagged, first_B_flagged`.
Report lighting retakes, retakes on non-lighting trials, and misses / false alarms separately.

## Recovering retake columns for an older session
Session `20261117-220424` ran before the five columns were written. To rebuild them from the saved photos:

```
python recover_retakes.py 20261117-220424 --count 26=2 28=2 32=1
```

`--count` gives retake counts from your own record (labelled operator-reported); leave out any
you don't know. The script first checks it reproduces `trials.csv` on every final photo, then
writes `trials_retakes_recovered.csv` and `retakes_recovery.md` into the session folder.
**`trials.csv` is never edited.** A retake count of 1 vs 2 on a clean final photo can't be
recovered, because only the first photo is saved.

## Watch test sketch
`wifi_scan_select.ino` (Oct 30) is a standalone test of the Waveshare ESP32-S3 Touch AMOLED 2.06:
Wi-Fi scan and touch-select screen, drawn with "GFX Library for Arduino". Open it in Arduino IDE
(board ESP32S3 Dev Module, USB CDC On Boot enabled, Flash 32MB, PSRAM OPI PSRAM) and upload.
It does not talk to `toolcheck`.

The watch link to the PC (`watch-serve`) is on a separate branch.
