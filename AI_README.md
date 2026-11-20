# V3 update (Nov 20): absent tool, model-only column, every perturbation

- **`absent_tools`** in `ai.json` (now `["Flathead screwdriver"]`, left at school). That slot
  must stay EMPTY: never in a removal or swap, expected empty at checkout and sign-off,
  flagged "should be empty" if anything is put in it. The model keeps all 12 classes.
  toolcheck needs NO change: do NOT edit slots.json; just leave the slot empty for
  capture-refs and calibrate. Log it as missing equipment, not a dropped tool. Set back to
  `[]` when the screwdriver is back. With 11 tools, use 11 or 22 removals.
- **Model only** columns (`M_flagged`, `M_miss`, `M_false_alarm`, `M_named_correctly`) in
  trials.csv and the summary: the AI alone (no Method A, no retake check) on the SAME final
  photo the hybrid judged. Summary table = A vs Model vs Hybrid.
- **`plan.cycle_perturbations`** (true): the complete-kit trials use every perturbation in
  the kit file (incl. "Cast a hand shadow") before repeating one.
- For other code calling toolai: `check_capture(..., gap, expected)`,
  `checkout_issues(..., expected)`, `score(content, result, expected)`; build `expected` with
  `common.expected_contents(tools, ai_cfg["absent_tools"])`.

# V3 (Nov 19, no-pink): what changed and how to run it

Changed after seeing hybrid session `hybrid-20261118-232906`, so: **logbook entry first,
then a NEW hybrid session. Trials 23 and 29 of that session are exploratory only.**

1. **Lighting override, mode `shape_ok`** (`lighting_override`, confidence 0.60). Accept the
   photo without a retake when the light changed (retake check) OR 3+ slots are flagged and
   every flag is colour only, AND Method A's SHAPE score is under its frozen limit in every
   slot, AND the AI's top answer is the slot's own tool in every slot at >= 0.60.
   Why: in that session Measuring tape, Wire stripper and Box cutter never reached 0.90 even in
   normal light, so the old rule (every slot >= 0.90) could never fire. Every removal and swap
   in that session pushed A's shape score past its limit; the light changes did not.
2. **Colour-only clear** (`decision.colour_only_clear_confidence` 0.60). A slot A flagged on
   colour only is cleared when the AI names its own tool at >= 0.60. Look-alike slots
   (`lookalike_pairs`) still need 0.90: stripper/cutter swaps were only just past A's shape limit.
3. **Glare augmentation** (`augment.glare_prob` 0.3). Training copies get a soft blob of the
   tool greyed out, the way glare makes part of a tool vanish in the tool-only crop.
   Aimed at the Measuring tape read as "empty" in 40 of 63 session 3 crops.

Rule check on the stored results of `hybrid-20261118-232906` (same model, no retraining;
calculated from slot_results.csv, not a new test): 0 misses; photos with a false alarm 4 -> 3;
trial 29 overridden with nothing hidden; trial 23 still asks for a retake (Pliers 0.50).
The 3 left are all Measuring tape, which needs the retrain.

## Steps
1. Copy `toolai/` and `config/ai.json` over your current no-pink copy (keep your `dataset/`,
   `models/`, `refs/`, `data/`). Do NOT re-run import-trials; the crops are unchanged.
2. Logbook entry (dated): the three changes above and why, frozen before the new session.
3. Train on ALL three sessions, nothing held out (session 3 is already used up):
   `python ai_main.py train --type cnn --test`
   (`--test` with nothing after it = hold out nothing. The warning that `evaluate` has nothing
   to score is expected.)
4. `capture-refs` and `calibrate` in toolcheck, as at the start of every session.
5. New hybrid session with the new model: `python ai_main.py hybrid` -> `hybrid-summary`.
   That session is the held-out evidence for v3.

# NO-PINK VERSION (Nov 18, 22:30)

Same code as `toolai_v2_Nov18`, but `config/ai.json` turns pink off:
`pink.enabled = false` and `board.footprint.enabled = false`.

- **Skip `pink`.** Order: `import-trials` -> check `dataset/tc-*/preview.png` and `suspects.png` -> `train` ...
- **How it finds the tool without pink:** each slot is compared with the same slot on that
  session's empty-board photo (`refs/<dir>/empty.png`), after the exposure is evened out on the
  paper. Pixels that changed are the tool. They go in a same-size grey window, so the shadow,
  floor and slot shape never reach the model.
- **Neighbouring tool reaching into the box:** a changed patch whose centre is nearer another
  slot's centre is dropped.
- **Needs** the three reference folders in `refs/`: `20261112-211325`, `20261112-223950` and
  `20261117-215055`. Without a session's `empty.png`, that session falls back to plain boxes
  and import warns.

On the synthetic board this version caught the planted mislabel, flagged nothing else, and
named 12/12 held-out swaps with no swaps in training. That shows the code works; it is not
evidence.

---

# Tool recognition model (hybrid D): `ai_main.py`, `toolai/`

Version: Nov 18, 2026. Replaces the Nov 13 `toolai_setup.zip`.

The model looks at each shadow-board slot and says **which tool is in it, or "empty"**,
with a confidence from 0 to 1. Method A still says which slots *changed*. The hybrid
combines both, slot by slot. It is trained on the PC and exported as a `.tflite` file
for the processor (REQ-TR-05).

## What is new since the Nov 13 version

| Feature | Why (your words or your data) |
|---|---|
| `import-trials`: turns the photos you already took in toolcheck sessions into labelled training crops | "create the AI model based on the images I took" |
| `pink`: pink pixels are painted grey before the model sees a crop | "try to ignore the color pink" (the model can't learn the shadow outline in place of the tool) |
| Lighting override: if the light changed but the model names every tool, no retake | your trial 27 note |
| `--allow-force`: `f` at a retake prompt accepts the photo, **tests only** | "brute force the photo if I am forever stuck in a retake" |
| Checkout check: the model looks at every checkout photo and stops you if a slot doesn't hold its own tool | session 3 trials 14 and 16 (previous swap not undone) |
| Retakes logged by category (`lighting` / `many_slots`), never lumped together | your trial 14 note: "shouldn't be logged as bad lighting" |
| The five retake columns are always written | missing from session 3's `trials.csv` |
| The model may cancel A's flag only when it's ≥ 0.90 sure the own tool is there ("cleared by AI") | your trial 17 and 29 notes: nuisance false alarms after reseating a tool |
| `unsure` (confidence < 0.60) counts as a flag | fail safe: never confirm the kit on a guess |
| `replay`: runs A and the hybrid side by side on a stored session's photos, no camera | compare on the same captures; also times the model on the Pi |
| Training adds copies of each crop with random brightness, colour temperature, a soft shadow, small rotation and shift | one copy of each tool, mostly one lighting setup |
| SVM baseline now also gives a confidence | so both models use the same rules |
| Look-alike pairs now include Wire stripper / Nippers | matches your 12-tool `kit.json` |

## Nov 18 evening update: cleaner training samples

You found crops showing the wood floor, two tools at once, different exposure between
sessions, crops labelled as a tool where only the shadow shows, and swap crops where the pink
shows. What the model sees now (`board` and `training` in `ai.json`):

1. **Exposure evened out** using the bare paper between slots. Each colour channel is scaled
   so the paper is the same brightness in every session.
2. **Tool-only view.** Each slot is compared with the same slot on that session's empty-board
   photo (`refs/<dir>/empty.png`, found from `refs_dir` in `session.json`). Only the pixels that
   differ, inside the slot's own shadow footprint, are kept. They go in a same-size grey window
   centred on the tool. The floor, the neighbouring tool, the pink, the shadow outline and the
   slot box shape are all gone, so the model can't tell which slot it is looking at, only
   which tool. An empty slot is a plain grey window.
3. **Swap crops left out of training** (`exclude_swap_crops`). They stay in the held-out
   evaluation, which is the honest swap test.
4. **Wrong labels flagged.** Import measures how much of each slot's shadow is covered. A crop
   labelled as a tool that looks empty (or the reverse) is marked `suspect` in `manifest.csv`
   and shown in `dataset/<session>/suspects.png`. Suspects are left out of training and
   evaluation (`skip_suspects`), and the number is reported. Any other bad crop you spot goes
   in `config/ai_bad_crops.txt`, one per line, e.g. `tc-20261112-212119/crops/t005_signoff_03.png`.

**New order:** `pink` FIRST, then `import-trials`. Then open each `dataset/tc-*/preview.png`
(one photo as the model sees it) and `suspects.png` before training. Your `refs/` folder must
still have the three reference folders the sessions used (`20261112-211325`, `20261112-223950`,
`20261117-215055`). If one is missing, import warns and that session falls back to plain boxes.

On the synthetic board, the model trained with no swap crops still named 12/12 held-out swaps
correctly in this view. That shows the code works, not how your tools will do.

## Put it in your project

Copy into the `toolcheck` project root (next to `main.py`):

```
ai_main.py
toolai/                 (whole folder; replaces the Nov 13 one)
config/ai.json          (replaces the Nov 13 one)
config/ai_exclude.json  (new)
tests/test_toolai.py    (replaces the Nov 13 one)
```

Install on the PC:

```
pip install "opencv-python>=4.8,<5" tensorflow scikit-learn joblib scipy
```

Use OpenCV 4.x: OpenCV 5 removed the HOG function the SVM baseline uses.

Then check `config/ai.json`: the look-alike names must match `config/slots.json` exactly.

## Workflow

```
python -m tests.test_toolai                     # 1. synthetic check that everything runs (~10 min)

python ai_main.py pink                          # 2. measure YOUR pink from refs/<current>/empty.png
                                                #    open config/pink_preview.png and check it
python ai_main.py import-trials                 # 3. every data/<session> -> dataset/tc-<session>
                                                #    check dataset/tc-*/preview.png and suspects.png

python ai_main.py train --type cnn --test tc-20261117-220424   # 4. hold out session 3
python ai_main.py train --type svm --test tc-20261117-220424   #    baseline for the comparison
#    5. FREEZE config/ai.json now (confidence limits), using the validation table train prints
python ai_main.py evaluate                      # 6. model alone, on the held-out session
python ai_main.py replay data/20261117-220424   # 7. A vs hybrid on session 3's real photos

python ai_main.py hybrid --removal 12 --swap 10 --complete 10   # 8. new live session
python ai_main.py hybrid-summary hybrid-<stamp>
```

**Step 2.** The import skips voided trials and the trials in `config/ai_exclude.json`.
I pre-filled that file with session 3 trials 14 and 16, from your notes. Confirm both.
It also skips sessions whose tools aren't the current kit, which leaves out the 16-tool
dry run automatically. The labels come from the trial plan (`truth` and `block` in
`trials.csv`), never from a person.

**Step 3.** `pink` lists the share of each tool's pixels that the mask would erase.
Every tool should be near 0%. Anything over 5% is marked CHECK; that means the tool has
pink or magenta on it. If that happens, narrow `hue_tol_deg` in `config/pink.json` and
rerun the preview. Freeze it before training. Each model stores the pink values it was
trained with in its `meta.json`.

**Step 4: which session to hold out.** Session 3 is the best test because it has the real
shadow and lamp photos (`t026/t028/t032_signoff_try1.png`). Those are the test of the
lighting override. Training on sessions 1 and 2 only is thin, though. Each session has
about 10 swaps (about 20 swapped-tool crops) and 12 removals (12 empty crops). If the
held-out results are weak on swaps, run one or two `collect` sessions first. Each
capture removes 1 to 3 tools or swaps a look-alike pair, so 60 captures give about
720 crops, and far more swaps and empties. Then train on everything except session 3.

**Step 5: freeze before you look.** `train` prints a table of validation confidences:
how many wrong answers sit above each limit, and how many right answers fall below it.
Pick `decision.min_confidence`, `decision.clear_confidence` and
`lighting_override.confidence` from that table, save `ai.json`, write it in the logbook,
and only then run `evaluate` and `replay`. These limits work like toolcheck's frozen
thresholds.

## How the hybrid decides (per slot)

| The model says | Method A | Result |
|---|---|---|
| confidence < 0.60 (`min_confidence`) | either | **FLAG** "unsure" |
| another tool, or empty | either | **FLAG** "wrong tool: X" or "missing" |
| own tool | not flagged | ok |
| own tool, confidence ≥ 0.90 (`clear_confidence`) | flagged | ok, "cleared by AI" (reseated) |
| own tool, confidence < 0.90 | flagged | **FLAG** "changed" |

The model can raise a flag that A missed. This covers A's checkout blind spot and
names swaps. It can cancel A's flag only when it is very sure. When either side is
unsure, the slot is flagged.

**Per photo:**

- **Lighting retake:** shadow on the board, or the room light shifted. The limits are
  toolcheck's frozen `lighting` thresholds.
- **Lighting override** (`lighting_override`, mode `all_own`): the light changed, but
  the model names every slot's own tool with confidence ≥ 0.90. The photo is accepted
  with no retake. If even one slot is missing, wrong or unsure, it retakes as before.
  The other mode, `all_confident`, also accepts a photo when the model confidently
  names a missing tool.
- **Many slots** (3+ flagged by A): if the light did *not* change and the model
  confidently names every flagged slot, the problems are reported without a retake.
- **Retakes run out:** the last photo's verdict stands (`retake_exhausted = 1`).

## Force-accept (tests only)

Enable it with `python ai_main.py hybrid ... --allow-force`, or set
`"test_mode": {"allow_force_accept": true}` in `ai.json`. Each retake prompt then adds
**f = FORCE-ACCEPT this photo**. The trial's verdict comes from that photo, and the
trial is marked `forced = 1`. `hybrid-summary` lists forced trials separately and leaves
them out of the miss and false-alarm tables. It exists only in the bench runner, never
in `watch-serve`. Set `retake.max_retakes` in `ai.json` to change how many retakes you
get before the last photo stands. `null` uses toolcheck's `settings.json` value.

## Rules that keep the result honest

1. **Split by session, never by image.** The held-out session is written into
   `meta.json`. `train` refuses to use a session for both training and testing.
   `replay` marks a session the model trained on as "NOT evidence".
2. **Freeze `ai.json` and `pink.json` before evaluating.** Changed something after
   seeing the results? Write a dated logbook entry, retrain, test on a **new** session,
   and keep and report the old evaluation.
3. **Report forced, voided and excluded trials** as their own lines. Never drop them.
4. The synthetic test proves only that the code runs. Its accuracy is not evidence.
5. The validation accuracy that `train` prints comes from the training sessions and is
   optimistic. Report the `evaluate` and `replay` numbers instead.

## Where this may differ from your PyCharm copy

This was built against the Nov 12 zip plus the Nov 17 retake patch, because your newer
local files aren't in the project folder. Check these points:

- **Photo names:** `import-trials` and `replay` expect
  `data/<session>/images/t001_checkout.png`, `t001_signoff.png` and
  `t001_signoff_try1.png`, plus `trials.csv` with the columns `trial`, `block`, `truth`,
  `instruction`, `void` and `void_reason`. If your local `trials.py` names them
  differently, tell me.
- **Method A and the lighting check** are imported from *your* `toolcheck`
  (`toolcheck.methods.method_a` and `toolcheck.features.gap_mask` / `lighting_change`),
  so the numbers match your trials. If an import fails, a copy of the Nov 17 code is used
  instead (`toolai/_toolcheck_copy.py`). `session.json` records which one ran
  (`toolcheck_code_used`).
- **`hybrid`** reads `config/settings.json` (it needs the `retake` block), `config/kit.json`
  and `refs/<current>/thresholds.json` (it needs the `lighting` entry). It prints a
  warning when either is missing.
- **Camera:** it reuses `toolcheck.camera.Camera`, so you get the same exposure lock and
  frame averaging. White balance still isn't locked; that's the open housekeeping item.

## On the Pi (CL7)

```
pip install opencv-python-headless numpy scipy ai-edge-litert
python ai_main.py replay data/20261117-220424 --model <model folder>   # no camera needed
python ai_main.py profile --n 30                                         # with the camera
```

Use `ai-edge-litert`, not TensorFlow: TensorFlow alone needs more memory than the Pi Zero
2 W's 512 MB. On the synthetic board, the 12-slot CNN is about 90 KB and takes about 8 ms
per photo on the cloud machine. That number is not a Pi measurement.

## Outputs

```
dataset/tc-<session>/      imported crops: session.json, manifest.csv (with tags), crops/
dataset/<stamp>/           collect sessions: also frames/
config/pink.json, config/pink_preview.png
models/<stamp>-<type>/     meta.json (incl. pink + augment used), train_report.json
                           (incl. val_confidence_table), model.tflite, model_int8.tflite,
                           model.keras | model.joblib
    eval-<stamp>/          eval.json (incl. accuracy_by_tag, lighting_override_on_retake_photos),
                           confusion.csv, slot_results.csv
data/<session>/replay-<model>/   trials.csv, summary.md, replay.json
data/hybrid-<stamp>/       session.json, trials.csv, slot_results.csv, images/, summary.md
```

Tags in `manifest.csv` and `eval.json`: `checkout`, `signoff`, `signoff_try1`/`retake_photo`
(the real bad-light photos), `lighting_trial`, `swap_crop`. A `swap_crop` is a tool sitting in
another tool's slot, where pink shows. You said you don't trust those crops, so their accuracy
is reported on its own line.
