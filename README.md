# toolcheck: bench PoC for tool return confirmation

## Setup (PyCharm, Windows)
1. Open this folder as a project in PyCharm.
2. Settings > Project > Python Interpreter > add a new virtual environment.
3. In the PyCharm terminal: `pip install -r requirements.txt`
4. Check everything works without the webcam: `python -m tests.test_pipeline`
   (should end with `session: run + analyze OK`).

## Bench night, in order
| Step | Command | What you do |
|---|---|---|
| 1 | `python main.py preview` | Fix camera and light. Whole board in frame. `q` to quit. If it opens the laptop camera, set `camera.index` to 1. |
| 2 | `python main.py define-slots` | Drag a box around each slot, in the order of `config/kit.json`. Leave a margin around each tool. |
| 3 | `python main.py capture-refs` | Full board, then empty board. Don't touch the camera between. |
| 4 | `python main.py calibrate` | 12 complete-kit captures, lifting and replacing 3 tools each time. Read the separation check: `WEAK` slots are worth a logbook note *before* trials. |
| 5 | `python main.py run --removal 90 --swap 30 --complete 60` | Follow the prompts. Enter = accept, `v` = void (procedural mistake only), `n` = note, `q` = stop. |
| 6 | `python main.py analyze <session>` | Prints and saves `data/<session>/summary.md`. |

Short on time? `run --removal 30 --swap 10 --complete 20` still gives honest numbers
with wider bounds. Report n and the bound.

## What to copy into the logbook
`session.json` (conditions, seed, thresholds, resolution), `summary.md`, and anything odd
you noticed. The CSVs and images are the raw evidence: keep them.

If the camera, light or board moves, redo steps 3–4 before more trials.
