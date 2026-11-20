"""Dataset collection on the shadow board.

Each capture starts from the COMPLETE kit. The script tells you what to change
(remove tools, swap a look-alike pair), you do it, it photographs the board and
saves one crop per slot labelled with what is actually in that slot: a tool
name or "empty". The label comes from the script's own instruction, never from
the operator. Then you restore the kit for the next capture.

Output: dataset/<session>/
    session.json   seed, conditions, resolution, slots, plan
    manifest.csv   one row per crop: capture, slot, label, file, block
    frames/        the full photo of every capture (kept as raw data)
    crops/         the slot crops the model trains on
"""
import csv
import random
from datetime import datetime

import cv2

from .board import current_refs_empty, make_board
from .common import DATASET, EMPTY, lookalike_pairs, save_json
from .pink import load_pink


def build_plan(tools, pairs, n, mix, rng):
    """Returns a list of captures; each is {"block", "remove": [...], "swap": [a, b] or None}.

    Blocks are balanced so every tool is removed about equally often and every
    look-alike pair is swapped about equally often.
    """
    n_complete = round(n * mix["complete"])
    n_swap = round(n * mix["swap"]) if pairs else 0
    n_remove = n - n_complete - n_swap

    plan = [{"block": "complete", "remove": [], "swap": None} for _ in range(n_complete)]

    # Removals: 1-3 tools, drawn from a shuffled deck so coverage stays even.
    deck = []
    for _ in range(n_remove):
        k = rng.choice([1, 1, 2, 3])
        picked = []
        while len(picked) < k:
            if not deck:
                deck = rng.sample(tools, len(tools))
            t = deck.pop()
            if t not in picked:
                picked.append(t)
        plan.append({"block": "removal", "remove": picked, "swap": None})

    # Swaps: cycle through the pairs; sometimes also remove one unrelated tool.
    for i in range(n_swap):
        a, b = pairs[i % len(pairs)]
        extra = []
        if rng.random() < 0.3:
            others = [t for t in tools if t not in (a, b)]
            extra = [rng.choice(others)]
        plan.append({"block": "swap", "remove": extra, "swap": [a, b]})

    rng.shuffle(plan)
    return plan


def truth_for(step, tools):
    """What is physically in each slot after the step: {slot_tool: content}."""
    content = {t: t for t in tools}
    for t in step["remove"]:
        content[t] = EMPTY
    if step["swap"]:
        a, b = step["swap"]
        content[a], content[b] = b, a      # slot a now holds b, and slot b holds a
    return content


def describe(step):
    parts = []
    if step["swap"]:
        a, b = step["swap"]
        parts.append(f"SWAP {a} <-> {b} (each into the other's slot)")
    if step["remove"]:
        parts.append("REMOVE " + ", ".join(step["remove"]))
    return "; ".join(parts) if parts else "leave the kit COMPLETE"


def ask_conditions(ask):
    q = [("lighting", "Lighting (e.g. ceiling LED, desk lamp on)"),
         ("placement", "Placement style (e.g. neat, deliberately crooked)"),
         ("operator", "Who is placing the tools"),
         ("logbook_entry", "Logbook entry number")]
    return {k: ask(f"  {label}: ").strip() for k, label in q}


def collect_session(cam, slots, ai_cfg, n, ask=input, seed=None, conditions=None, name=None):
    tools = [s["tool"] for s in slots]
    pairs = lookalike_pairs(ai_cfg, tools)
    margin = ai_cfg["crop_margin"]
    seed = seed if seed is not None else int(datetime.now().timestamp())
    rng = random.Random(seed)
    board = make_board(slots, margin, ai_cfg["board"], load_pink(ai_cfg), current_refs_empty(), "current refs")
    plan = build_plan(tools, pairs, n, ai_cfg["collect_mix"], rng)

    session = name or datetime.now().strftime("%Y%m%d-%H%M%S")
    sdir = DATASET / session
    (sdir / "frames").mkdir(parents=True, exist_ok=True)
    (sdir / "crops").mkdir(parents=True, exist_ok=True)
    if conditions is None:
        print("\nConditions for this collection session (they are saved with the data):")
        conditions = ask_conditions(ask)
    save_json(sdir / "session.json", {
        "session": session, "started": datetime.now().isoformat(timespec="seconds"),
        "seed": seed, "captures": n, "resolution": list(cam.resolution()),
        "crop_margin": margin, "conditions": conditions, "slots": slots, "plan": plan})

    mf = open(sdir / "manifest.csv", "w", newline="", encoding="utf-8")
    w = csv.writer(mf)
    w.writerow(["capture", "slot", "label", "file", "block"])
    print(f"\nCollection {session}: {n} captures, {len(tools)} crops each -> {sdir}")
    print("Type q then Enter at any prompt to stop early (everything so far is kept).\n")
    done = 0
    try:
        for i, step in enumerate(plan, start=1):
            if ask(f"[{i}/{n}] Kit COMPLETE? Then: {describe(step)}. Enter to capture...").strip() == "q":
                break
            img, _ = cam.grab()
            cv2.imwrite(str(sdir / "frames" / f"{i:04d}.png"), img)
            content = truth_for(step, tools)
            crops = board.crops(img)
            for j, slot in enumerate(slots):
                fn = f"crops/{i:04d}_{j:02d}.png"
                cv2.imwrite(str(sdir / fn), crops[j])
                w.writerow([i, slot["tool"], content[slot["tool"]], fn, step["block"]])
            mf.flush()
            done += 1
            if step["remove"] or step["swap"]:
                ask("  Captured. Restore the kit (everything back in its own slot), Enter...")
    finally:
        mf.close()
    print(f"\nSaved {done} captures ({done * len(tools)} crops) in {sdir}")
    return session
