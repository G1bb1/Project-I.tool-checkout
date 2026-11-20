"""Turn the photos you already took in toolcheck sessions into labelled training data.

Every toolcheck trial saved (in data/<session>/images/):
    t<NN>_checkout.png        the kit before the job        -> every slot holds its own tool
    t<NN>_signoff.png         the final sign-off photo      -> labelled from the trial plan
    t<NN>_signoff_try1.png    the first photo when a retake  -> same labels as the sign-off
                              fired (bad light / shadow)        (tagged "retake_photo")
The label comes from the script's plan (trials.csv "truth" + block), never from
a person, exactly like the trial results themselves. Each session's OWN slot boxes
(from its session.json) are used, so moving the camera between sessions is fine.

Skipped automatically: voided trials, trials listed in config/ai_exclude.json
(operator errors where the photo doesn't match the plan), and sessions whose
tools aren't the current kit (the 16-tool dry run on blank paper).

Output: dataset/tc-<session>/ with the same layout as `collect`
(session.json, manifest.csv, crops/), plus columns source, trial, image, tags.
Frames are not copied (they stay in toolcheck's data/ folder).
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .board import make_board, refs_empty_for
from .common import CONFIG, DATASET, EMPTY, crop_slot, load_json, save_json, slot_truth
from .pink import load_pink

MANIFEST_FIELDS = ["capture", "slot", "label", "file", "block", "source", "trial", "image", "tags",
                   "tool_share", "suspect", "suspect_why"]
LIGHT_WORDS = ("shadow", "lamp", "curtain", "light")


def load_exclusions():
    p = CONFIG / "ai_exclude.json"
    if not p.exists():
        return {}
    out = {}
    for e in load_json(p).get("exclude", []):
        out[(str(e["session"]), int(e["trial"]))] = e.get("reason", "")
    return out


def import_session(tc_dir, kit_tools, ai_cfg, include_void=False):
    tc_dir = Path(tc_dir)
    sj = load_json(tc_dir / "session.json")
    sid = sj["session"]
    slots = sj["slots"]
    tools = [s["tool"] for s in slots]
    if sorted(tools) != sorted(kit_tools):
        print(f"  {sid}: SKIPPED, its tools are not the current kit "
              f"(extra {sorted(set(tools) - set(kit_tools))}, missing {sorted(set(kit_tools) - set(tools))})")
        return None
    images = tc_dir / "images"
    if not images.exists():
        print(f"  {sid}: SKIPPED, no images/ folder")
        return None
    excl = load_exclusions()
    margin = ai_cfg["crop_margin"]
    name = f"tc-{sid}"
    out = DATASET / name
    (out / "crops").mkdir(parents=True, exist_ok=True)

    with open(tc_dir / "trials.csv", newline="", encoding="utf-8") as f:
        trials = list(csv.DictReader(f))
    empty_path = refs_empty_for(sj.get("refs_dir"))
    board = make_board(slots, margin, ai_cfg["board"], load_pink(ai_cfg), empty_path,
                       f"{sid}: refs {sj.get('refs_dir')}")
    preview_done = False

    rows, skipped, missing_files, cap = [], [], [], 0
    crop_cache = {}
    for tr in trials:
        n = int(tr["trial"])
        if tr.get("void") == "1" and not include_void:
            skipped.append((n, "voided: " + tr.get("void_reason", "")))
            continue
        if (sid, n) in excl:
            skipped.append((n, "ai_exclude.json: " + excl[(sid, n)]))
            continue
        block = tr["block"]
        truth = json.loads(tr["truth"])
        light = block == "complete" and any(w in tr["instruction"].lower() for w in LIGHT_WORDS)
        signoff_truth = slot_truth(tools, truth, block)
        complete = {t: t for t in tools}
        for kind, fname, content, block_label in (
                ("checkout", f"t{n:03d}_checkout.png", complete, "complete"),
                ("signoff", f"t{n:03d}_signoff.png", signoff_truth, block),
                ("signoff_try1", f"t{n:03d}_signoff_try1.png", signoff_truth, block)):
            p = images / fname
            if not p.exists():
                if kind != "signoff_try1":
                    missing_files.append(fname)
                continue
            img = cv2.imread(str(p))
            crops = board.crops(img)
            if not preview_done:
                board.preview(img, out / "preview.png")
                preview_done = True
            cap += 1
            for j, s in enumerate(slots):
                label = content[s["tool"]]
                tags = [kind]
                if light:
                    tags.append("lighting_trial")
                if kind == "signoff_try1":
                    tags.append("retake_photo")
                if label not in (s["tool"], EMPTY):
                    tags.append("swap_crop")
                crop_name = f"crops/t{n:03d}_{kind}_{j:02d}.png"
                cv2.imwrite(str(out / crop_name), crops[j])
                d = board.last_content[j] if board.tool_only else None
                rows.append({"capture": cap, "slot": s["tool"], "label": label, "file": crop_name,
                             "block": block_label, "source": sid, "trial": n, "image": fname,
                             "tags": ";".join(tags), "tool_share": "" if d is None else f"{d:.4f}",
                             "suspect": 0, "suspect_why": ""})
                crop_cache[crop_name] = crop_slot(img, s, margin)   # the real slot, for review

    suspects = flag_suspects(rows)
    if suspects:
        write_sheet(out / "suspects.png", [(crop_cache[r["file"]], f"t{r['trial']} {r['image'][5:-4]} {r['slot'][:12]}",
                                            r["suspect_why"]) for r in suspects])

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    save_json(out / "session.json", {
        "session": name, "started": sj.get("started", ""), "imported_from": str(tc_dir),
        "toolcheck_session": sid, "conditions": sj.get("conditions", {}),
        "resolution": sj.get("resolution"), "crop_margin": margin, "slots": slots,
        "photos": cap, "crops": len(rows),
        "board": {"empty_ref": str(empty_path) if empty_path else None,
                  "footprints": board.has_footprints(), "config": ai_cfg["board"]},
        "skipped_trials": [{"trial": t, "why": why} for t, why in skipped],
        "missing_files": missing_files})
    swaps = sum("swap_crop" in r["tags"] for r in rows)
    retake = sum("retake_photo" in r["tags"] for r in rows) // max(1, len(slots))
    print(f"  {sid} -> dataset/{name}: {cap} photos, {len(rows)} crops "
          f"({swaps} swap crops, {retake} retake photos, {len(suspects)} SUSPECT labels -> suspects.png); "
          f"skipped trials {[t for t, _ in skipped]}"
          + (f"; MISSING {len(missing_files)} files, e.g. {missing_files[:3]}" if missing_files else ""))
    return name


def flag_suspects(rows):
    """Crops whose label disagrees with the photo, e.g. labelled 'Pliers' but only the pink
    shadow is visible (tool not put back), or labelled 'empty' but a tool is there.

    tool_share = share of the slot's footprint covered by pixels that differ from the
    empty-board photo (0 = looks exactly like the empty shadow). Per slot, the dividing
    line is halfway between the median of crops labelled 'empty' and of crops labelled
    with a tool (or a third of the tool median if the slot was never emptied).
    This only marks crops for you to look at; it never changes a label."""
    by_slot = {}
    for r in rows:
        if r["tool_share"] != "" and "swap_crop" not in r["tags"]:   # a small tool in a big slot is normal
            by_slot.setdefault(r["slot"], []).append(r)
    out = []
    for slot, rs in by_slot.items():
        full = [float(r["tool_share"]) for r in rs if r["label"] != EMPTY]
        emp = [float(r["tool_share"]) for r in rs if r["label"] == EMPTY]
        if not full:
            continue
        mf = float(np.median(full))
        line = (mf + float(np.median(emp))) / 2 if emp and np.median(emp) < mf else mf / 3
        for r in rs:
            sc = float(r["tool_share"])
            if r["label"] != EMPTY and sc < line:
                r["suspect"], r["suspect_why"] = 1, f"says {r['label'][:10]}, looks EMPTY"
            elif r["label"] == EMPTY and sc > line:
                r["suspect"], r["suspect_why"] = 1, "says empty, tool there"
            if r["suspect"]:
                out.append(r)
    return out


def write_sheet(path, items, per_row=6, size=170):
    tiles = []
    for img, title, why in items:
        t = cv2.resize(img, (size, size))
        t = cv2.copyMakeBorder(t, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, title, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(t, why[:30], (3, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 200), 1, cv2.LINE_AA)
        tiles.append(t)
    blank = np.full_like(tiles[0], 255)
    rows = [cv2.hconcat(tiles[i:i + per_row] + [blank] * (per_row - len(tiles[i:i + per_row])))
            for i in range(0, len(tiles), per_row)]
    cv2.imwrite(str(path), cv2.vconcat(rows))


def import_trials(tc_dirs, kit_tools, ai_cfg, include_void=False):
    from .pink import PINK_FILE
    if (ai_cfg["board"].get("footprint", {}).get("enabled") and ai_cfg.get("pink", {}).get("enabled")
            and not PINK_FILE.exists()):
        raise SystemExit("Run `python ai_main.py pink` first: the slot footprints are found from the pink shadows.")
    names = []
    print(f"Importing {len(tc_dirs)} toolcheck session(s):")
    for d in tc_dirs:
        if not (Path(d) / "session.json").exists():
            print(f"  {d}: SKIPPED, no session.json")
            continue
        n = import_session(d, kit_tools, ai_cfg, include_void)
        if n:
            names.append(n)
    print("Open dataset/<session>/preview.png for each session: every slot should show one tool, "
          "no floor, similar brightness.")
    print("Next: python ai_main.py train --test <held-out session>")
    return names
