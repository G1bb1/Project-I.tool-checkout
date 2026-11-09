"""Project paths and small JSON helpers."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
REFS = ROOT / "refs"
DATA = ROOT / "data"

KIT_FILE = CONFIG / "kit.json"
SETTINGS_FILE = CONFIG / "settings.json"
SLOTS_FILE = CONFIG / "slots.json"
CURRENT_REFS_FILE = REFS / "current.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_settings():
    return load_json(SETTINGS_FILE)


def load_kit():
    return load_json(KIT_FILE)


def load_slots():
    if not SLOTS_FILE.exists():
        raise SystemExit("No config/slots.json yet. Run: python main.py define-slots")
    return load_json(SLOTS_FILE)["slots"]


def current_refs_dir():
    if not CURRENT_REFS_FILE.exists():
        raise SystemExit("No references yet. Run: python main.py capture-refs")
    return REFS / load_json(CURRENT_REFS_FILE)["dir"]
