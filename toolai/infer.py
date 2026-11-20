"""Run a trained model on slot crops: the code that would run on the custom hardware.

For a CNN this uses the .tflite file (what gets deployed), not the Keras model.
Interpreter lookup order: ai_edge_litert (new name) -> tflite_runtime -> tensorflow.
On a Pi install ONE of:  pip install ai-edge-litert   or   pip install tflite-runtime
"""
import time
from pathlib import Path

import numpy as np

from .common import EMPTY, MODELS, crop_slot, load_json
from .features import model_input


def _interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
    return Interpreter(model_path=str(path), num_threads=4)


def resolve_model_dir(name=None):
    if name:
        p = Path(name)
        return p if p.is_absolute() else MODELS / name
    dirs = sorted(d for d in MODELS.iterdir() if (d / "meta.json").exists()) if MODELS.exists() else []
    if not dirs:
        raise SystemExit("No trained model. Run: python ai_main.py train")
    return dirs[-1]


class Predictor:
    def __init__(self, model_dir, use_int8=False):
        self.dir = Path(model_dir)
        self.meta = load_json(self.dir / "meta.json")
        self.classes = self.meta["classes"]
        self.size = self.meta["input_size"]
        self.margin = self.meta["crop_margin"]
        self.kind = self.meta["type"]
        self.pink = self.meta.get("pink")      # frozen at training time
        self.board = None
        if self.kind == "svm":
            import joblib
            self.model_file = self.dir / "model.joblib"
            self.clf = joblib.load(self.model_file)
        else:
            name = "model_int8.tflite" if use_int8 else "model.tflite"
            self.model_file = self.dir / name
            self.itp = _interpreter(self.model_file)
            self.inp = self.itp.get_input_details()[0]
            self.out = self.itp.get_output_details()[0]
            n = len(self.classes)
            # One batch of all slots at once.
            self._batch = None
            self._n = n

    def use_board(self, slots, empty_path, source=""):
        """Cut slots exactly as the training crops were cut (footprints + exposure), using the
        empty-board photo of THIS camera setup. Models trained before this existed: no-op."""
        from .board import make_board
        if self.meta.get("board"):
            self.board = make_board(slots, self.margin, self.meta["board"], self.pink, empty_path, source)
        return self

    def crops(self, img, slots):
        if self.board is not None:
            return self.board.crops(img)
        return [crop_slot(img, s, self.margin) for s in slots]

    def model_size_kb(self):
        return self.model_file.stat().st_size / 1024

    def prepare(self, crops):
        return np.stack([model_input(c, self.size, self.kind, self.pink) for c in crops])

    def predict(self, X):
        """Returns (class indices, confidences 0-1)."""
        if self.kind == "svm":
            if hasattr(self.clf, "predict_proba"):
                p = self.clf.predict_proba(X)
                full = np.zeros((len(X), len(self.classes)), np.float32)
                full[:, self.clf.classes_] = p
                return full.argmax(1), full.max(1)
            return self.clf.predict(X), np.ones(len(X), np.float32)   # older model: no confidence
        b = len(X)
        if self._batch != b:
            self.itp.resize_tensor_input(self.inp["index"], [b, self.size, self.size, 3])
            self.itp.allocate_tensors()
            self._batch = b
        if self.inp["dtype"] == np.uint8:
            data = X.astype(np.uint8)
        else:
            data = X.astype(np.float32)
        self.itp.set_tensor(self.inp["index"], data)
        self.itp.invoke()
        p = self.itp.get_tensor(self.out["index"]).astype(np.float32)
        if self.out["dtype"] == np.uint8:
            scale, zero = self.out["quantization"]
            p = (p - zero) * scale
        return p.argmax(1), p.max(1)


def classify_slots(pred, img, slots):
    """[(content, confidence)] for every slot of one photo, plus seconds taken."""
    t0 = time.perf_counter()
    X = pred.prepare(pred.crops(img, slots))
    idx, conf = pred.predict(X)
    out = [(pred.classes[int(i)], float(c)) for i, c in zip(idx, conf)]
    b = pred.board
    if b is not None and b.tool_only and len(b.last_content) == len(out):
        lim = b.cfg.get("tool_only", {}).get("empty_share", 0.0)
        # Too little changed vs the empty board: empty by rule, not by the model.
        out = [(EMPTY, 1.0) if share < lim else o for o, share in zip(out, b.last_content)]
    return out, time.perf_counter() - t0


def check_frame(pred, img, slots):
    """One full check on one photo. Returns (per-slot results, stage timings in seconds)."""
    t0 = time.perf_counter()
    crops = pred.crops(img, slots)
    X = pred.prepare(crops)
    t1 = time.perf_counter()
    idx, conf = pred.predict(X)
    t2 = time.perf_counter()
    results = []
    for k, s in enumerate(slots):
        content = pred.classes[int(idx[k])]
        results.append({"slot": s["tool"], "content": content,
                        "confidence": None if conf is None else float(conf[k]),
                        "flag": content != s["tool"],
                        "reason": None if content == s["tool"] else
                        ("missing" if content == EMPTY else f"wrong tool: {content}")})
    t3 = time.perf_counter()
    return results, {"crop_prepare_s": t1 - t0, "inference_s": t2 - t1, "decide_s": t3 - t2}
