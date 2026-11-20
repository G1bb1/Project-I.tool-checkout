"""Train the slot classifier on the PC (REQ-TR-05: trained on the computer).

Split is BY SESSION, never by image: crops from one capture are near-identical,
so mixing them across train and test would make accuracy look better than it is.
The test sessions are named in the model's meta.json and must not be used for
any decision before the evaluation (same rule as toolcheck's thresholds).

Output: models/<stamp>-<type>/
    meta.json            classes, input size, crop margin, sessions used, settings
    model.keras          (cnn) Keras model
    model.tflite         (cnn) for the processor: dynamic-range quantised
    model_int8.tflite    (cnn) fully int8, for a microcontroller route later
    model.joblib         (svm) scaler + SVM
    train_report.json    per-class counts, validation accuracy, training time
"""
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .augment import make_augmenter
from .common import (DATASET, MODELS, class_names, list_sessions, load_ai_config, read_manifest,
                     row_skip_reason, save_json)
from .features import model_input
from .pink import load_pink


def load_crops(sessions, classes, size, kind, pink=None, aug_cfg=None, seed=0):
    """Returns X, y, group (index of the original crop; copies share it so a crop and
    its augmented copies never end up on both sides of the validation split)."""
    X, y, groups = [], [], []
    idx = {c: i for i, c in enumerate(classes)}
    copies = (aug_cfg or {}).get("copies", 0)
    aug = make_augmenter(aug_cfg, np.random.default_rng(seed)) if copies else None
    g = 0
    ai_cfg = load_ai_config()
    skipped = Counter()
    for s in sessions:
        for row in read_manifest(DATASET / s):
            why = row_skip_reason(s, row, ai_cfg, training=True)
            if why:
                skipped[why] += 1
                continue
            if row["label"] not in idx:
                raise SystemExit(f"{s}: label {row['label']!r} is not a known class. "
                                 "Was the dataset collected with a different kit?")
            crop = cv2.imread(str(DATASET / s / row["file"]))
            X.append(model_input(crop, size, kind, pink))
            y.append(idx[row["label"]])
            groups.append(g)
            for _ in range(copies):
                X.append(model_input(crop, size, kind, pink, aug))
                y.append(idx[row["label"]])
                groups.append(g)
            g += 1
    if skipped:
        print("Left out of training:", dict(skipped))
    return np.array(X), np.array(y), np.array(groups)


def confidence_table(conf, correct):
    """What the validation crops say about each confidence limit (to freeze ai.json BEFORE
    the held-out evaluation). Validation crops come from the training sessions."""
    out = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        keep = conf >= t
        out.append({"limit": t, "share_above": round(float(keep.mean()), 4),
                    "wrong_above": int((keep & ~correct).sum()),
                    "right_below": int((~keep & correct).sum())})
    return out


def _originals(grp):
    """True for the first (un-augmented) entry of each group."""
    first = np.ones(len(grp), bool)
    first[1:] = grp[1:] != grp[:-1]
    return first


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    total = counts.sum()
    return {i: (total / (n_classes * c)) if c else 0.0 for i, c in enumerate(counts)}


def build_cnn(size, n_classes, base):
    import tensorflow as tf
    from tensorflow.keras import layers

    # Augmentation is done beforehand in numpy (augment.py), so the exported
    # model has no random layers and the full-int8 export works.
    inp = layers.Input((size, size, 3), dtype="float32")
    x = inp
    if base == "mobilenet":
        # Needs the ImageNet weights download on first use (internet on the PC).
        x = layers.Rescaling(1 / 127.5, offset=-1)(x)
        bb = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3), alpha=0.35,
                                               include_top=False, weights="imagenet")
        bb.trainable = False
        x = bb(x, training=False)
    else:
        # Small CNN from scratch: no download, light enough for a Pi Zero 2 W.
        x = layers.Rescaling(1 / 255.0)(x)
        for f in (16, 32, 64, 96):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def export_tflite(model, X_rep, out_dir):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    (out_dir / "model.tflite").write_bytes(conv.convert())

    def rep():
        for i in range(min(200, len(X_rep))):
            yield [X_rep[i:i + 1].astype(np.float32)]
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    try:
        (out_dir / "model_int8.tflite").write_bytes(conv.convert())
    except Exception as e:      # augmentation layers can block full-int8; float model still works
        print(f"  (int8 export skipped: {e.__class__.__name__})")


def train(slots, ai_cfg, kind, test_sessions=None, train_sessions=None, epochs=None, base="small"):
    tools = [s["tool"] for s in slots]
    classes = class_names(tools)
    size = ai_cfg["input_size"]
    all_s = list_sessions()
    if not all_s:
        raise SystemExit("No dataset yet. Run: python ai_main.py collect --n 60")
    if test_sessions is None:
        test_sessions = [all_s[-1]] if len(all_s) > 1 else []
    if train_sessions is None:
        train_sessions = [s for s in all_s if s not in test_sessions]
    if not train_sessions:
        raise SystemExit("Need at least one session to train on that is not a test session.")
    if not test_sessions:
        print("WARNING: nothing is held out, so `evaluate` has nothing to score. The next "
              "hybrid session (run with the frozen ai.json) is this model's held-out test.")

    overlap = set(test_sessions) & set(train_sessions)
    if overlap:
        raise SystemExit(f"{sorted(overlap)} cannot be both trained on and held out.")
    pink = load_pink(ai_cfg)
    aug_cfg = ai_cfg.get("augment", {"copies": 0})
    print(f"Training {kind} on {train_sessions}; held out for evaluation: {test_sessions}")
    print(f"Pink removal: {'off' if not pink else pink['source']}; "
          f"augmented copies per crop: {aug_cfg.get('copies', 0)}")
    X, y, grp = load_crops(train_sessions, classes, size, kind, pink, aug_cfg, ai_cfg.get("seed", 0))
    counts = Counter(classes[i] for i in y[_originals(grp)])
    print("Crops per class:", dict(counts))
    missing = [c for c in classes if counts.get(c, 0) == 0]
    if missing:
        print(f"WARNING: no training crops for {missing}; the model can never predict them.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = MODELS / f"{stamp}-{kind}"
    out.mkdir(parents=True, exist_ok=True)
    # Validation = 15% of the ORIGINAL crops (whole groups), and only their
    # un-augmented version is scored.
    rng = np.random.default_rng(ai_cfg.get("seed", 0))
    ug = rng.permutation(np.unique(grp))
    val_groups = set(ug[:max(1, int(0.15 * len(ug)))].tolist())
    in_val = np.array([g in val_groups for g in grp])
    first = _originals(grp)
    val = np.where(in_val & first)[0]
    tr = np.where(~in_val)[0]
    n_val = len(val)
    t0 = time.perf_counter()

    if kind == "svm":
        import joblib
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.calibration import CalibratedClassifierCV
        # Calibrated SVM so it gives a confidence (0-1) like the CNN.
        clf = make_pipeline(StandardScaler(), CalibratedClassifierCV(
            SVC(C=10, gamma="scale", class_weight="balanced"), cv=3, ensemble=False))
        clf.fit(X[tr], y[tr])
        pv = clf.predict_proba(X[val])
        val_pred = clf.classes_[pv.argmax(1)]
        val_conf = pv.max(1)
        val_acc = float((val_pred == y[val]).mean())
        clf.fit(X, y)                                   # final model on all training sessions
        joblib.dump(clf, out / "model.joblib")
        files = ["model.joblib"]
    else:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(ai_cfg.get("seed", 0))
        model = build_cnn(size, len(classes), base)
        cb = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                               monitor="val_accuracy")]
        hist = model.fit(X[tr].astype(np.float32), y[tr],
                         validation_data=(X[val].astype(np.float32), y[val]),
                         epochs=epochs or ai_cfg["epochs"], batch_size=32,
                         class_weight=class_weights(y[tr], len(classes)),
                         callbacks=cb, verbose=2)
        pv = model.predict(X[val].astype(np.float32), verbose=0)
        val_pred, val_conf = pv.argmax(1), pv.max(1)
        val_acc = float((val_pred == y[val]).mean())
        model.save(out / "model.keras")
        export_tflite(model, X[tr], out)
        files = [p.name for p in out.iterdir() if p.suffix in (".keras", ".tflite")]

    train_s = time.perf_counter() - t0
    meta = {"type": kind, "base": base if kind == "cnn" else None, "classes": classes,
            "input_size": size, "crop_margin": ai_cfg["crop_margin"], "pink": pink, "board": ai_cfg["board"],
            "augment": aug_cfg,
            "train_sessions": train_sessions, "test_sessions": test_sessions,
            "created": datetime.now().isoformat(timespec="seconds"), "files": sorted(files)}
    save_json(out / "meta.json", meta)
    save_json(out / "train_report.json", {
        "crops_per_class": dict(counts), "train_crops_incl_copies": int(len(tr)),
        "val_crops": int(n_val),
        "val_accuracy_same_sessions": val_acc, "train_seconds": round(train_s, 1),
        "val_confidence_table": confidence_table(val_conf, val_pred == y[val]),
        "note": "Validation crops come from the training sessions, so this number is optimistic. "
                "Report the held-out evaluation instead."})
    sizes = {f: (out / f).stat().st_size for f in files}
    print(f"\nSaved {out}")
    print(f"  validation accuracy (same sessions, optimistic): {val_acc:.3f}")
    print(f"  file sizes (bytes): {sizes}")
    print("  validation confidence (use it to freeze ai.json 'decision' and 'lighting_override' NOW):")
    print("    limit  share>=limit  wrong>=limit  right<limit")
    for r in confidence_table(val_conf, val_pred == y[val]):
        print(f"    {r['limit']:.2f}   {r['share_above']:10.1%}  {r['wrong_above']:12d}  {r['right_below']:11d}")
    print(f"Next: python ai_main.py evaluate {out.name}")
    return out
