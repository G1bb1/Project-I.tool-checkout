"""Inputs for the two model types. The same preprocessing runs in training,
evaluation, replay and live checks, with the pink values frozen in the model's
meta.json, so a model always sees crops prepared exactly as it was trained.

    crop -> letterbox to size x size -> (training only: random lighting/placement)
         -> pink painted grey -> model

cnn: the prepared crop itself (uint8; scaling happens inside the model).
svm: hand-made features: HOG on greyscale (shape) + a LAB (a, b) colour histogram
     (colour, e.g. the two tapes). A classical baseline to compare the CNN against.
"""
import cv2
import numpy as np

from .common import letterbox
from .pink import remove_pink

_HOG = None


def _hog(size):
    global _HOG
    if _HOG is None or _HOG[0] != size:
        cell = size // 8
        _HOG = (size, cv2.HOGDescriptor((size, size), (cell * 2, cell * 2),
                                        (cell, cell), (cell, cell), 9))
    return _HOG[1]


def prepare_image(crop, size, pink=None, augment_fn=None):
    img = letterbox(crop, size)
    if augment_fn is not None:
        img = augment_fn(img)
    return remove_pink(img, pink)


def svm_from_image(img, size):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = _hog(size).compute(grey).ravel()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [1, 2], None, [8, 8], [0, 256, 0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-6)
    return np.concatenate([h, hist * 10.0]).astype(np.float32)


def model_input(crop, size, kind, pink=None, augment_fn=None):
    img = prepare_image(crop, size, pink, augment_fn)
    return img if kind == "cnn" else svm_from_image(img, size)
