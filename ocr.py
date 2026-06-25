"""
Frozen OCR wrapper — the recognizer used to SUPERVISE and EVALUATE restoration.

Two distinct roles in this project:
  1. Evaluation (this file): full-image detection+recognition to score whether a
     restored image is readable. Any off-the-shelf OCR works; we use EasyOCR.
  2. Training supervision (later, separate module): a *differentiable* recognizer
     (e.g. PARSeq/TrOCR) whose per-character logits drive the confidence loss
     L_conf. That needs gradient flow, so it is kept separate from this eval wrapper.

This wrapper is intentionally backend-agnostic: `OCREngine` defines the interface,
and concrete backends implement `.read()`. A `MockOCR` backend lets the rest of the
harness be unit-tested without the heavy model download.

Usage:
    from ocr import get_ocr
    ocr = get_ocr("easyocr")          # or "mock" for tests
    result = ocr.read(img_bgr)        # -> OCRResult(text, confidence, boxes)
"""

import os
# EasyOCR/OpenCV and PyTorch both link an OpenMP runtime on Windows, which
# triggers "libiomp5md.dll already initialized" (OMP Error #15). This makes the
# duplicate load non-fatal. Set before torch/easyocr import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class OCRResult:
    text: str                      # all recognized text, reading order, space-joined
    confidence: float              # mean per-word confidence in [0,1]
    boxes: List[Tuple] = field(default_factory=list)   # [(box, word, conf), ...]


class OCREngine:
    """Interface every backend implements."""
    def read(self, img_bgr: np.ndarray) -> OCRResult:
        raise NotImplementedError


class MockOCR(OCREngine):
    """Deterministic stand-in for tests — 'reads' text painted into a side channel,
    or just returns empty. Lets harness logic be exercised with no model download."""
    def __init__(self, scripted=None):
        # scripted: optional dict id(img)->text for controllable tests
        self.scripted = scripted or {}

    def read(self, img_bgr):
        return OCRResult(text=self.scripted.get(id(img_bgr), ""), confidence=0.0)


class EasyOCRBackend(OCREngine):
    """EasyOCR: self-contained PyTorch detection+recognition, CPU or CUDA.
    Downloads its models on first construction."""
    def __init__(self, langs=("en",), gpu=False):
        import easyocr
        self.reader = easyocr.Reader(list(langs), gpu=gpu, verbose=False)

    def read(self, img_bgr):
        # EasyOCR expects RGB or a path; it accepts numpy arrays (BGR works, but
        # convert for correctness). detail=1 -> [(box, text, conf), ...]
        import cv2
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        dets = self.reader.readtext(rgb, detail=1, paragraph=False)
        if not dets:
            return OCRResult(text="", confidence=0.0, boxes=[])
        words = [d[1] for d in dets]
        confs = [float(d[2]) for d in dets]
        return OCRResult(
            text=" ".join(words),
            confidence=sum(confs) / len(confs),
            boxes=[(d[0], d[1], float(d[2])) for d in dets],
        )


_REGISTRY = {"easyocr": EasyOCRBackend, "mock": MockOCR}
_cache = {}


def get_ocr(backend="easyocr", **kwargs) -> OCREngine:
    """Construct (and cache) an OCR backend. Caches because model loading is slow."""
    key = (backend, tuple(sorted(kwargs.items())))
    if key not in _cache:
        if backend not in _REGISTRY:
            raise ValueError(f"Unknown OCR backend '{backend}'. Options: {list(_REGISTRY)}")
        _cache[key] = _REGISTRY[backend](**kwargs)
    return _cache[key]


if __name__ == "__main__":
    # End-to-end smoke test on the degradation self-test images, if present.
    import os, cv2
    from metrics import cer, evaluate_readability

    d = os.path.join(os.path.dirname(__file__), "_selftest")
    if not os.path.exists(os.path.join(d, "sharp.png")):
        print("Run `python degrade.py` first to create _selftest images."); raise SystemExit

    truth = "PARACETAMOL 500mg Take 1 tablet twice daily Exp 08/2027 Batch A5R5"
    try:
        ocr = get_ocr("easyocr", gpu=False)
    except ImportError:
        print("EasyOCR not installed — `pip install easyocr`. (Harness still importable.)")
        raise SystemExit

    sharp = cv2.imread(os.path.join(d, "sharp.png"))
    deg   = cv2.imread(os.path.join(d, "degraded_0.png"))

    r_sharp = ocr.read(sharp)
    r_deg   = ocr.read(deg)
    print("SHARP    read:", repr(r_sharp.text), f"(conf {r_sharp.confidence:.2f})")
    print("  CER vs truth:", round(cer(r_sharp.text, truth), 3))
    print("DEGRADED read:", repr(r_deg.text), f"(conf {r_deg.confidence:.2f})")
    print("  CER vs truth:", round(cer(r_deg.text, truth), 3))
    print()
    print("This gap (degraded CER >> sharp CER) is the headroom restoration must recover.")
