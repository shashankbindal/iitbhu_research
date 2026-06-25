"""
Evaluation harness — the Weeks 1-2 deliverable that produces baseline numbers.

Pipeline:  image --(restore_fn)--> restored --(frozen OCR)--> text --(metrics)--> CER/word-acc

The restoration step is PLUGGABLE so the same harness scores every condition on
identical data:
    * raw            : restore_fn = identity            (the floor)
    * oracle         : restore_fn = return sharp image  (the ceiling, synthetic only)
    * RT-Focuser     : restore_fn = rt_focuser(img)     (baseline #2 in the proposal)
    * ours (R_theta) : restore_fn = model(img)          (the method)

On the synthetic set we know the exact ground-truth text. On VizWiz we use the
consensus answer as a weak text target (substring 'contains' is the meaningful
metric there).

Run:
    python evaluate.py                 # synthetic demo: raw vs oracle headroom
"""

import os
import numpy as np
import cv2

from degrade import degrade
from ocr import get_ocr
from metrics import evaluate_readability, cer


# ── synthetic eval set (known ground-truth text) ──────────────────────────────

_LABELS = [
    ["PARACETAMOL 500mg", "Take 1 tablet twice daily", "Exp 08/2027 Batch A5R5"],
    ["AMOXICILLIN 250mg", "3 times a day after food", "Keep below 25C"],
    ["WHOLE MILK 1 Litre", "Best before 12 NOV", "Pasteurised Homogenised"],
    ["IBUPROFEN 200mg", "Do not exceed 6 in 24h", "Lot 7741 Exp 2026-03"],
    ["VITAMIN D3 1000 IU", "One capsule daily", "60 softgels"],
    ["INSTANT COFFEE 100g", "Add hot water and stir", "Origin Colombia"],
]


def render_label(lines, w=680, line_h=58, pad=24):
    """Render sharp dark-on-light label text — stand-in for a clean document/label."""
    h = pad * 2 + line_h * len(lines)
    img = np.full((h, w, 3), 244, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (pad, pad + 40 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05, (25, 25, 25), 2, cv2.LINE_AA)
    return img


def build_synthetic_evalset(n=6, seed=0):
    """Return [(degraded_bgr, sharp_bgr, gt_text), ...] with known text."""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        lines = _LABELS[i % len(_LABELS)]
        sharp = render_label(lines)
        deg = degrade(sharp, seed=int(rng.integers(0, 1_000_000)))
        items.append((deg, sharp, " ".join(lines)))
    return items


# ── core evaluation ───────────────────────────────────────────────────────────

def identity(img):
    return img


def evaluate_restore(evalset, restore_fn, ocr, label=""):
    """Run restore_fn -> OCR -> metrics over an eval set; return a report."""
    preds, gts = [], []
    for deg, _sharp, gt in evalset:
        restored = restore_fn(deg)
        preds.append(ocr.read(restored).text)
        gts.append(gt)
    report = evaluate_readability(preds, gts)
    if label:
        print(f"  {label:14s}: {report}")
    return report, preds


def oracle_restore_factory(evalset):
    """Restoration upper bound: replace each degraded image with its sharp source.
    Maps by array identity, so call with the same evalset objects."""
    lookup = {id(deg): sharp for deg, sharp, _ in evalset}
    return lambda img: lookup.get(id(img), img)


if __name__ == "__main__":
    print("Building synthetic eval set (known ground-truth text)...")
    evalset = build_synthetic_evalset(n=6, seed=0)

    print("Loading frozen OCR (EasyOCR)...")
    ocr = get_ocr("easyocr", gpu=False)

    print("\nReadability by restoration condition (lower CER / higher word-acc = better):")
    raw_rep, _   = evaluate_restore(evalset, identity, ocr, "raw (floor)")
    orc_rep, _   = evaluate_restore(evalset, oracle_restore_factory(evalset), ocr, "oracle (ceiling)")

    print("\nHeadroom restoration must recover:")
    print(f"  CER       {raw_rep.mean_cer:.3f}  ->  {orc_rep.mean_cer:.3f}   "
          f"(reduction available: {raw_rep.mean_cer - orc_rep.mean_cer:.3f})")
    print(f"  word-acc  {raw_rep.word_accuracy:.3f}  ->  {orc_rep.word_accuracy:.3f}")
    print("\nWhen R_theta is trained, plug it in as restore_fn and its numbers should")
    print("land between 'raw' and 'oracle'. That gap closed = the paper's headline result.")
