"""
Phase 5 — controlled Config A vs Config B head-to-head.

Identical protocol to Phase 2: the frozen 61-label set (held_out_eval_v2.json),
the same EasyOCR, the same per-index deterministic degradation. The ONLY thing
that differs between the two R_theta rows is the architecture (Config B adds the
recognition-guided FiLM bottleneck; everything else — data, recipe, iters, seed —
is identical). So any CER gap is attributable to the architecture change.

Conditions: raw / RT-Focuser / R_theta-A / R_theta-B / oracle(sharp).

    python phase5_compare.py
"""

import os
import numpy as np
import torch

from held_out_eval import build, load_model
from train_stage1 import make_restore_fn
from compare import rtfocuser_restore_fn
from model import count_params
from ocr import get_ocr
from metrics import cer

RTF_ONNX = "../models/rt_focuser.onnx"
CKPT_A = "checkpoints/r_theta_w48_stage1.pth"
CKPT_B = "checkpoints/r_theta_w48_stage1_configB.pth"
ONNX_A = "checkpoints/r_theta_w48.onnx"


def stats(vals):
    a = np.array(vals)
    return a.mean(), a.std(), a.min(), a.max()


def trainable_M(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def mb(path):
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else float("nan")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    items = build()                       # [(deg, sharp, gt), ...] — frozen 61-set
    ocr = get_ocr("easyocr", gpu=(device == "cuda"))

    model_a = load_model(CKPT_A, device)
    model_b = load_model(CKPT_B, device)
    restore_a = make_restore_fn(model_a, device)
    restore_b = make_restore_fn(model_b, device)

    conditions = [("raw", lambda bgr: bgr)]
    if os.path.exists(RTF_ONNX):
        conditions.append(("RT-Focuser", rtfocuser_restore_fn(RTF_ONNX)))
    conditions += [
        ("R_theta Config A", restore_a),
        ("R_theta Config B", restore_b),
        ("oracle (sharp)", None),         # special-cased: OCR the sharp image
    ]

    # Pre-read OCR per condition over the whole frozen set.
    results = {name: [] for name, _ in conditions}
    for deg, sharp, gt in items:
        for name, fn in conditions:
            img = sharp if name.startswith("oracle") else fn(deg)
            results[name].append(cer(ocr.read(img).text, gt))

    # Params / size annotations.
    extra = {
        "R_theta Config A": f"{trainable_M(model_a):.2f} M / {mb(ONNX_A):.2f} MB",
        "R_theta Config B": f"{trainable_M(model_b):.2f} M (+rec) / {mb(CKPT_B):.2f} MB ckpt",
    }
    if os.path.exists(RTF_ONNX):
        extra["RT-Focuser"] = f"5.85 M / {mb(RTF_ONNX):.2f} MB"

    print(f"\nFrozen held-out set: {len(items)} labels   OCR: EasyOCR\n")
    hdr = f"{'Condition':<20} {'CER mean':>9} {'std':>6} {'min':>5} {'max':>5}  {'params / size':<28}"
    print(hdr); print("-" * len(hdr))
    for name, _ in conditions:
        m, s, lo, hi = stats(results[name])
        print(f"{name:<20} {m:>9.3f} {s:>6.3f} {lo:>5.2f} {hi:>5.2f}  {extra.get(name, ''):<28}")

    # Decision gate (vs Phase-2 per-sample std as the noise floor).
    a_mean = np.mean(results["R_theta Config A"])
    b_mean = np.mean(results["R_theta Config B"])
    a_std = np.std(results["R_theta Config A"])
    delta = a_mean - b_mean               # positive => B better (lower CER)
    se = a_std / np.sqrt(len(items))
    print("\n--- Decision ---")
    print(f"Config A CER = {a_mean:.3f},  Config B CER = {b_mean:.3f}")
    print(f"delta (A - B) = {delta:+.3f}   (per-sample std {a_std:.3f}, SE {se:.3f})")
    if delta > se:
        print(f"=> Config B is LOWER by >1 SE ({delta:+.3f} > {se:.3f}): carry B to Stage 2.")
    elif delta > 0:
        print(f"=> Config B lower but within noise ({delta:+.3f} <= 1 SE {se:.3f}): NOT decisive.")
    else:
        print(f"=> Config B NOT better ({delta:+.3f}): architecture change alone did not "
              f"improve CER at Stage-1 scale; carry A forward.")


if __name__ == "__main__":
    main()
