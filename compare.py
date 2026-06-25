"""
Stage 3 (integration) — produce the paper's comparison table.

Scores every restoration condition on the SAME synthetic eval set with the SAME
frozen OCR, so the numbers are directly comparable. This is the table that goes
in the paper once R_theta is trained on the GPU:

    condition      CER      word-acc
    raw            ...      ...        (no restoration — the floor)
    RT-Focuser     ...      ...        (generic deblurrer baseline, optional)
    R_theta (S1)   ...      ...        (our model, stage-1 pretrained)
    R_theta (S2)   ...      ...        (our model, label-free VizWiz adapted)
    oracle         ...      ...        (sharp image — the ceiling)

    python compare.py --s1 checkpoints/r_theta_w48_stage1.pth \
                      --s2 checkpoints/r_theta_w48_stage2.pth \
                      [--rtfocuser ../models/rt_focuser.onnx]
"""

import argparse
import os
import numpy as np
import cv2
import torch

from evaluate import build_synthetic_evalset, evaluate_restore, identity, oracle_restore_factory
from ocr import get_ocr
from model import RestoreNet
from train_stage1 import make_restore_fn


def load_restore_fn(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device)
    model = RestoreNet(ck.get("width", 48)).to(device)
    model.load_state_dict(ck["model"])
    return make_restore_fn(model, device)


def rtfocuser_restore_fn(onnx_path):
    """Wrap the demo's RT-Focuser ONNX as a restore_fn for the baseline row."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def fn(bgr):
        h, w = bgr.shape[:2]
        ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
        padded = cv2.copyMakeBorder(bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = rgb.transpose(2, 0, 1)[None]
        y = sess.run(None, {name: x})[0][0]
        out = (y.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)[:h, :w]
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s1", default="", help="stage-1 checkpoint")
    p.add_argument("--s2", default="", help="stage-2 checkpoint")
    p.add_argument("--rtfocuser", default="", help="RT-Focuser ONNX (optional baseline)")
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evalset = build_synthetic_evalset(n=args.n, seed=2024)
    ocr = get_ocr("easyocr", gpu=(device == "cuda"))

    conditions = [("raw", identity)]
    if args.rtfocuser and os.path.exists(args.rtfocuser):
        conditions.append(("RT-Focuser", rtfocuser_restore_fn(args.rtfocuser)))
    if args.s1 and os.path.exists(args.s1):
        conditions.append(("R_theta (S1)", load_restore_fn(args.s1, device)))
    if args.s2 and os.path.exists(args.s2):
        conditions.append(("R_theta (S2)", load_restore_fn(args.s2, device)))
    conditions.append(("oracle", oracle_restore_factory(evalset)))

    print(f"\nComparison on {args.n} synthetic eval images (frozen OCR: EasyOCR):\n")
    print(f"  {'condition':14s}  {'CER':>6s}  {'word-acc':>8s}")
    print("  " + "-" * 32)
    for name, fn in conditions:
        rep, _ = evaluate_restore(evalset, fn, ocr)
        print(f"  {name:14s}  {rep.mean_cer:6.3f}  {rep.word_accuracy:8.3f}")
    print("\n(With trained weights this is the paper's headline table. Smoke-weight")
    print(" runs show the harness; real numbers come from the GPU Stage-1/2 runs.)")


if __name__ == "__main__":
    main()
