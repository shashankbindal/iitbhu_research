"""
Phase 1 — real-photo smoke test (qualitative, no ground truth).

All numbers so far are on synthetic degradation. This checks the Stage-1
checkpoint isn't doing something synthetic-specific that won't transfer to real
phone photos. For each image in real_photos/, runs raw / RT-Focuser / R_theta,
then EasyOCR on each, and prints the three OCR strings side by side. Saves the
restored images to smoke_test_outputs/ for visual review.

No CER (no ground truth) — this is a judgment call: does R_theta's output look
reasonable on real photos, or broken (artifacts, colour shift, hallucination)?

Setup: drop 5-10 real phone photos containing text into research/real_photos/
(blurry / low-light preferred to match the target distribution), then:

    python smoke_test_real.py
"""

import os
import glob
import torch
import cv2

from model import RestoreNet
from train_stage1 import make_restore_fn
from ocr import get_ocr

CKPT = "checkpoints/r_theta_w48_stage1.pth"
PHOTOS_DIR = "real_photos"
OUT_DIR = "smoke_test_outputs"
RTFOCUSER = "../models/rt_focuser.onnx"


def main():
    paths = sorted(sum([glob.glob(os.path.join(PHOTOS_DIR, e))
                        for e in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")], []))
    if not paths:
        print(f"No images in {PHOTOS_DIR}/. Drop 5-10 real phone photos there first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=device)
    model = RestoreNet(ck.get("width", 48)).to(device)
    model.load_state_dict(ck["model"])
    r_theta = make_restore_fn(model, device)

    rtf = None
    if os.path.exists(RTFOCUSER):
        from compare import rtfocuser_restore_fn
        rtf = rtfocuser_restore_fn(RTFOCUSER)

    ocr = get_ocr("easyocr", gpu=(device == "cuda"))
    os.makedirs(OUT_DIR, exist_ok=True)

    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            print(f"skip (unreadable): {p}"); continue
        name = os.path.splitext(os.path.basename(p))[0]
        print(f"\n=== {os.path.basename(p)}  ({bgr.shape[1]}x{bgr.shape[0]}) ===")

        raw_txt = ocr.read(bgr).text
        print(f"  raw       : {raw_txt!r}")

        if rtf is not None:
            rtf_img = rtf(bgr)
            print(f"  RT-Focuser: {ocr.read(rtf_img).text!r}")
            cv2.imwrite(os.path.join(OUT_DIR, f"{name}_rtfocuser.png"), rtf_img)

        rt_img = r_theta(bgr)
        print(f"  R_theta   : {ocr.read(rt_img).text!r}")
        cv2.imwrite(os.path.join(OUT_DIR, f"{name}_rtheta.png"), rt_img)
        cv2.imwrite(os.path.join(OUT_DIR, f"{name}_raw.png"), bgr)

    print(f"\nRestored images saved to {OUT_DIR}/ for visual review.")
    print("Judgment call: does R_theta look reasonable, or broken (artifacts / colour")
    print("shift / hallucinated texture)? Record the verdict in RESULTS.md (Phase 1).")


if __name__ == "__main__":
    main()
