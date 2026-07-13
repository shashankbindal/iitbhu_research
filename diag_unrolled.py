"""
Diagnostic: why did the unrolled restorer score WORSE than raw on the frozen-61
set (0.553 vs 0.536)? Runs a handful of samples, prints GT/raw/restored OCR text,
the estimated kernel's centre-tap mass and alpha/beta, and saves before/after PNGs
for visual inspection.
"""
import cv2
import torch

from held_out_eval import build
from unrolled import UnrolledRestorer
from train_stage1 import make_restore_fn
from ocr import get_ocr
from metrics import cer

CKPT = "checkpoints/r_theta_w48_stage1_unrolled.pth"
N = 6

device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(CKPT, map_location=device)
model = UnrolledRestorer().to(device)
model.load_state_dict(ck["model"])
model.eval()
restore = make_restore_fn(model, device)
ocr = get_ocr("easyocr", gpu=(device == "cuda"))

items = build()[:N]
for i, (deg, sharp, gt) in enumerate(items):
    out = restore(deg)
    raw_txt = ocr.read(deg).text
    res_txt = ocr.read(out).text
    k, a, b = model.estimator(make_restore_fn.__globals__["torch"].from_numpy(
        cv2.cvtColor(deg, cv2.COLOR_BGR2RGB).astype("float32").transpose(2, 0, 1)[None] / 255.0
    ).to(device))
    c = k.shape[-1] // 2
    print(f"[{i}] GT      : {gt!r}")
    print(f"    raw OCR : {raw_txt!r}  (CER {cer(raw_txt, gt):.2f})")
    print(f"    restored: {res_txt!r}  (CER {cer(res_txt, gt):.2f})")
    print(f"    kernel centre-mass={k[0,0,c,c].item():.3f}  alpha={a.item():.3f}  beta={b.item():.3f}")
    cv2.imwrite(f"_selftest/diag_{i}_deg.png", deg)
    cv2.imwrite(f"_selftest/diag_{i}_out.png", out)
    cv2.imwrite(f"_selftest/diag_{i}_sharp.png", sharp)
    print()

print("Saved before/after/sharp PNGs to _selftest/diag_*.png")
