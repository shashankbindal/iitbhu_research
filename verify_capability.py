"""
Capability check (CPU-friendly): can R_theta actually restore readability?

This is a deliberate *overfit* demonstration, not a generalization test. Heavy
deblurring needs a GPU + many iterations to generalize; what we verify here on
CPU is that the architecture + loss CAN turn an OCR-unreadable degraded image
into an OCR-readable one when given enough gradient steps on a small set. If it
can overfit a handful of images and the frozen OCR's CER drops, the design is
sound and the full GPU run is just "more data, more steps."

Uses a moderate motion-blur-dominant degradation (breaks OCR but is a well-posed
deblur), width=32, a few hundred steps on 4 fixed images.

    python verify_capability.py
"""

import os
import numpy as np
import cv2
import torch

from model import RestoreNet
from dataset import SyntheticTextPairs
from degrade import DegradeConfig
from train_stage1 import restoration_loss, make_restore_fn
from ocr import get_ocr
from metrics import cer

# moderate, restorable degradation: motion blur + mild noise/jpeg, no crushing low-light
CFG = DegradeConfig(
    p_motion_blur=1.0, motion_len=(11, 17),
    p_defocus=0.0, p_lowlight=0.0,
    p_noise=0.5, noise_sigma=(3, 8),
    p_jpeg=0.7, jpeg_quality=(45, 70),
)

KNOWN_TEXT = [
    "PARACETAMOL 500mg",
    "TAKE 1 TABLET DAILY",
    "EXP 08 2027",
    "BATCH A5R5",
]


def render(text, w=360, h=64):
    img = np.full((h, w, 3), 244, np.uint8)
    cv2.putText(img, text, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 25, 25), 2, cv2.LINE_AA)
    return img


def to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1)


def main():
    rng = np.random.default_rng(3)
    from degrade import degrade
    sharps = [render(t) for t in KNOWN_TEXT]
    degs = [degrade(s, CFG, seed=int(rng.integers(0, 1e6))) for s in sharps]

    deg_t = torch.stack([to_tensor(d) for d in degs])
    sharp_t = torch.stack([to_tensor(s) for s in sharps])

    model = RestoreNet(32)
    # small-random head init (instead of zero) so it learns faster in few steps
    for p in [model.head.weight, model.head.bias]:
        torch.nn.init.normal_(p, std=1e-3)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    print("Overfitting 4 fixed images (width=32)...")
    model.train()
    for step in range(1, 801):
        opt.zero_grad()
        loss = restoration_loss(model(deg_t), sharp_t)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}")

    # OCR before/after
    ocr = get_ocr("easyocr", gpu=False)
    restore = make_restore_fn(model, "cpu")
    out = os.path.join(os.path.dirname(__file__), "_selftest")
    os.makedirs(out, exist_ok=True)

    print("\nPer-image OCR (CER vs known text):")
    rows = []
    cer_deg, cer_res = [], []
    for i, (d, s, txt) in enumerate(zip(degs, sharps, KNOWN_TEXT)):
        res = restore(d)
        rd = ocr.read(d).text
        rr = ocr.read(res).text
        cd, cr = cer(rd, txt), cer(rr, txt)
        cer_deg.append(cd); cer_res.append(cr)
        print(f"  [{txt!r}]")
        print(f"     degraded OCR: {rd!r:40s} CER {cd:.2f}")
        print(f"     restored OCR: {rr!r:40s} CER {cr:.2f}")
        sep = np.full((4, d.shape[1], 3), 128, np.uint8)
        rows.append(np.vstack([s, sep, d, sep, res]))
    cv2.imwrite(os.path.join(out, "capability_sharp_deg_restored.png"), np.vstack(rows))

    md, mr = float(np.mean(cer_deg)), float(np.mean(cer_res))
    print(f"\nMean CER  degraded {md:.3f}  ->  restored {mr:.3f}  "
          f"({'IMPROVED' if mr < md else 'no gain'} by {md-mr:.3f})")
    print("Saved sharp/degraded/restored strip to _selftest/capability_sharp_deg_restored.png")


if __name__ == "__main__":
    main()
