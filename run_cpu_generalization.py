"""
A real (reduced-scale) Stage-1 training run that measures GENERALIZATION.

Context: the dev sandbox cannot use the GPU (CUDA torch wheels are unreachable;
CUDA available == False), so the full 60k-iter GPU run in TRAIN_ON_GPU.md cannot
execute here. This script runs a genuine but smaller CPU training and then
evaluates on UNSEEN held-out label images — turning the earlier overfit
capability proof into a real generalization result (held-out improvement > 0).

Scope honesty: trains on a motion-blur-dominant degradation (the primary
blind-capture failure mode — camera shake); the full composite distribution
(heavy low-light, strong defocus) needs the GPU run. Train and held-out eval use
the SAME degradation family but DISJOINT images, so it is a fair generalization
test, not overfitting.

    python run_cpu_generalization.py --iters 5000 --width 32
"""

import argparse
import os
import itertools
import numpy as np
import cv2
import torch

from model import RestoreNet, count_params
from dataset import SyntheticTextPairs
from degrade import DegradeConfig, degrade
from train_stage1 import restoration_loss, make_restore_fn
from ocr import get_ocr
from metrics import evaluate_readability

# motion-blur-dominant, learnable in a CPU budget (the dominant real degradation)
CFG = DegradeConfig(
    p_motion_blur=1.0, motion_len=(9, 19),
    p_defocus=0.3, defocus_rad=(2, 4),
    p_lowlight=0.0,
    p_noise=0.5, noise_sigma=(3, 10),
    p_jpeg=0.7, jpeg_quality=(40, 70),
)

# held-out label strings the model never sees during training
HELDOUT = [
    "PARACETAMOL 500MG", "TAKE 1 TABLET DAILY", "EXP 08 2027", "BATCH A5R5",
    "NET WT 250G", "BEST BEFORE 12 NOV", "AMOXICILLIN 250MG", "STORE BELOW 25C",
]


def render_label(text, w=360, h=64):
    img = np.full((h, w, 3), 244, np.uint8)
    s = 1.0
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
    if tw > w - 12:
        s *= (w - 12) / tw
    cv2.putText(img, text, (8, int(h / 2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, s, (25, 25, 25), 2, cv2.LINE_AA)
    return img


def build_heldout(seed=4321):
    rng = np.random.default_rng(seed)
    items = []
    for t in HELDOUT:
        sharp = render_label(t)
        deg = degrade(sharp, CFG, seed=int(rng.integers(0, 1 << 31)))
        items.append((deg, sharp, t))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    model = RestoreNet(args.width)
    print(f"R_theta width={args.width} params={count_params(model)/1e6:.2f}M  iters={args.iters}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ds = SyntheticTextPairs(length=args.iters * args.batch, crop=(48, 192), cfg=CFG, seed=0)
    ld = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    model.train()
    run = 0.0
    for it, (deg, sharp) in enumerate(itertools.islice(ld, args.iters), 1):
        opt.zero_grad(set_to_none=True)
        loss = restoration_loss(model(deg), sharp)
        loss.backward(); opt.step()
        run += loss.item()
        if it % 250 == 0:
            print(f"  iter {it:5d}/{args.iters}  loss {run/250:.4f}", flush=True)
            run = 0.0

    os.makedirs(args.out, exist_ok=True)
    ck = os.path.join(args.out, f"r_theta_w{args.width}_cpurun.pth")
    torch.save({"model": model.state_dict(), "width": args.width}, ck)
    print("saved", ck)

    # GENERALIZATION eval on unseen held-out labels
    print("\nEvaluating on UNSEEN held-out labels (frozen OCR)...")
    ocr = get_ocr("easyocr", gpu=False)
    heldout = build_heldout()
    restore = make_restore_fn(model, "cpu")

    raw_preds = [ocr.read(d).text for d, _, _ in heldout]
    res_preds = [ocr.read(restore(d)).text for d, _, _ in heldout]
    gts = [t for _, _, t in heldout]
    raw = evaluate_readability(raw_preds, gts)
    res = evaluate_readability(res_preds, gts)

    print(f"\n  raw (held-out)      : {raw}")
    print(f"  R_theta (held-out)  : {res}")
    print(f"\n  CER {raw.mean_cer:.3f} -> {res.mean_cer:.3f}  "
          f"({'GENERALIZES (+)' if res.mean_cer < raw.mean_cer else 'no gain'} "
          f"by {raw.mean_cer-res.mean_cer:+.3f})   "
          f"word-acc {raw.word_accuracy:.3f} -> {res.word_accuracy:.3f}")

    # save a before/after strip on held-out
    rows = []
    for d, s, _ in heldout[:4]:
        r = restore(d)
        sep = np.full((4, d.shape[1], 3), 128, np.uint8)
        rows.append(np.vstack([d, sep, r]))
    cv2.imwrite(os.path.join("_selftest", "heldout_deg_top_restored_bottom.png"), np.vstack(rows))
    print("saved held-out before/after strip to _selftest/")


if __name__ == "__main__":
    main()
