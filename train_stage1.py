"""
Stage 1 — supervised pre-training of R_theta on synthetic (degraded, sharp) pairs.

Clean targets exist here because the data is synthetic, so this is ordinary
supervised restoration: minimise pixel L1 + an edge (Sobel) term that emphasises
text strokes. The result is a strong initialisation for Stage 2's label-free
adaptation on real VizWiz (where no clean target exists).

Device-agnostic: uses CUDA + AMP when available (your RTX 3050), plain CPU
otherwise. The built-in eval measures the only thing that matters — whether the
trained model lowers a frozen OCR's CER versus the raw degraded image.

Smoke test (CPU, proves the loop end-to-end, ~minutes):
    python train_stage1.py --smoke
Full run (GPU):
    python train_stage1.py --width 48 --iters 60000 --batch 32 --crop 64 256
"""

import argparse
import os
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from model import RestoreNet, pad_to_multiple, count_params
from dataset import SyntheticTextPairs


# ── losses ────────────────────────────────────────────────────────────────────

_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
_SOBEL_Y = _SOBEL_X.t().contiguous()


def edge_loss(pred, target):
    """L1 between Sobel-gradient magnitudes — pushes the net to reconstruct sharp
    text strokes rather than a blurry low-frequency average."""
    k = torch.stack([_SOBEL_X, _SOBEL_Y]).unsqueeze(1).to(pred.device)  # (2,1,3,3)
    g = pred.mean(1, keepdim=True)       # luminance
    t = target.mean(1, keepdim=True)
    gp = F.conv2d(g, k, padding=1)
    gt = F.conv2d(t, k, padding=1)
    return F.l1_loss(gp, gt)


def restoration_loss(pred, target, lambda_edge=0.5):
    return F.l1_loss(pred, target) + lambda_edge * edge_loss(pred, target)


# ── bridge: numpy BGR image <-> model tensor (for the OCR eval harness) ────────

def make_restore_fn(model, device):
    model.eval()

    def fn(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        x, (h, w) = pad_to_multiple(x, 4)
        with torch.no_grad():
            y = model(x)[:, :, :h, :w]
        out = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return fn


# ── training ──────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print(f"device={device}  AMP={use_amp}")

    model = RestoreNet(args.width).to(device)
    print(f"R_theta width={args.width}  params={count_params(model)/1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ds = SyntheticTextPairs(length=args.iters * args.batch,
                            crop=(args.crop[0], args.crop[1]), seed=0)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=use_amp, drop_last=True)

    model.train()
    running = 0.0
    for it, (deg, sharp) in enumerate(itertools.islice(loader, args.iters), 1):
        deg, sharp = deg.to(device), sharp.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(deg)
            loss = restoration_loss(pred, sharp, args.lambda_edge)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running += loss.item()
        if it % args.log_every == 0:
            print(f"  iter {it:6d}/{args.iters}  loss {running/args.log_every:.4f}")
            running = 0.0

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, f"r_theta_w{args.width}_stage1.pth")
    torch.save({"model": model.state_dict(), "width": args.width}, ckpt)
    print("saved", ckpt)
    return model, device


def evaluate(model, device, n=6):
    """The verification that matters: does R_theta lower OCR CER vs raw?"""
    from evaluate import build_synthetic_evalset, evaluate_restore, identity
    from ocr import get_ocr
    print("\nLoading frozen OCR for readability check...")
    ocr = get_ocr("easyocr", gpu=(device == "cuda"))
    evalset = build_synthetic_evalset(n=n, seed=999)   # unseen seed
    restore = make_restore_fn(model, device)
    print("Readability (lower CER / higher word-acc = better):")
    raw, _  = evaluate_restore(evalset, identity, ocr, "raw")
    ours, _ = evaluate_restore(evalset, restore, ocr, "R_theta")
    dcer = raw.mean_cer - ours.mean_cer
    print(f"\n  CER {raw.mean_cer:.3f} -> {ours.mean_cer:.3f}  "
          f"({'improved' if dcer > 0 else 'WORSE'} by {abs(dcer):.3f})")
    return raw, ours


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--iters", type=int, default=60000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--crop", type=int, nargs=2, default=[64, 256])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_edge", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_n", type=int, default=6)
    p.add_argument("--no_eval", action="store_true")
    p.add_argument("--out", type=str, default="checkpoints")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU config to verify the loop end-to-end")
    args = p.parse_args()

    if args.smoke:
        args.width, args.iters, args.batch, args.crop = 16, 400, 8, [48, 192]
        args.log_every, args.eval_n = 50, 6
        print("SMOKE config:", vars(args))

    model, device = train(args)
    if not args.no_eval:
        evaluate(model, device, n=args.eval_n)


if __name__ == "__main__":
    main()
