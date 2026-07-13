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
from nima import emd_loss


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

    is_b = (args.config == "b")
    is_u = (args.config == "u")
    if is_b:
        from model import RestoreNetConfigB
        from recognizer import get_recognizer
        recognizer = get_recognizer("tinycrnn").to(device)
        model = RestoreNetConfigB(args.width, recognizer).to(device)
    elif is_u:
        from unrolled import UnrolledRestorer
        model = UnrolledRestorer().to(device)
        # FFT/complex ops are risky under fp16 autocast — train the unrolled
        # model in fp32 (it is tiny, so memory is not a concern).
        use_amp = False
        scaler_note = "AMP disabled for config u (FFT in fp32)"
        print(scaler_note)
    else:
        model = RestoreNet(args.width).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]   # excludes frozen recognizer
    print(f"Config {args.config.upper()}  width={args.width}  "
          f"trainable params={sum(p.numel() for p in trainable)/1e6:.2f}M  "
          f"total={count_params(model)/1e6:.2f}M")

    start_it = 0
    suffix = {"b": "_configB", "u": "_unrolled"}.get(args.config, "")
    ckpt = os.path.join(args.out, f"r_theta_w{args.width}_stage1{suffix}.pth")
    if args.resume and os.path.exists(ckpt):
        prev = torch.load(ckpt, map_location=device)
        model.load_state_dict(prev["model"])
        start_it = prev.get("iters", 0)
        print(f"resumed from {ckpt} at iter {start_it}")

    if is_u and args.warmstart_quality > 0 and start_it == 0:
        # Diagnostic finding (diag_quality_isolated.py): trained ALONE on the
        # EMD loss, the encoder+quality_head break out of a collapsed
        # "predict the average" plateau around iter ~1800-2000 and reach
        # correlation 0.7-0.9 with true severity by iter ~4000-6000. In the
        # JOINT loop below, the kernel loss (much larger scale, present from
        # iteration 1) dominates the shared encoder's early gradient and the
        # quality signal never gets that chance — both the kernel AND quality
        # heads were observed stuck predicting the average simultaneously.
        # Warm-starting the encoder on the isolated, well-conditioned EMD loss
        # first — before the kernel/pixel losses compete for the same
        # features — is the direct fix this measurement motivates.
        print(f"warm-starting encoder+quality_head via isolated EMD loss "
              f"({args.warmstart_quality} iters)...", flush=True)
        ws_ds = SyntheticTextPairs(length=args.warmstart_quality * args.batch,
                                   crop=(args.crop[0], args.crop[1]), seed=999999,
                                   return_kernel=True, ksize=model.ksize)
        ws_loader = torch.utils.data.DataLoader(
            ws_ds, batch_size=args.batch, shuffle=True,
            num_workers=args.workers, drop_last=True)
        ws_opt = torch.optim.Adam(
            list(model.estimator.enc.parameters()) +
            list(model.estimator.quality_head.parameters()), lr=args.lr)
        ws_running = 0.0
        for ws_it, (deg, _sharp, _kgt, qgt) in enumerate(ws_loader, 1):
            deg, qgt = deg.to(device), qgt.to(device)
            h = model.estimator.enc(deg).flatten(1)
            pred = model.estimator.quality_head(h)
            loss = emd_loss(pred, qgt)
            ws_opt.zero_grad(set_to_none=True)
            loss.backward()
            ws_opt.step()
            ws_running += loss.item()
            if ws_it % args.log_every == 0:
                print(f"  [warmstart] iter {ws_it:5d}/{args.warmstart_quality}  "
                      f"loss_q={ws_running/args.log_every:.4f}", flush=True)
                ws_running = 0.0
        print("warm-start complete", flush=True)

    opt = torch.optim.Adam(trainable, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ds_kwargs = {"return_kernel": True, "ksize": model.ksize} if is_u else {}
    ds = SyntheticTextPairs(length=args.iters * args.batch,
                            crop=(args.crop[0], args.crop[1]), seed=0, **ds_kwargs)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=use_amp, drop_last=True,
        persistent_workers=(args.workers > 0))

    os.makedirs(args.out, exist_ok=True)

    def save(it):
        torch.save({"model": model.state_dict(), "width": args.width, "iters": it}, ckpt)

    remaining = args.iters - start_it
    if remaining <= 0:
        print(f"already at iter {start_it} >= target {args.iters}; nothing to do")
        return model, device

    model.train()
    running, running_k, running_q, last_it = 0.0, 0.0, 0.0, start_it
    # Crash-resilient: Windows DataLoader can hit shared-memory limits late in a
    # run (error 1455). If anything in the loop fails, we still save the
    # already-trained model and proceed to eval rather than losing everything.
    try:
        for rel_it, batch in enumerate(itertools.islice(loader, remaining), 1):
            it = start_it + rel_it
            if is_u:
                deg, sharp, kgt, qgt = batch
                kgt, qgt = kgt.to(device), qgt.to(device)
            else:
                deg, sharp = batch
            deg, sharp = deg.to(device), sharp.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if is_u:
                    # kernel supervision: Stage-1 data is synthetic, so the TRUE
                    # blur kernel is known. Pixel loss alone leaves the estimator
                    # near-identity even on severe blur (ill-conditioned FFT
                    # gradient); this gives it a direct, well-posed signal.
                    pred, kpred, qpred = model(deg, return_kernel=True)
                    loss_pix = restoration_loss(pred, sharp, args.lambda_edge)
                    # sum over kernel bins (not F.l1_loss's default per-BIN mean,
                    # which divides by 625 and makes the gradient ~625x too weak
                    # to move the estimator) -> proper 0-2 scale distributional
                    # distance per sample, averaged over the batch.
                    loss_k = (kpred - kgt.unsqueeze(1)).abs().sum(dim=(1, 2, 3)).mean()
                    # NIMA-style quality supervision (nima.py): a well-conditioned,
                    # directly-supervised "how degraded is this image" signal —
                    # fixes the diagnosed plateau where the kernel/gate alone
                    # couldn't learn to differentiate blur severity per image.
                    loss_q = emd_loss(qpred, qgt)
                    loss = loss_pix + args.w_kernel * loss_k + args.w_quality * loss_q
                else:
                    pred = model(deg)
                    loss = restoration_loss(pred, sharp, args.lambda_edge)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            if is_u:
                running_k += loss_k.item()
                running_q += loss_q.item()
            last_it = it
            if it % args.log_every == 0:
                extra = ""
                if is_b:
                    g, b = model._last_film
                    extra = f"  |gamma|={g:.4f} |beta|={b:.4f}"   # Phase-4 collapse check
                if is_u:
                    # centre-tap mass: 1.0 = predicting identity kernels for
                    # everything (estimator collapse), ~0 = never identity.
                    # Healthy = varies with input; watch it move off both extremes.
                    # quality: mean predicted score, 1=worst 10=best -- watch this
                    # track true severity (drop for a batch skewed toward blur).
                    extra = (f"  k_centre={model.kernel_stats():.3f}"
                              f"  loss_k={running_k/args.log_every:.4f}"
                              f"  loss_q={running_q/args.log_every:.4f}"
                              f"  quality={model.quality_stats():.2f}")
                    running_k, running_q = 0.0, 0.0
                print(f"  iter {it:6d}/{args.iters}  loss {running/args.log_every:.4f}{extra}", flush=True)
                running = 0.0
            if it % args.ckpt_every == 0:
                save(it)
                print(f"  [checkpoint @ {it}]", flush=True)
    except Exception as e:
        print(f"\nWARNING: training loop stopped at iter {last_it}: {type(e).__name__}: {e}", flush=True)

    save(last_it)
    print(f"saved {ckpt} (trained {last_it} iters)", flush=True)
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
    p.add_argument("--config", choices=["a", "b", "u"], default="a",
                   help="a = RestoreNet U-Net; b = +FiLM (shelved); "
                        "u = unrolled MAP restorer (physics-derived, unrolled.py)")
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--iters", type=int, default=60000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--crop", type=int, nargs=2, default=[64, 256])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_edge", type=float, default=0.5)
    p.add_argument("--w_kernel", type=float, default=1.0,
                   help="config u only: weight of the true-blur-kernel L1 supervision loss")
    p.add_argument("--w_quality", type=float, default=1.0,
                   help="config u only: weight of the NIMA-style quality EMD supervision loss")
    p.add_argument("--warmstart_quality", type=int, default=0,
                   help="config u only: iters to pretrain encoder+quality_head on the "
                        "EMD loss alone (no kernel/pixel loss) before joint training — "
                        "fixes gradient competition observed to stall the quality head "
                        "when trained jointly from iteration 1 (see diag_quality_isolated.py)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=2500)
    p.add_argument("--eval_n", type=int, default=6)
    p.add_argument("--no_eval", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="continue from the existing checkpoint's saved iter count")
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
