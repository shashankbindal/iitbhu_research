"""
Stage 2 — label-free adaptation of R_theta on real degraded images.

Warm-starts from the Stage-1 checkpoint, then fine-tunes with the label-free
objective (losses_stage2.stage2_loss): NO clean ground-truth image is ever used.
A small ReblurNet is trained jointly (used only by L_reblur, discarded after).

Data:
  * --vizwiz <dir> <ann.json> : the real target — VizWiz text-subset images +
    consensus answers (weak L_vqa labels). The intended training source.
  * default (smoke / no vizwiz): a synthetic degraded-only stream (degraded image
    + the known rendered text as the L_vqa label) that uses NO sharp target — it
    exercises the exact same label-free path for verification without a download.

Recognizer:
  * --recognizer trocr     : microsoft/trocr-small-printed (GPU run; the real signal)
  * --recognizer tinycrnn  : fast random stand-in (CPU machinery check only)

Smoke (CPU, verifies the loop with no clean target, ~seconds):
    python train_stage2.py --smoke
GPU run:
    python train_stage2.py --recognizer trocr --init checkpoints/r_theta_w48_stage1.pth \
        --width 48 --iters 20000 --batch 16
"""

import argparse
import os
import numpy as np
import cv2
import torch

from model import RestoreNet, count_params
from recognizer import get_recognizer
from losses_stage2 import ReblurNet, stage2_loss
from dataset import _render_line, _to_tensor   # reuse the synthetic renderer
from degrade import degrade, DegradeConfig


# ── data streams ──────────────────────────────────────────────────────────────

def synthetic_stream(batch, crop, seed=0):
    """Infinite (degraded_batch, texts) generator — NO sharp target returned.
    Stands in for 'VizWiz image + answer' to verify the label-free loop."""
    h, w = crop
    cfg = DegradeConfig()
    rng = np.random.default_rng(seed)
    while True:
        degs, texts = [], []
        for _ in range(batch):
            r = np.random.default_rng(int(rng.integers(0, 1 << 31)))
            sharp = _render_line(r, w, h)
            # recover the text we rendered (re-render is deterministic per rng) —
            # for the stream we instead read it back via a light OCR-free trick:
            # store text by re-deriving from the same rng is complex, so render
            # with a known string here:
            txt = _STREAM_TEXTS[int(rng.integers(0, len(_STREAM_TEXTS)))]
            sharp = _render_known(txt, w, h)
            degs.append(_to_tensor(degrade(sharp, cfg, seed=int(rng.integers(0, 1 << 31)))))
            texts.append(txt)
        yield torch.stack(degs), texts


_STREAM_TEXTS = ["PARACETAMOL 500MG", "TAKE 1 TABLET DAILY", "EXP 08 2027",
                 "BATCH A5R5", "NET WT 250G", "BEST BEFORE 12 NOV"]


def vizwiz_stream(images_dir, ann_path, batch, crop, seed=0):
    """The REAL target: stream (image_batch, answers) from the VizWiz text subset.
    No clean target — the image IS the real degraded photo; the consensus answer
    is the weak L_vqa label. Images are resized to the recognition crop size.
    Plug this in by passing --vizwiz_images/--vizwiz_ann."""
    from data_vizwiz import load_annotations
    samples = load_annotations(ann_path, images_dir, text_only=True, require_answerable=True)
    if not samples:
        raise RuntimeError(f"no text-subset samples found in {ann_path}")
    print(f"VizWiz text subset: {len(samples)} samples")
    h, w = crop
    rng = np.random.default_rng(seed)
    while True:
        idxs = rng.integers(0, len(samples), size=batch)
        imgs, texts = [], []
        for i in idxs:
            s = samples[int(i)]
            bgr = cv2.imread(s.image_path)
            if bgr is None:
                continue
            bgr = cv2.resize(bgr, (w, h))
            imgs.append(_to_tensor(bgr))
            texts.append(s.answer)
        if imgs:
            yield torch.stack(imgs), texts


def _render_known(text, w, h):
    img = np.full((h, w, 3), 244, np.uint8)
    scale = 1.0
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    if tw > w - 12:
        scale *= (w - 12) / tw
    cv2.putText(img, text, (8, int(h / 2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, scale, (25, 25, 25), 2, cv2.LINE_AA)
    return img


# ── training ──────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  recognizer={args.recognizer}")

    R = RestoreNet(args.width).to(device)
    if args.init and os.path.exists(args.init):
        ck = torch.load(args.init, map_location=device)
        R.load_state_dict(ck["model"]); print("warm-started R_theta from", args.init)
    else:
        for p in [R.head.weight, R.head.bias]:        # avoid frozen-identity start
            torch.nn.init.normal_(p, std=1e-3)
        print("no init checkpoint — small-random head init")
    print(f"R_theta width={args.width} params={count_params(R)/1e6:.2f}M")

    reblur = ReblurNet().to(device)
    rec = get_recognizer(args.recognizer, **({"device": device} if args.recognizer == "trocr" else {}))
    if hasattr(rec, "to"):
        rec.to(device)

    opt = torch.optim.Adam(list(R.parameters()) + list(reblur.parameters()), lr=args.lr)
    if args.vizwiz_images and args.vizwiz_ann:
        stream = vizwiz_stream(args.vizwiz_images, args.vizwiz_ann, args.batch, tuple(args.crop))
    else:
        print("no --vizwiz_images/--vizwiz_ann given: using synthetic degraded stream")
        stream = synthetic_stream(args.batch, tuple(args.crop), seed=0)

    R.train()
    for it in range(1, args.iters + 1):
        deg, texts = next(stream)
        deg = deg.to(device)
        restored = R(deg)
        total, comp = stage2_loss(restored, deg, rec, reblur, texts=texts,
                                  w_conf=args.w_conf, w_vqa=args.w_vqa,
                                  w_reblur=args.w_reblur, w_content=args.w_content)
        opt.zero_grad(set_to_none=True); total.backward(); opt.step()
        if it % args.log_every == 0 or it == 1:
            print(f"  iter {it:5d}/{args.iters}  total={total.item():.4f}  {comp}")

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, f"r_theta_w{args.width}_stage2.pth")
    torch.save({"model": R.state_dict(), "width": args.width}, ckpt)
    print("saved", ckpt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recognizer", choices=["trocr", "tinycrnn"], default="trocr")
    p.add_argument("--init", type=str, default="")
    p.add_argument("--vizwiz_images", type=str, default="", help="dir of VizWiz images")
    p.add_argument("--vizwiz_ann", type=str, default="", help="VizWiz annotation .json")
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--crop", type=int, nargs=2, default=[48, 192])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--w_conf", type=float, default=1.0)
    p.add_argument("--w_vqa", type=float, default=1.0)
    p.add_argument("--w_reblur", type=float, default=1.0)
    p.add_argument("--w_content", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--out", type=str, default="checkpoints")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.recognizer, args.width, args.iters = "tinycrnn", 16, 20
        args.batch, args.crop, args.log_every = 4, [48, 192], 5
        print("SMOKE config:", vars(args))

    train(args)


if __name__ == "__main__":
    main()
