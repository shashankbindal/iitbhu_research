"""
Phase 2 — frozen held-out evaluation set (>=50 labels) with mean/std CER.

n=8 was a go/no-go signal, not a reliable baseline. This builds a FIXED set of 60
realistic label strings (disjoint from the procedural training generation),
renders + degrades each deterministically (fixed per-index seed), and reports
mean / std / min / max CER for raw vs a given checkpoint.

The label list is frozen in `held_out_eval_v2.json` — this exact set is what
Config A and Config B are compared on (Phase 5). Do not regenerate per run.

    python held_out_eval.py --ckpt checkpoints/r_theta_w48_stage1.pth   # eval a model
    python held_out_eval.py --freeze                                     # (re)write the json
"""

import argparse
import json
import os
import numpy as np
import cv2
import torch

from degrade import degrade, DegradeConfig
from ocr import get_ocr
from model import RestoreNet
from train_stage1 import make_restore_fn
from metrics import cer

JSON = "held_out_eval_v2.json"
RENDER_W, RENDER_H = 384, 64
SEED_BASE = 7000

# 60 realistic label strings — medicine, dosage, product, grocery, nutrition,
# signage. Disjoint from the random word+number combos the training set generates.
HELDOUT_V2 = [
    "PARACETAMOL 650MG", "CETIRIZINE 10MG", "AZITHROMYCIN 500MG", "OMEPRAZOLE 20MG",
    "METFORMIN 500MG", "ASPIRIN 75MG", "AMOXICILLIN 250MG", "IBUPROFEN 400MG",
    "PANTOPRAZOLE 40MG", "VITAMIN D3 60000 IU", "CALCIUM 500MG", "ZINCOVIT SYRUP",
    "TAKE TWICE DAILY", "ONE TABLET AFTER MEALS", "STORE IN COOL DRY PLACE",
    "KEEP AWAY FROM CHILDREN", "DO NOT EXCEED STATED DOSE", "SHAKE WELL BEFORE USE",
    "FOR EXTERNAL USE ONLY", "CONSULT YOUR DOCTOR", "REFRIGERATE AFTER OPENING",
    "NET WT 200G", "NET WEIGHT 1 KG", "BEST BEFORE 06 2027", "MRP RS 145",
    "MFG DEC 2025", "BATCH NO B2024A", "EXPIRY 11 2026", "MFD 03 2026",
    "TOOR DAL 1KG", "BASMATI RICE 5KG", "SUNFLOWER OIL 1L", "TATA SALT 1KG",
    "AMUL BUTTER 100G", "AASHIRVAAD ATTA 10KG", "TATA TEA GOLD 250G", "MAGGI NOODLES 70G",
    "ENERGY 250 KCAL", "PROTEIN 12G", "TOTAL FAT 8G", "SODIUM 200MG",
    "CARBOHYDRATE 30G", "ADDED SUGAR 5G", "DIETARY FIBRE 3G", "CHOLESTEROL 0MG",
    "EMERGENCY EXIT", "PUSH TO OPEN", "NO PARKING", "STAFF ONLY",
    "GSTIN 09ABCDE1234F1Z5", "CUSTOMER CARE 18001234", "WWW EXAMPLE COM",
    "INVOICE NO 4471", "TOTAL AMOUNT 879 00", "CASH RECEIVED 1000",
    "HSN CODE 30049099", "FSSAI 10024051001234", "MADE IN INDIA",
    "MODEL NO KA240Y", "SERIAL 480546DJ", "WARRANTY 2 YEARS",
]


def render_label(text, w=RENDER_W, h=RENDER_H):
    img = np.full((h, w, 3), 244, np.uint8)
    s = 1.0
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
    if tw > w - 12:
        s *= (w - 12) / tw
    cv2.putText(img, text, (8, int(h / 2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, s, (25, 25, 25), 2, cv2.LINE_AA)
    return img


def freeze():
    json.dump({"strings": HELDOUT_V2, "seed_base": SEED_BASE,
               "render": {"w": RENDER_W, "h": RENDER_H}, "degrade": "default"},
              open(JSON, "w"), indent=2)
    print(f"froze {len(HELDOUT_V2)} labels -> {JSON}")


def build():
    """Deterministically render+degrade the frozen label set."""
    spec = json.load(open(JSON))
    cfg = DegradeConfig()
    items = []
    for i, t in enumerate(spec["strings"]):
        sharp = render_label(t, spec["render"]["w"], spec["render"]["h"])
        deg = degrade(sharp, cfg, seed=spec["seed_base"] + i)   # fixed per-index
        items.append((deg, t))
    return items


def stats(vals):
    a = np.array(vals)
    return a.mean(), a.std(), a.min(), a.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/r_theta_w48_stage1.pth")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    if args.freeze or not os.path.exists(JSON):
        freeze()
        if args.freeze:
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    model = RestoreNet(ck.get("width", 48)).to(device)
    model.load_state_dict(ck["model"])
    restore = make_restore_fn(model, device)
    ocr = get_ocr("easyocr", gpu=(device == "cuda"))

    items = build()
    raw_cers, res_cers = [], []
    for deg, gt in items:
        raw_cers.append(cer(ocr.read(deg).text, gt))
        res_cers.append(cer(ocr.read(restore(deg)).text, gt))

    rm, rs, rmin, rmax = stats(raw_cers)
    om, os_, omin, omax = stats(res_cers)
    print(f"\nHeld-out eval v2 ({len(items)} labels)  ckpt={os.path.basename(args.ckpt)}\n")
    print(f"  {'cond':10s}  {'mean':>6s} {'std':>6s} {'min':>6s} {'max':>6s}")
    print("  " + "-" * 40)
    print(f"  {'raw':10s}  {rm:6.3f} {rs:6.3f} {rmin:6.3f} {rmax:6.3f}")
    print(f"  {'R_theta':10s}  {om:6.3f} {os_:6.3f} {omin:6.3f} {omax:6.3f}")
    print(f"\n  mean CER {rm:.3f} -> {om:.3f}  (improved {rm-om:+.3f})")


if __name__ == "__main__":
    main()
