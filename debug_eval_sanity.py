"""
Phase 0 — metric sanity check.

The held-out eval reported word_acc=0.000 for both raw and R_theta, and WER got
worse (0.786 -> 0.856) while CER improved (0.582 -> 0.455). Before trusting any
number from this eval as a baseline, confirm this is real model behaviour and not
a metric bug (case mismatch / whitespace / unicode between prediction and GT).

Reconstructs the EXACT held-out set used by train_stage1.evaluate()
(build_synthetic_evalset(n=8, seed=999)), runs OCR on raw and R_theta-restored,
and prints repr() of every string so mismatches are visible.

    python debug_eval_sanity.py
"""

import torch
from evaluate import build_synthetic_evalset
from ocr import get_ocr
from model import RestoreNet
from train_stage1 import make_restore_fn
from metrics import normalize_text, cer, wer, evaluate_readability

CKPT = "checkpoints/r_theta_w48_stage1.pth"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=device)
    model = RestoreNet(ck.get("width", 48)).to(device)
    model.load_state_dict(ck["model"])
    restore = make_restore_fn(model, device)

    evalset = build_synthetic_evalset(n=8, seed=999)   # the exact held-out set
    ocr = get_ocr("easyocr", gpu=(device == "cuda"))

    print("Per-sample strings (repr shows hidden whitespace/case):\n")
    raw_preds, res_preds, gts = [], [], []
    for i, (deg, _sharp, gt) in enumerate(evalset):
        raw_txt = ocr.read(deg).text
        res_txt = ocr.read(restore(deg)).text
        raw_preds.append(raw_txt); res_preds.append(res_txt); gts.append(gt)
        print(f"[{i}] GT        : {gt!r}")
        print(f"    raw OCR   : {raw_txt!r}   (CER {cer(raw_txt, gt):.2f}, WER {wer(raw_txt, gt):.2f})")
        print(f"    R_theta   : {res_txt!r}   (CER {cer(res_txt, gt):.2f}, WER {wer(res_txt, gt):.2f})")
        # explicit exact-match check after normalization
        print(f"    norm GT == norm R_theta ? {normalize_text(gt) == normalize_text(res_txt)}")
        print()

    raw = evaluate_readability(raw_preds, gts)
    res = evaluate_readability(res_preds, gts)
    print("Aggregate:")
    print(f"  raw     : {raw}")
    print(f"  R_theta : {res}")
    print("\nDiagnosis hints:")
    print("  - word_acc is EXACT full-string match after normalization. For multi-word")
    print("    labels, one wrong char => not exact => 0. At CER 0.45 this is expected.")
    print("  - WER can rise while CER falls if restoration changes word segmentation")
    print("    (OCR splits/merges words differently). Inspect the strings above to confirm.")


if __name__ == "__main__":
    main()
