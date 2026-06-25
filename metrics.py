"""
Readability metrics for evaluating restoration by downstream recognition.

The primary success metric of this project is NOT pixel fidelity (PSNR/SSIM) but
whether a frozen OCR can correctly read the restored image. These functions
score predicted transcriptions against ground-truth text.

  CER  — Character Error Rate    = edit_distance(pred, gt) / len(gt)   (lower better)
  WER  — Word Error Rate         = word_edit_distance / num_gt_words   (lower better)
  word_accuracy — exact-match rate over samples                        (higher better)

All pure Python, no heavy deps — runs anywhere.
"""

import re
from dataclasses import dataclass


def normalize_text(s: str, keep_case: bool = False) -> str:
    """Lowercase (optional), collapse whitespace, strip surrounding punctuation.
    Keeps interior alphanumerics and common label characters so e.g. '500mg'
    and '08/2027' survive — important for medicine/product labels."""
    if s is None:
        return ""
    s = s.strip()
    if not keep_case:
        s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings (character level)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1,      # deletion
                           cur[j - 1] + 1,   # insertion
                           prev[j - 1] + cost))  # substitution
        prev = cur
    return prev[-1]


def _token_edit_distance(a_tokens, b_tokens) -> int:
    """Levenshtein distance at the token (word) level."""
    if a_tokens == b_tokens:
        return 0
    prev = list(range(len(b_tokens) + 1))
    for i, ta in enumerate(a_tokens, 1):
        cur = [i]
        for j, tb in enumerate(b_tokens, 1):
            cost = 0 if ta == tb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(pred: str, gt: str, keep_case: bool = False) -> float:
    """Character Error Rate. Returns 0.0 for a perfect read, up to >1 if pred
    is much longer than gt. gt='' returns 0.0 if pred='' else 1.0."""
    p, g = normalize_text(pred, keep_case), normalize_text(gt, keep_case)
    if len(g) == 0:
        return 0.0 if len(p) == 0 else 1.0
    return edit_distance(p, g) / len(g)


def wer(pred: str, gt: str, keep_case: bool = False) -> float:
    """Word Error Rate."""
    p, g = normalize_text(pred, keep_case), normalize_text(gt, keep_case)
    pt, gt_tok = p.split(), g.split()
    if len(gt_tok) == 0:
        return 0.0 if len(pt) == 0 else 1.0
    return _token_edit_distance(pt, gt_tok) / len(gt_tok)


def contains(pred: str, gt: str, keep_case: bool = False) -> bool:
    """Whether the ground-truth string appears inside the prediction — the
    relevant signal for VizWiz weak supervision, where the answer text should
    appear somewhere in what the OCR reads off the whole image."""
    return normalize_text(gt, keep_case) in normalize_text(pred, keep_case)


@dataclass
class ReadabilityReport:
    n: int
    mean_cer: float
    mean_wer: float
    word_accuracy: float     # fraction of exact (normalized) matches
    contains_rate: float     # fraction where gt is substring of pred

    def __str__(self):
        return (f"n={self.n}  CER={self.mean_cer:.3f}  WER={self.mean_wer:.3f}  "
                f"word_acc={self.word_accuracy:.3f}  contains={self.contains_rate:.3f}")


def evaluate_readability(preds, gts, keep_case: bool = False) -> ReadabilityReport:
    """Aggregate metrics over a dataset of (pred, gt) transcriptions."""
    assert len(preds) == len(gts), "preds and gts must align"
    n = len(preds)
    if n == 0:
        return ReadabilityReport(0, 0.0, 0.0, 0.0, 0.0)
    cers = [cer(p, g, keep_case) for p, g in zip(preds, gts)]
    wers = [wer(p, g, keep_case) for p, g in zip(preds, gts)]
    exact = [normalize_text(p, keep_case) == normalize_text(g, keep_case)
             for p, g in zip(preds, gts)]
    cont = [contains(p, g, keep_case) for p, g in zip(preds, gts)]
    return ReadabilityReport(
        n=n,
        mean_cer=sum(cers) / n,
        mean_wer=sum(wers) / n,
        word_accuracy=sum(exact) / n,
        contains_rate=sum(cont) / n,
    )


if __name__ == "__main__":
    # self-test against hand-computed expectations
    assert edit_distance("kitten", "sitting") == 3
    assert cer("PARACETAMOL", "PARACETAMOL") == 0.0
    assert abs(cer("PARACETAMOL", "PARACETMOL") - (1 / 10)) < 1e-9   # one deletion vs 10-char gt
    assert wer("take 1 tablet", "take 2 tablet") == 1 / 3
    assert contains("exp 08/2027 batch a5r5", "a5r5")
    assert not contains("paracetamol", "ibuprofen")

    # a realistic before/after: raw OCR garbles, restored OCR is clean
    raw_preds      = ["paracetan0l 50Omg", "toke 1 toblet twlce dolly"]
    restored_preds = ["paracetamol 500mg", "take 1 tablet twice daily"]
    truth          = ["paracetamol 500mg", "take 1 tablet twice daily"]
    print("raw     :", evaluate_readability(raw_preds, truth))
    print("restored:", evaluate_readability(restored_preds, truth))
    print("\nAll metric self-tests passed.")
