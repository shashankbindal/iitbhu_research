"""
Synthetic text-crop pair dataset for Stage-1 supervised pre-training.

Procedurally renders label-like text lines (sharp ground truth), then applies the
blind-capture degradation from degrade.py to produce (degraded, sharp) pairs.
This gives effectively unlimited Stage-1 training data with perfect supervision
and NO external download — the clean targets exist here precisely because the
data is synthetic (the whole point of Stage 1; Stage 2 has no clean targets).

Vocabulary is biased toward the target domain (medicine / product / grocery
labels) so the pre-trained features transfer to real VizWiz text.

Returns tensors in [0,1], CHW, float32.
"""

import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from degrade import degrade, degrade_with_kernel, identity_kernel, DegradeConfig
from nima import severity_to_quality_dist


_WORDS = [
    "TABLET", "TABLETS", "CAPSULE", "CAPSULES", "DAILY", "TWICE", "ONCE",
    "TAKE", "AFTER", "FOOD", "BEFORE", "EXP", "BATCH", "LOT", "NET", "WT",
    "BEST", "USE", "BY", "STORE", "BELOW", "KEEP", "COOL", "DRY", "PLACE",
    "PARACETAMOL", "IBUPROFEN", "AMOXICILLIN", "VITAMIN", "CALCIUM", "ZINC",
    "MILK", "JUICE", "COFFEE", "SUGAR", "SALT", "FLOUR", "RICE", "OIL",
    "INGREDIENTS", "DIRECTIONS", "WARNING", "DOSAGE", "CONTENTS", "ORIGIN",
]
_UNITS = ["mg", "ml", "g", "kg", "IU", "mcg", "L", "%"]


def _rand_token(rng):
    r = rng.random()
    if r < 0.45:
        return rng.choice(_WORDS)
    if r < 0.65:
        return f"{rng.integers(1, 1000)}{rng.choice(_UNITS)}"        # 500mg
    if r < 0.80:
        return f"{rng.integers(1, 31):02d}/{rng.integers(2024, 2030)}"  # date
    if r < 0.92:
        return str(rng.integers(0, 100000))                          # number
    # alphanumeric code like A5R5 / BG1678AU
    n = rng.integers(3, 9)
    chars = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    return "".join(rng.choice(list(chars)) for _ in range(n))


def _bg_fg(rng):
    """Pick (background BGR, foreground BGR) with adequate contrast. ~35% of the
    time use a coloured background (matches real product/sign photos), else the
    grayscale dark-on-light / light-on-dark used before."""
    if rng.random() < 0.35:
        bg = rng.integers(40, 230, size=3)
        lum = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2]    # BGR luminance
        base = int(rng.integers(10, 70)) if lum > 128 else int(rng.integers(190, 250))
        fg = np.clip(base + rng.integers(-25, 25, size=3), 0, 255)
        return [int(c) for c in bg], [int(c) for c in fg]
    dark_on_light = rng.random() < 0.8
    bgv = int(rng.integers(225, 252)) if dark_on_light else int(rng.integers(8, 45))
    fgv = int(rng.integers(10, 55)) if dark_on_light else int(rng.integers(205, 250))
    return [bgv] * 3, [fgv] * 3


def _render_line(rng, w, h):
    """Render one sharp text line into an (h,w,3) BGR uint8 image."""
    bg, fg = _bg_fg(rng)
    img = np.full((h, w, 3), bg, np.uint8)

    ntok = rng.integers(1, 5)
    text = " ".join(_rand_token(rng) for _ in range(ntok))
    scale = float(rng.uniform(0.6, 1.4))      # wider scale range for size diversity
    thick = int(rng.integers(1, 3))
    font = rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
                       cv2.FONT_HERSHEY_COMPLEX])
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    # fit horizontally
    if tw > w - 10:
        scale *= (w - 10) / max(tw, 1)
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = int(rng.integers(5, max(6, w - tw - 4)))
    y = int(h / 2 + th / 2)
    cv2.putText(img, text, (x, y), font, scale, tuple(fg), thick, cv2.LINE_AA)
    return img


def _to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


# Mild degradation for the "near-identity" fraction — teaches the model that a
# nearly-sharp input should be left nearly untouched.
MILD_CFG = DegradeConfig(
    p_motion_blur=0.5, motion_len=(3, 7),
    p_defocus=0.2, defocus_rad=(1, 2),
    p_lowlight=0.0,
    p_noise=0.3, noise_sigma=(2, 5),
    p_jpeg=0.5, jpeg_quality=(60, 88),
)


class SyntheticTextPairs(Dataset):
    """(degraded, sharp) pairs. A `p_clean` fraction are near-identity samples
    (exact-sharp or only mildly degraded) so the model learns to PRESERVE
    already-readable content instead of over-processing it — without this the
    model destroys real, mostly-sharp photos (see RESULTS.md Phase 1)."""

    def __init__(self, length=20000, crop=(64, 256), cfg: DegradeConfig = None,
                 p_clean=0.4, seed=0, return_kernel=False, ksize=25):
        self.length = length
        self.h, self.w = crop
        self.cfg = cfg or DegradeConfig()
        self.p_clean = p_clean
        self.base_seed = seed
        # return_kernel: also yield the TRUE composed blur kernel per sample —
        # only meaningful for synthetic data (Stage-1); used to directly
        # supervise unrolled.py's kernel estimator (see RESULTS.md, unrolled
        # model's under-corrective failure mode).
        self.return_kernel = return_kernel
        self.ksize = ksize

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.base_seed + idx)
        sharp = _render_line(rng, self.w, self.h)
        r = rng.random()
        if not self.return_kernel:
            if r < self.p_clean * 0.5:
                deg = sharp.copy()                                          # exact identity
            elif r < self.p_clean:
                deg = degrade(sharp, MILD_CFG, seed=int(rng.integers(0, 1_000_000)))   # mild
            else:
                deg = degrade(sharp, self.cfg, seed=int(rng.integers(0, 1_000_000)))   # full
            return _to_tensor(deg), _to_tensor(sharp)

        if r < self.p_clean * 0.5:
            deg, kernel, severity = sharp.copy(), identity_kernel(self.ksize), {}
        elif r < self.p_clean:
            deg, kernel, severity = degrade_with_kernel(sharp, MILD_CFG, ksize=self.ksize,
                                                         seed=int(rng.integers(0, 1_000_000)))
        else:
            deg, kernel, severity = degrade_with_kernel(sharp, self.cfg, ksize=self.ksize,
                                                         seed=int(rng.integers(0, 1_000_000)))
        centre = kernel[self.ksize // 2, self.ksize // 2]
        quality = severity_to_quality_dist(kernel_centre_mass=centre,
                                           noise_sev=severity.get("noise", 0.0),
                                           jpeg_sev=severity.get("jpeg", 0.0),
                                           lowlight_sev=severity.get("lowlight", 0.0))
        return (_to_tensor(deg), _to_tensor(sharp), torch.from_numpy(kernel),
                torch.from_numpy(quality))


if __name__ == "__main__":
    import os
    ds = SyntheticTextPairs(length=8, crop=(64, 256), seed=1)
    print(f"dataset len={len(ds)}, sample tensors:", end=" ")
    deg, sharp = ds[0]
    print("deg", tuple(deg.shape), deg.dtype, f"[{deg.min():.2f},{deg.max():.2f}]")

    # dump a strip of pairs to eyeball
    out = os.path.join(os.path.dirname(__file__), "_selftest")
    os.makedirs(out, exist_ok=True)
    rows = []
    for i in range(4):
        d, s = ds[i]
        d = (d.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1]
        s = (s.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1]
        rows.append(np.vstack([s, np.full((4, s.shape[1], 3), 128, np.uint8), d]))
    strip = np.hstack(rows) if rows else None
    cv2.imwrite(os.path.join(out, "pairs_sharp_top_degraded_bottom.png"), strip)
    print("Wrote pair preview to _selftest/pairs_sharp_top_degraded_bottom.png")
