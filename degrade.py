"""
Synthetic degradation generator for blind-capture conditions.

Produces (degraded, sharp) training pairs from sharp source images for Stage-1
supervised pre-training of the restoration network R_theta. The degradation
pipeline is deliberately modelled on the failure modes that dominate real
VizWiz photos (Gurari et al., CVPR 2018): hand-tremor motion blur, defocus,
low light, sensor noise, and JPEG compression — applied in random order and
severity, following the high-order degradation idea from Real-ESRGAN
(Wang et al., 2021) but tuned for text legibility rather than natural images.

This module is pure NumPy/OpenCV and runs on CPU — used both to build the
Stage-1 dataset and, at evaluation time, to create controlled degraded test
sets where we DO know the ground-truth text.

Usage:
    from degrade import degrade, DegradeConfig
    deg = degrade(sharp_bgr)                      # random severity
    deg = degrade(sharp_bgr, seed=123)            # reproducible
"""

from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class DegradeConfig:
    # probability each degradation is applied in a given sample
    p_motion_blur: float = 0.7
    p_defocus:     float = 0.5
    p_lowlight:    float = 0.5
    p_noise:       float = 0.6
    p_jpeg:        float = 0.8

    # severity ranges (sampled uniformly)
    motion_len:    tuple = (7, 25)     # motion-blur kernel length in px
    defocus_rad:   tuple = (2, 6)      # defocus disk radius in px
    lowlight_gamma: tuple = (1.3, 2.6) # >1 darkens
    noise_sigma:   tuple = (4, 18)     # gaussian noise std (0-255 scale)
    jpeg_quality:  tuple = (25, 70)    # lower = worse


def _rng(seed):
    return np.random.default_rng(seed)


def _motion_blur_kernel(length, angle_deg):
    """A line kernel of given length and angle — models directional hand shake."""
    length = max(3, int(length) | 1)          # force odd, >=3
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    s = k.sum()
    return k / s if s > 0 else k


def _disk_kernel(radius):
    """A filled-disk kernel — models defocus / out-of-focus blur."""
    radius = max(1, int(radius))
    size = radius * 2 + 1
    yy, xx = np.ogrid[:size, :size]
    disk = ((xx - radius) ** 2 + (yy - radius) ** 2) <= radius ** 2
    disk = disk.astype(np.float32)
    return disk / disk.sum()


def _norm01(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0)) if hi > lo else 0.0


def _degrade_core(img_bgr, cfg, rng, want_kernel):
    """Shared implementation. Returns (degraded, kernel_or_None, severity).
    `kernel` is the single linear blur kernel equivalent to whichever of
    motion-blur/defocus were applied this call (composed via 2D convolution —
    convolution is associative, so this exactly reproduces the effect of
    applying them in sequence). `severity` is a dict of {"noise","jpeg",
    "lowlight"} each in [0,1] (0 = not applied, 1 = worst of the configured
    range) — the non-blur degradations, which the kernel can't represent but
    which still matter for how degraded the image looks overall (used by
    nima.py's severity_to_quality_dist). Both extras are only computed when
    want_kernel=True (they're for the config-u kernel/quality-supervised
    training path; plain degrade() skips this work)."""
    x = img_bgr.astype(np.float32)
    severity = {"noise": 0.0, "jpeg": 0.0, "lowlight": 0.0}

    # randomise the order of blur operations so the model doesn't learn a fixed chain
    kernels = []
    if rng.random() < cfg.p_motion_blur:
        length = rng.uniform(*cfg.motion_len)
        angle = rng.uniform(0, 180)
        kernels.append(_motion_blur_kernel(length, angle))
    if rng.random() < cfg.p_defocus:
        kernels.append(_disk_kernel(rng.uniform(*cfg.defocus_rad)))
    rng.shuffle(kernels)
    for k in kernels:
        x = cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT)
    kernel = _compose_kernel(kernels) if want_kernel else None

    # low light: gamma-darken (then the recogniser must cope with crushed shadows)
    if rng.random() < cfg.p_lowlight:
        gamma = rng.uniform(*cfg.lowlight_gamma)
        x = 255.0 * np.power(np.clip(x / 255.0, 0, 1), gamma)
        if want_kernel:
            severity["lowlight"] = _norm01(gamma, *cfg.lowlight_gamma)

    # sensor noise
    if rng.random() < cfg.p_noise:
        sigma = rng.uniform(*cfg.noise_sigma)
        x = x + rng.normal(0, sigma, x.shape)
        if want_kernel:
            severity["noise"] = _norm01(sigma, *cfg.noise_sigma)

    x = np.clip(x, 0, 255).astype(np.uint8)

    # JPEG compression artefacts (applied last, like a real camera pipeline)
    if rng.random() < cfg.p_jpeg:
        q = int(rng.uniform(*cfg.jpeg_quality))
        ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            x = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if want_kernel:
            lo, hi = cfg.jpeg_quality
            severity["jpeg"] = _norm01(hi - q, 0, hi - lo)   # lower q = worse = higher severity

    return x, kernel, severity


def degrade(img_bgr, cfg: DegradeConfig = None, seed=None):
    """
    Apply a randomly-sampled chain of blind-capture degradations to a sharp image.

    Args:
        img_bgr: uint8 HxWx3 BGR image (the sharp ground truth).
        cfg:     DegradeConfig (defaults if None).
        seed:    int for reproducibility, or None for random.

    Returns:
        degraded uint8 HxWx3 BGR image, same size as input.
    """
    x, _, _ = _degrade_core(img_bgr, cfg or DegradeConfig(), _rng(seed), want_kernel=False)
    return x


KSIZE_DEFAULT = 25


def identity_kernel(ksize=KSIZE_DEFAULT):
    """The kernel meaning 'no blur' — a single centre tap. Used as the target
    for kernel-supervision when no blur op was applied at all."""
    k = np.zeros((ksize, ksize), np.float32)
    k[ksize // 2, ksize // 2] = 1.0
    return k


def _center_crop_pad(k, ksize):
    """Centre-crop or zero-pad a (odd, odd) kernel to (ksize, ksize), renormalising
    to sum=1 (cropping a large combined kernel can clip a small amount of tail
    mass — acceptable for a supervision target, not for the deconvolution itself)."""
    h, w = k.shape
    cy, cx = h // 2, w // 2
    half = ksize // 2
    pad_y, pad_x = max(0, half - cy), max(0, half - cx)
    if pad_y or pad_x:
        k = np.pad(k, ((pad_y, pad_y), (pad_x, pad_x)))
        cy, cx = cy + pad_y, cx + pad_x
    out = k[cy - half:cy + half + 1, cx - half:cx + half + 1].astype(np.float32)
    s = out.sum()
    return out / s if s > 0 else out


def _compose_kernel(kernels, ksize=KSIZE_DEFAULT):
    """Combine the list of blur kernels applied this call into the single
    equivalent kernel (identity if none were applied)."""
    if not kernels:
        return identity_kernel(ksize)
    from scipy.signal import convolve2d
    k = kernels[0]
    for k2 in kernels[1:]:
        k = convolve2d(k, k2, mode="full")
    return _center_crop_pad(k.astype(np.float32), ksize)


def degrade_with_kernel(img_bgr, cfg: DegradeConfig = None, seed=None, ksize=KSIZE_DEFAULT):
    """Like degrade(), but also returns (kernel, severity):
      kernel   — the TRUE equivalent blur kernel (centred, sum=1, (ksize,ksize)).
      severity — dict of {"noise","jpeg","lowlight"} in [0,1], the non-blur
                 degradations the kernel can't represent.
    Both usable only because Stage-1 data is synthetic. `kernel` supervises
    unrolled.py's kernel estimator; `kernel`+`severity` together build the
    NIMA-style quality target via nima.py's severity_to_quality_dist."""
    x, k, severity = _degrade_core(img_bgr, cfg or DegradeConfig(), _rng(seed), want_kernel=True)
    if k.shape[0] != ksize:
        k = _center_crop_pad(k, ksize)
    return x, k, severity


def make_pair(sharp_bgr, cfg: DegradeConfig = None, seed=None):
    """Convenience: return (degraded, sharp) for supervised Stage-1 training."""
    return degrade(sharp_bgr, cfg, seed), sharp_bgr


if __name__ == "__main__":
    # quick self-test: synthesise a sharp text image, degrade it, save both.
    import os
    canvas = np.full((200, 640, 3), 245, np.uint8)
    for i, line in enumerate(["PARACETAMOL 500mg", "Take 1 tablet twice daily", "Exp 08/2027  Batch A5R5"]):
        cv2.putText(canvas, line, (20, 55 + i * 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (20, 20, 20), 2, cv2.LINE_AA)
    out_dir = os.path.join(os.path.dirname(__file__), "_selftest")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "sharp.png"), canvas)
    for s in range(3):
        cv2.imwrite(os.path.join(out_dir, f"degraded_{s}.png"), degrade(canvas, seed=s))
    print("Wrote sharp + 3 degraded samples to", out_dir)
