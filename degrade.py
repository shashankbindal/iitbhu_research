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
    cfg = cfg or DegradeConfig()
    rng = _rng(seed)
    x = img_bgr.astype(np.float32)

    # randomise the order of blur operations so the model doesn't learn a fixed chain
    ops = []
    if rng.random() < cfg.p_motion_blur:
        length = rng.uniform(*cfg.motion_len)
        angle = rng.uniform(0, 180)
        ops.append(("filter", _motion_blur_kernel(length, angle)))
    if rng.random() < cfg.p_defocus:
        ops.append(("filter", _disk_kernel(rng.uniform(*cfg.defocus_rad))))
    rng.shuffle(ops)
    for _, k in ops:
        x = cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT)

    # low light: gamma-darken (then the recogniser must cope with crushed shadows)
    if rng.random() < cfg.p_lowlight:
        gamma = rng.uniform(*cfg.lowlight_gamma)
        x = 255.0 * np.power(np.clip(x / 255.0, 0, 1), gamma)

    # sensor noise
    if rng.random() < cfg.p_noise:
        sigma = rng.uniform(*cfg.noise_sigma)
        x = x + rng.normal(0, sigma, x.shape)

    x = np.clip(x, 0, 255).astype(np.uint8)

    # JPEG compression artefacts (applied last, like a real camera pipeline)
    if rng.random() < cfg.p_jpeg:
        q = int(rng.uniform(*cfg.jpeg_quality))
        ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            x = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return x


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
