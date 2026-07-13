"""
NIMA-style quality/degradation-severity assessment (Talebi & Milanfar, 2018,
"NIMA: Neural Image Assessment"). Predicts a DISTRIBUTION over quality buckets
1..K (not a single scalar) via a softmax head, trained with Earth Mover's
Distance (EMD) loss against a target distribution — the distributional
formulation respects bucket ORDER (predicting bucket 4 for a true 5 is a
smaller error than predicting bucket 1, unlike cross-entropy) and gives
smoother gradients than a one-hot target.

Adapted here for a technical-distortion signal, not aesthetics: NIMA's own
target distributions come from spread human ratings (AVA dataset); ours come
from the KNOWN synthetic degradation severity (blur-kernel spread + noise /
JPEG / low-light severity) — ground truth is exact by construction here,
unlike human-rated aesthetic NIMA.

Purpose: give unrolled.py's DegradationEstimator an EXPLICIT, well-conditioned
"how degraded is this image" signal. Diagnosed failure this fixes: the
gate/kernel-shape had to discover blur severity implicitly from a noisy
pixel+kernel loss and got stuck in an under-differentiated equilibrium (see
RESULTS.md / PROGRESS_RTFOCUSER_TO_UNROLLED.md, "genuine plateau" finding).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NBUCKETS = 10   # quality buckets 1..10 (10 = pristine, 1 = worst)


class QualityHead(nn.Module):
    """Small NIMA-style head: pooled features -> softmax distribution over
    NBUCKETS quality buckets."""
    def __init__(self, c_in, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_in, hidden), nn.GELU(),
            nn.Linear(hidden, NBUCKETS),
        )

    def forward(self, feat):
        return F.softmax(self.net(feat), dim=-1)          # (B, NBUCKETS)

    @staticmethod
    def expected_score(dist):
        """Scalar quality in [1, NBUCKETS]: E[bucket index], 10=best."""
        buckets = torch.arange(1, NBUCKETS + 1, device=dist.device, dtype=dist.dtype)
        return (dist * buckets).sum(-1)


def emd_loss(pred, target, r=2):
    """Squared Earth Mover's Distance (Talebi & Milanfar eq. 2): compares
    CUMULATIVE distributions so the loss respects bucket order."""
    cdf_pred = torch.cumsum(pred, dim=-1)
    cdf_target = torch.cumsum(target, dim=-1)
    return ((cdf_pred - cdf_target).abs() ** r).mean(-1).pow(1.0 / r).mean()


def severity_to_quality_dist(kernel_centre_mass, noise_sev=0.0, jpeg_sev=0.0,
                             lowlight_sev=0.0):
    """Map KNOWN synthetic degradation severity -> a soft target distribution
    over quality buckets (numpy in, numpy out — built once per training
    sample on CPU inside the Dataset, not batched on GPU).

    kernel_centre_mass: true kernel's centre-tap mass in [0,1] (1 = no blur).
    noise_sev / jpeg_sev / lowlight_sev: in [0,1], 0 = not applied/mild,
    1 = worst of the configured range (see degrade.py's severity computation).
    """
    import numpy as np
    severity = ((1 - kernel_centre_mass) * 0.55 + noise_sev * 0.15 +
                jpeg_sev * 0.15 + lowlight_sev * 0.15)
    severity = max(0.0, min(1.0, severity))
    score = 1.0 + (NBUCKETS - 1) * (1.0 - severity)        # severity=0 -> score=10
    buckets = np.arange(1, NBUCKETS + 1, dtype=np.float32)
    dist = np.exp(-0.5 * ((buckets - score) / 1.2) ** 2)
    return (dist / dist.sum()).astype(np.float32)


if __name__ == "__main__":
    import numpy as np

    # 1. severity->distribution sanity: sharp image should peak near bucket 10,
    #    heavily-blurred should peak near bucket 1.
    d_sharp = severity_to_quality_dist(kernel_centre_mass=1.0)
    d_blur = severity_to_quality_dist(kernel_centre_mass=0.02, noise_sev=0.8, jpeg_sev=0.8)
    assert d_sharp.argmax() == NBUCKETS - 1, d_sharp
    assert d_blur.argmax() <= 2, d_blur
    print(f"sharp dist peak={d_sharp.argmax()+1}  blur dist peak={d_blur.argmax()+1}")

    # 2. QualityHead + EMD loss: gradient flows, loss=0 for identical distributions.
    head = QualityHead(c_in=32)
    feat = torch.randn(4, 32)
    pred = head(feat)
    assert torch.allclose(pred.sum(-1), torch.ones(4), atol=1e-5)
    target = torch.from_numpy(np.stack([d_sharp, d_blur, d_sharp, d_blur]))
    loss = emd_loss(pred, target)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    self_loss = emd_loss(target, target)
    assert self_loss.item() < 1e-5, f"EMD(target,target) should be ~0, got {self_loss.item()}"
    print(f"emd_loss(pred,target)={loss.item():.4f}  emd_loss(target,target)={self_loss.item():.2e}")

    # 3. expected_score sanity
    es_sharp = QualityHead.expected_score(torch.from_numpy(d_sharp).unsqueeze(0))
    es_blur = QualityHead.expected_score(torch.from_numpy(d_blur).unsqueeze(0))
    assert es_sharp.item() > es_blur.item()
    print(f"expected_score sharp={es_sharp.item():.2f}  blur={es_blur.item():.2f}")
    print("ALL SELF-TESTS PASSED")
