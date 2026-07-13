"""
Unrolled MAP restoration network — the physics-derived replacement for the
standard encoder-decoder (professor's critique: encoder-decoders are engineering;
the novelty must live in the model's mathematics).

Observation model of a blind-captured photo (VizWiz):

    y = alpha * (k (*) x) + beta + n

  x : the clean scene            k    : UNKNOWN blur kernel (shake / defocus)
  y : the photo we actually get  alpha,beta : first-order low-light/exposure
  n : sensor noise               (*)  : convolution

Restoration is the MAP problem, NOT a black-box image-to-image mapping:

    x_hat = argmin_x  1/2 || k (*) x - y' ||^2  +  lambda * Phi(x)        (1)
            with y' = (y - beta) / alpha   (illumination-corrected observation)

Half-quadratic splitting (HQS) introduces z ~ x and alternates two steps:

  DATA step — closed form in the Fourier domain, EXACT, zero learned params:

      x_{t+1} = F^{-1}[ ( conj(F k) . F y'  +  mu_t . F z_t )
                        / ( |F k|^2 + mu_t ) ]                            (2)

  PRIOR step — the proximal operator of Phi, the ONLY learned component:

      z_{t+1} = prox_{Phi}( x_{t+1} )  ~  a tiny shared denoiser CNN      (3)

We unroll T stages of (2)-(3) into a network. Every block has a mathematical
meaning: (2) is a Wiener/deconvolution step, (3) is a learned image prior.

The unknown kernel k is predicted per-image by a small estimator whose output
goes through a softmax, then is mixed with a delta kernel by a learned gate:

      k = (1 - s) * delta  +  s * softmax(logits),   s = sigmoid(gate)    (4)

Both terms live on the probability simplex, so k >= 0 and sum(k) = 1 — k is
PROVABLY a weighted average, i.e. a pure blur, by construction. Two guarantees
follow:

  * Identity for sharp inputs: if the estimator outputs k = delta (and
    alpha=1, beta=0), step (2) returns y exactly — a sharp photo passes through
    UNCHANGED by mathematics, not by hoping the training data taught it.
    (This is the constructive fix for the Phase-1 failure where the U-Net
    destroyed sharp real photos.)
  * Identity at initialisation: gate starts at -4 (s ~ 0.02, k ~ delta) and the
    denoiser's last conv is zero-init, so the untrained network ~= identity.

Parameter budget: ~0.2M (shared prox across stages) vs 0.44M for the U-Net
Config A and 5.85M for RT-Focuser — phone-friendly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nima import QualityHead, NBUCKETS


# ── kernel + illumination estimator ──────────────────────────────────────────

class DegradationEstimator(nn.Module):
    """Predicts, per image: a NIMA-style quality distribution, the blur kernel k
    (simplex-constrained, eq. 4), and a first-order illumination correction
    (alpha, beta) with (1, 0) at init.

    The quality head (nima.py) is trained with a direct, well-conditioned EMD
    loss against the KNOWN synthetic degradation severity — unlike the kernel
    shape (a 625-way distribution the model must discover largely from an
    ill-conditioned pixel+kernel loss), "how degraded is this image overall" is
    a single well-supervised scalar signal that trains fast. Its predicted
    quality is fed back in two ways: (1) concatenated as an explicit input
    feature to the kernel/gate/illumination heads, and (2) added as a direct,
    DETERMINISTIC bias into the gate's pre-sigmoid logit (`quality_gate_scale`,
    zero-init so this contributes nothing at init and can't break the
    identity-at-init guarantee) — this gives the gate an immediate,
    correctly-directed per-image "how much to deviate from identity" signal
    instead of relying solely on the gate's own weight escaping its saturated
    -4-bias region via gradient alone (see RESULTS.md / diagnostic showing the
    gate/kernel-shape without this got stuck in an under-differentiated
    plateau, ~0.90-0.94 centre-mass regardless of true blur severity)."""
    def __init__(self, ksize=25, ch=32):
        super().__init__()
        self.ksize = ksize
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, ch, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.quality_head = QualityHead(c_in=ch)
        self.to_kernel = nn.Linear(ch + 1, ksize * ksize)   # +1: quality feature
        self.to_gate = nn.Linear(ch + 1, 1)
        self.to_illum = nn.Linear(ch + 1, 2)
        # zero-init: contributes nothing until learned, so identity-at-init
        # holds regardless of the quality head's (initially inaccurate) output
        self.quality_gate_scale = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.to_kernel.weight); nn.init.zeros_(self.to_kernel.bias)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.constant_(self.to_gate.bias, -4.0)   # s ~ 0.018 => k ~ delta at init
        nn.init.zeros_(self.to_illum.weight); nn.init.zeros_(self.to_illum.bias)

    def forward(self, y):
        b = y.shape[0]
        h = self.enc(y).flatten(1)                                  # (B, ch)

        quality_dist = self.quality_head(h)                         # (B, NBUCKETS)
        quality_score = QualityHead.expected_score(quality_dist)     # (B,) in [1,NBUCKETS]
        quality_norm = (quality_score - 1) / (NBUCKETS - 1)          # (B,) in [0,1], 1=sharp
        h_aug = torch.cat([h, quality_norm.unsqueeze(-1)], dim=-1)   # (B, ch+1)

        spread = F.softmax(self.to_kernel(h_aug), dim=1)             # simplex
        gate_bias = self.quality_gate_scale * (1.0 - quality_norm)   # explicit, deterministic
        gate_raw = self.to_gate(h_aug).squeeze(-1) + gate_bias
        s = torch.sigmoid(gate_raw).unsqueeze(-1)                    # (B, 1)

        delta = torch.zeros_like(spread)
        delta[:, (self.ksize * self.ksize) // 2] = 1.0              # centre tap
        k = (1 - s) * delta + s * spread                            # eq. (4): still simplex
        k = k.view(b, 1, self.ksize, self.ksize)
        a_raw, b_raw = self.to_illum(h_aug).chunk(2, dim=1)
        alpha = 1.0 + 0.5 * torch.tanh(a_raw)                       # in (0.5, 1.5), init 1
        beta = 0.2 * torch.tanh(b_raw)                              # in (-.2, .2), init 0
        return k, alpha.view(b, 1, 1, 1), beta.view(b, 1, 1, 1), quality_dist


# ── the closed-form data step (eq. 2) ────────────────────────────────────────

def psf2otf(k, shape):
    """Zero-pad kernel to image size and circularly shift its centre to (0,0),
    then FFT — the optical transfer function used by eq. (2)."""
    b, _, ks, _ = k.shape
    h, w = shape
    pad = k.new_zeros(b, 1, h, w)
    pad[:, :, :ks, :ks] = k
    pad = torch.roll(pad, shifts=(-(ks // 2), -(ks // 2)), dims=(-2, -1))
    return torch.fft.rfft2(pad)


def data_step(z, y, otf, mu, fy=None, otf2=None):
    """x = argmin ||k(*)x - y||^2 + mu||x - z||^2 — exact solution via FFT.
    Same kernel applied to each colour channel (physically correct).
    fy / otf2 (=|otf|^2) can be precomputed once per forward — they do not
    change across the T stages."""
    if fy is None:
        fy = torch.fft.rfft2(y)
    if otf2 is None:
        otf2 = otf.abs().pow(2)
    fz = torch.fft.rfft2(z)
    num = torch.conj(otf) * fy + mu * fz
    den = otf2 + mu
    return torch.fft.irfft2(num / den, s=y.shape[-2:])


# ── the learned proximal operator (eq. 3) ────────────────────────────────────

class ProxDenoiser(nn.Module):
    """Tiny residual CNN standing in for prox_Phi. Shared across all T stages
    (that is what keeps the whole model ~0.2M params). Zero-init last conv =>
    prox == identity at start."""
    def __init__(self, ch=48):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, 3, 3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return x + self.body(x)


# ── the unrolled network ─────────────────────────────────────────────────────

class UnrolledRestorer(nn.Module):
    """T unrolled HQS iterations of the MAP problem (1). Interface-compatible
    with RestoreNet: input/output are (B,3,H,W) images in [0,1]."""
    def __init__(self, stages=4, ksize=25, prox_ch=48, est_ch=32):
        super().__init__()
        self.stages = stages
        self.ksize = ksize
        self.estimator = DegradationEstimator(ksize=ksize, ch=est_ch)
        self.prox = ProxDenoiser(ch=prox_ch)
        # per-stage penalty weight mu_t > 0 via softplus; init ~ 0.5
        self.mu_raw = nn.Parameter(torch.full((stages,), -0.2))
        self._last_kernel = None                     # for logging / inspection
        self._last_quality = None

    def forward(self, x, return_kernel=False):
        k, alpha, beta, quality_dist = self.estimator(x)
        self._last_kernel = k.detach()
        self._last_quality = quality_dist.detach()
        y = (x - beta) / alpha                       # illumination-corrected observation
        # reflect-pad so the FFT's circular boundary assumption is harmless;
        # half the kernel support (+1) is what wrap-around can actually reach
        p = self.ksize // 2 + 1
        y = F.pad(y, (p, p, p, p), mode="reflect")
        otf = psf2otf(k, y.shape[-2:])
        fy = torch.fft.rfft2(y)                      # constant across stages
        otf2 = otf.abs().pow(2)
        z = y                                        # z_0 = observation
        for t in range(self.stages):
            mu = F.softplus(self.mu_raw[t])
            xt = data_step(z, y, otf, mu, fy=fy, otf2=otf2)  # eq. (2) — exact, 0 params
            z = self.prox(xt)                        # eq. (3) — learned prior
        out = z[:, :, p:-p, p:-p]
        out = torch.clamp(out, 0.0, 1.0)
        if return_kernel:
            # NOT detached — lets kernel/quality supervision losses backprop
            # straight into the estimator, bypassing the ill-conditioned
            # FFT-division gradient path (see RESULTS.md: pixel loss alone
            # left the estimator stuck near-identity even on severe blur).
            return out, k, quality_dist
        return out

    def kernel_stats(self):
        """(mean centre-tap mass, i.e. how close to identity) for logging."""
        if self._last_kernel is None:
            return 0.0
        c = self.ksize // 2
        return float(self._last_kernel[:, 0, c, c].mean())

    def quality_stats(self):
        """Mean predicted quality score (1=worst, NBUCKETS=best) for logging —
        watch this track the true severity as training progresses."""
        if self._last_quality is None:
            return 0.0
        return float(QualityHead.expected_score(self._last_quality).mean())


# ── self-tests ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    net = UnrolledRestorer()
    n = sum(p.numel() for p in net.parameters())
    print(f"params = {n/1e6:.3f}M  (stages={net.stages}, ksize={net.ksize})")

    # 1. shape + odd sizes (no /4 constraint — fully convolutional + FFT)
    x = torch.rand(2, 3, 67, 203)
    y = net(x)
    assert y.shape == x.shape, y.shape
    print(f"shape ok: {tuple(x.shape)} -> {tuple(y.shape)}")

    # 2. near-identity at init (gate -4 => k~delta; prox zero-init)
    d = (y - x).abs().max().item()
    print(f"identity-at-init max|out-in| = {d:.4f}")
    assert d < 0.03, "untrained network should be ~identity"

    # 3. Wiener data-step correctness: blur with a KNOWN kernel, deconvolve with
    #    that same kernel -> must beat the blurry input on MSE vs the original.
    #    Use a STRUCTURED image (blocks = edges + low/mid frequencies, like real
    #    photos). White noise would be a broken test: blur irreversibly destroys
    #    its high-frequency energy, which no deconvolver can recover.
    img = F.interpolate(torch.rand(1, 3, 12, 12), size=(96, 96), mode="nearest")
    kk = torch.zeros(1, 1, 25, 25)
    g = torch.arange(25).float() - 12
    gk = torch.exp(-(g**2) / (2 * 2.0**2))
    kk[0, 0] = gk[:, None] * gk[None, :]
    kk = kk / kk.sum()
    pp = 25
    imgp = F.pad(img, (pp, pp, pp, pp), mode="reflect")
    otf = psf2otf(kk, imgp.shape[-2:])
    blurp = torch.fft.irfft2(otf * torch.fft.rfft2(imgp), s=imgp.shape[-2:])
    deconv = data_step(blurp, blurp, otf, mu=torch.tensor(1e-3))
    mse_blur = F.mse_loss(blurp[..., pp:-pp, pp:-pp], img).item()
    mse_dec = F.mse_loss(deconv[..., pp:-pp, pp:-pp], img).item()
    print(f"wiener sanity: mse(blur)={mse_blur:.5f} -> mse(deconv)={mse_dec:.5f}")
    assert mse_dec < mse_blur * 0.5, "data step must actually deconvolve"

    # 4. kernel is a valid blur: non-negative, sums to 1. quality_dist is a
    #    valid distribution too.
    k, a, b, qd = net.estimator(x)
    assert k.min() >= 0 and torch.allclose(k.sum(dim=(1, 2, 3)), torch.ones(2), atol=1e-5)
    assert torch.allclose(qd.sum(-1), torch.ones(2), atol=1e-5)
    print(f"kernel simplex ok (min={k.min():.4f}, sums={k.sum(dim=(1,2,3)).tolist()}), "
          f"alpha={a.flatten().tolist()}, beta={b.flatten().tolist()}, "
          f"quality_dist sums={qd.sum(-1).tolist()}")

    # 5. quality_gate_scale is zero-init -> at init, quality has NO effect on the
    #    gate regardless of the (initially inaccurate) quality prediction --
    #    identity-at-init must hold independent of the quality head.
    assert net.estimator.quality_gate_scale.item() == 0.0
    print("quality_gate_scale zero-init confirmed (identity-at-init unaffected by quality head)")

    # 6. gradients reach every component from PIXEL loss alone. NB: quality_head
    #    is correctly EXCLUDED here -- at init its only two exits (the zero-init
    #    to_kernel/to_gate/to_illum weights, and zero-init quality_gate_scale)
    #    are themselves zero, so it genuinely gets no pixel-loss gradient yet by
    #    design (this is what keeps identity-at-init exact); it trains via its
    #    own dedicated EMD loss instead (test 7).
    net.zero_grad()
    loss = F.l1_loss(net(x), torch.rand_like(x))
    loss.backward()
    # NB: prox.body.6 is the zero-init head — it gets gradient immediately;
    # layers behind it start with zero grad by design (identity-at-init).
    for name in ["estimator.to_kernel.weight", "estimator.to_gate.bias",
                 "prox.body.6.weight", "mu_raw"]:
        p = dict(net.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no grad: {name}"
    qp = dict(net.named_parameters())["estimator.quality_head.net.0.weight"]
    assert qp.grad is None or qp.grad.abs().sum() == 0, "quality_head should get NO pixel-loss grad at init"
    print("grad flow ok (estimator / gate / prox / mu); quality_head correctly isolated from pixel loss at init")

    # 7. quality-supervision path: return_kernel=True also returns quality_dist,
    #    and EMD loss against a target distribution backprops into the quality head.
    from nima import emd_loss, severity_to_quality_dist
    import numpy as np
    net.zero_grad()
    out, kpred, qpred = net(x, return_kernel=True)
    target = torch.from_numpy(np.stack([severity_to_quality_dist(0.05),
                                        severity_to_quality_dist(0.9)]))
    qloss = emd_loss(qpred, target)
    qloss.backward()
    qp = dict(net.named_parameters())["estimator.quality_head.net.2.weight"]
    assert qp.grad is not None and qp.grad.abs().sum() > 0
    print(f"quality EMD-loss backprop ok (loss={qloss.item():.4f})")
    print("ALL SELF-TESTS PASSED")
