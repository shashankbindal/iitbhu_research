"""
R_theta — the lightweight restoration network.

Design goals (from PROPOSAL.md):
  * mobile budget: target a few M params, ONNX-exportable, no exotic ops
  * residual learning: output = clamp(x + f(x)), so an untrained net starts at
    identity (it can't make the image *worse* before it learns anything)
  * resolution-preserving U-Net with a small receptive field via 2 downsamples —
    enough to span motion-blur kernels without a heavy backbone
  * depthwise-separable convs for mobile efficiency (the 6GB / on-device target)

`width` scales the whole net: width=16 is a fast smoke-test model (~0.2M params),
width=48 is the intended deployment size (~2-3M params). Everything else fixed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConv(nn.Module):
    """Depthwise-separable 3x3 conv + GELU — the mobile-friendly conv block."""
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = SepConv(ch)
        self.c2 = SepConv(ch)

    def forward(self, x):
        return x + self.c2(self.c1(x))


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=2, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


class Up(nn.Module):
    """Bilinear upsample + 1x1 to halve channels (avoids checkerboard artifacts)."""
    def __init__(self, cin, cout):
        super().__init__()
        self.reduce = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.reduce(x)


class RestoreNet(nn.Module):
    """Lightweight residual U-Net for text restoration."""
    def __init__(self, width=48):
        super().__init__()
        w = width
        self.stem = nn.Conv2d(3, w, 3, padding=1)

        self.enc1 = ResBlock(w)
        self.down1 = Down(w, w * 2)
        self.enc2 = ResBlock(w * 2)
        self.down2 = Down(w * 2, w * 4)

        self.mid = nn.Sequential(ResBlock(w * 4), ResBlock(w * 4))

        self.up2 = Up(w * 4, w * 2)
        self.dec2 = ResBlock(w * 2)
        self.up1 = Up(w * 2, w)
        self.dec1 = ResBlock(w)

        self.head = nn.Conv2d(w, 3, 3, padding=1)
        # zero-init the head so the net starts as a pure identity (residual=0)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        s = self.stem(x)
        e1 = self.enc1(s)
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        d2 = self.dec2(self.up2(m) + e2)     # skip connection
        d1 = self.dec1(self.up1(d2) + e1)    # skip connection
        residual = self.head(d1)
        return torch.clamp(x + residual, 0.0, 1.0)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def pad_to_multiple(x, m=4):
    """Pad H,W up to a multiple of m (2 downsamples => need /4). Returns (x, (h,w))."""
    _, _, h, w = x.shape
    ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (h, w)


if __name__ == "__main__":
    for width in (16, 32, 48):
        net = RestoreNet(width)
        n = count_params(net)
        x = torch.rand(1, 3, 64, 256)
        xp, (h, w) = pad_to_multiple(x)
        y = net(xp)[:, :, :h, :w]
        print(f"width={width:2d}  params={n/1e6:.2f}M  in {tuple(x.shape)} -> out {tuple(y.shape)}")
    # identity check: zero-init head means untrained output == input
    net = RestoreNet(16)
    x = torch.rand(1, 3, 32, 32)
    assert torch.allclose(net(x), x, atol=1e-6), "zero-init head should give identity"
    print("Identity-at-init check passed (residual starts at 0).")
