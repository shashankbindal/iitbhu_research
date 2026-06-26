"""
Config B — Recognition-Guided FiLM Modulation.

FiLM (Feature-wise Linear Modulation): condition the restoration decoder on a
frozen recognizer's view of the *input* image. The recognizer's spatial conv
feature map is global-pooled and projected to per-channel (gamma, beta), which
scale/shift one decoder stage:

    decoder_feat = decoder_feat * (1 + gamma) + beta

Because gamma/beta are global (spatially broadcast), the recognizer and decoder
spatial resolutions need not match — no alignment required.

v1: a single injection point (the bottleneck). Zero-initialised so Config B
starts *identical* to Config A (gamma=beta=0 => no modulation); any change is
therefore attributable to learned modulation, keeping the A-vs-B comparison clean.
"""

import torch
import torch.nn as nn


class FiLMModulator(nn.Module):
    def __init__(self, c_rec, c_target):
        super().__init__()
        self.reduce = nn.Conv2d(c_rec, c_rec, 1)
        self.act = nn.GELU()
        self.to_film = nn.Conv2d(c_rec, 2 * c_target, 1)
        # zero-init => gamma=0, beta=0 at start => identity (Config B == A at init)
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)

    def forward(self, rec_feat):
        h = self.act(self.reduce(rec_feat))
        h = h.mean(dim=(2, 3), keepdim=True)          # global pool -> (B, c_rec, 1, 1)
        gamma, beta = self.to_film(h).chunk(2, dim=1) # each (B, c_target, 1, 1)
        return gamma, beta


def apply_film(feat, gamma, beta):
    return feat * (1 + gamma) + beta
