"""
Stage-2 label-free losses — train R_theta on real degraded images with NO clean
ground-truth target. This is the core novelty (see PROPOSAL.md sec 4.2).

Four terms:
  L_conf    : frozen recognizer should read the restored crop confidently
              (self-supervised; via recognizer.confidence_loss)
  L_vqa     : restored crop's recognized text should match the VizWiz answer
              (weak text label, no clean image; via recognizer.text_nll)
  L_reblur  : a small learned reblur of the restored image should reproduce the
              degraded input (after Nah et al. 2021) — anchors output to the real
              scene and blocks hallucination
  L_content : low-frequency consistency between restored and degraded — the net
              may sharpen but must not invent new large-scale structure

L_reblur + L_content are the degeneracy guards: without them L_conf can collapse
to a trivial high-confidence output (e.g. blank). They provide dense, recogniser-
independent gradients, so training is stable even when the confidence signal is weak.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReblurNet(nn.Module):
    """Small CNN that re-applies blur. Trained jointly with R_theta; only used by
    the L_reblur consistency term, discarded at inference."""
    def __init__(self, ch=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, 3, 3, padding=1),
        )

    def forward(self, restored):
        return torch.clamp(restored + self.net(restored), 0.0, 1.0)


def content_loss(restored, degraded, factor=8):
    """L1 between low-frequency (downsampled) versions — preserves scene structure."""
    r = F.avg_pool2d(restored, factor)
    d = F.avg_pool2d(degraded, factor)
    return F.l1_loss(r, d)


def reblur_loss(restored, degraded, reblur_net):
    """Re-blurring the restored image should reproduce the degraded observation."""
    return F.l1_loss(reblur_net(restored), degraded)


def stage2_loss(restored, degraded, recognizer, reblur_net,
                texts=None, crops=None,
                w_conf=1.0, w_vqa=1.0, w_reblur=1.0, w_content=1.0):
    """Assemble the full label-free objective. `crops` are the text-region crops
    fed to the recognizer (default: the whole restored image). Returns
    (total_loss, components_dict)."""
    if crops is None:
        crops = restored

    comp = {}
    total = restored.new_zeros(())

    comp["content"] = content_loss(restored, degraded)
    total = total + w_content * comp["content"]

    comp["reblur"] = reblur_loss(restored, degraded, reblur_net)
    total = total + w_reblur * comp["reblur"]

    if w_conf > 0:
        comp["conf"] = recognizer.confidence_loss(crops)
        total = total + w_conf * comp["conf"]

    if texts is not None and w_vqa > 0:
        comp["vqa"] = recognizer.text_nll(crops, texts)
        total = total + w_vqa * comp["vqa"]

    return total, {k: float(v.detach()) for k, v in comp.items()}


if __name__ == "__main__":
    # machinery test: full label-free loss computes and gradients reach R_theta,
    # using NO sharp/clean target anywhere.
    from model import RestoreNet
    from recognizer import get_recognizer

    deg = torch.rand(2, 3, 48, 192)            # degraded input (no clean target!)
    R = RestoreNet(16)
    reblur = ReblurNet()
    rec = get_recognizer("tinycrnn")
    opt = torch.optim.Adam(list(R.parameters()) + list(reblur.parameters()), lr=1e-3)

    # re-init R head small so it isn't a frozen identity for the test
    for p in [R.head.weight, R.head.bias]:
        nn.init.normal_(p, std=1e-3)

    print("Label-free training steps (no clean target):")
    for step in range(1, 6):
        restored = R(deg)
        total, comp = stage2_loss(restored, deg, rec, reblur,
                                  texts=["PARACETAMOL 500MG", "BATCH A5R5"])
        opt.zero_grad(); total.backward(); opt.step()
        gnorm = sum(p.grad.norm().item() for p in R.parameters() if p.grad is not None)
        print(f"  step {step}: total={total.item():.4f}  {comp}  R_theta grad={gnorm:.4f}")
    assert gnorm > 0, "no gradient reached R_theta"
    print("OK: full Stage-2 objective trains R_theta with NO clean ground truth.")
