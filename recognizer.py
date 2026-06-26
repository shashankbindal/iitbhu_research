"""
Differentiable recognizer — provides the gradient signal that trains R_theta in
Stage 2 WITHOUT any clean ground-truth image.

Two roles, one interface (`Recognizer`):
  * confidence_loss(imgs)        -> scalar : low when the recognizer reads the
                                             (restored) crop decisively. Drives
                                             L_conf. Fully self-supervised.
  * text_nll(imgs, target_texts) -> scalar : negative log-likelihood of the known
                                             text. Drives the weak L_vqa term
                                             (VizWiz answers; no clean image).

Backends:
  * TrOCRRecognizer  — microsoft/trocr-small-printed. The real recognizer for the
                       GPU run. (Heavy: ~5 min to load on CPU, and transformers
                       5.3.0 on this box has a tokenizer bug — so it is NOT
                       smoke-tested here; it is the configured choice for the GPU
                       environment.)
  * TinyCRNN         — a small CTC CRNN used only to verify the Stage-2 gradient
                       plumbing on CPU (random init; no real reading ability).
                       Same interface, so train_stage2.py is backend-agnostic.

Both operate on image crops in [0,1], NCHW.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Character vocabulary for the CTC stand-in (index 0 = CTC blank).
_CHARS = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/.-%"
_C2I = {c: i + 1 for i, c in enumerate(_CHARS)}
VOCAB = len(_CHARS) + 1


def encode_texts(texts):
    """List[str] -> (flat_targets LongTensor, lengths LongTensor) for CTC loss."""
    seqs = [[_C2I.get(ch, _C2I[" "]) for ch in t.upper()] for t in texts]
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    flat = torch.tensor([i for s in seqs for i in s], dtype=torch.long)
    return flat, lengths


class Recognizer:
    """Interface every backend implements."""
    def confidence_loss(self, imgs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def text_nll(self, imgs: torch.Tensor, texts) -> torch.Tensor:
        raise NotImplementedError


# ── lightweight stand-in (CPU machinery test) ─────────────────────────────────

class TinyCRNN(Recognizer, nn.Module):
    """Small CTC CRNN: conv stack collapses height -> 1, leaving a width sequence
    of per-character logits. Frozen (no grad to its own weights) so it only routes
    gradients into the input image, exactly like the real frozen recognizer."""
    def __init__(self, width=32, height=32):
        nn.Module.__init__(self)
        self.height = height
        self.width = width
        c = width
        self.body = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),       # H/2
            nn.Conv2d(c, c * 2, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),   # H/4
            nn.Conv2d(c * 2, c * 4, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, None)),                                  # H->1
        )
        self.head = nn.Conv2d(c * 4, VOCAB, 1)
        for p in self.parameters():
            p.requires_grad_(False)        # frozen recognizer

    def feat_channels(self):
        """Channel count of the conv feature map exposed by conv_features()."""
        return self.width * 4

    def conv_features(self, imgs):
        """Config B — the spatial conv feature map BEFORE the height-collapse pool
        and sequence flattening, used as the FiLM conditioning signal. Returns
        (B, width*4, H/4, W')."""
        x = F.interpolate(imgs, size=(self.height, imgs.shape[-1]),
                          mode="bilinear", align_corners=False)
        feat = x
        for layer in self.body[:-1]:       # all layers except the final AdaptiveAvgPool
            feat = layer(feat)
        return feat

    def _logits(self, imgs):
        # resize to a fixed recognition height, keep width (text line aspect)
        x = F.interpolate(imgs, size=(self.height, imgs.shape[-1]),
                          mode="bilinear", align_corners=False)
        f = self.body(x)                   # (B, C, 1, W')
        logits = self.head(f).squeeze(2)   # (B, VOCAB, W')
        return logits.permute(2, 0, 1)     # (T=W', B, VOCAB) for CTC

    def confidence_loss(self, imgs):
        logp = F.log_softmax(self._logits(imgs), dim=-1)
        p = logp.exp()
        entropy = -(p * logp).sum(-1).mean()     # low entropy = confident read
        return entropy

    def text_nll(self, imgs, texts):
        logits = self._logits(imgs)                       # (T,B,V)
        logp = F.log_softmax(logits, dim=-1)
        T, B, _ = logp.shape
        targets, tgt_len = encode_texts(texts)
        in_len = torch.full((B,), T, dtype=torch.long)
        targets, tgt_len, in_len = targets.to(imgs.device), tgt_len.to(imgs.device), in_len.to(imgs.device)
        return F.ctc_loss(logp, targets, in_len, tgt_len, blank=0, zero_infinity=True)


# ── real recognizer for the GPU run ───────────────────────────────────────────

class TrOCRRecognizer(Recognizer):
    """microsoft/trocr-small-printed. Use on the GPU machine. Loads lazily.

    TrOCR's processor normalises to 384x384 with fixed mean/std; we replicate that
    in torch so gradients stay connected from R_theta's output through to the loss.
    """
    IMG = 384
    MEAN = (0.5, 0.5, 0.5)
    STD = (0.5, 0.5, 0.5)

    def __init__(self, name="microsoft/trocr-small-printed", device="cuda"):
        from transformers import VisionEncoderDecoderModel, AutoTokenizer
        self.model = VisionEncoderDecoderModel.from_pretrained(name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # use_fast=False avoids the tiktoken conversion bug seen on some versions
        self.tok = AutoTokenizer.from_pretrained(name, use_fast=False)
        self.device = device
        self.start_id = self.model.config.decoder.decoder_start_token_id
        self._mean = torch.tensor(self.MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(self.STD, device=device).view(1, 3, 1, 1)

    def _prep(self, imgs):
        x = F.interpolate(imgs, size=(self.IMG, self.IMG), mode="bilinear", align_corners=False)
        return (x - self._mean) / self._std

    def confidence_loss(self, imgs, steps=16):
        """Entropy of the decoder distribution under its own greedy unroll."""
        px = self._prep(imgs)
        enc = self.model.get_encoder()(pixel_values=px).last_hidden_state
        ids = torch.full((imgs.size(0), 1), self.start_id, dtype=torch.long, device=self.device)
        total = 0.0
        for _ in range(steps):
            out = self.model.decoder(input_ids=ids, encoder_hidden_states=enc)
            logits = out.logits[:, -1]                    # (B,V)
            logp = F.log_softmax(logits, -1)
            total = total + -(logp.exp() * logp).sum(-1).mean()
            ids = torch.cat([ids, logp.argmax(-1, keepdim=True)], 1)
        return total / steps

    def text_nll(self, imgs, texts):
        px = self._prep(imgs)
        labels = self.tok(list(texts), return_tensors="pt", padding=True).input_ids.to(self.device)
        return self.model(pixel_values=px, labels=labels).loss


def get_recognizer(backend="tinycrnn", **kwargs) -> Recognizer:
    if backend == "tinycrnn":
        return TinyCRNN(**kwargs)
    if backend == "trocr":
        return TrOCRRecognizer(**kwargs)
    raise ValueError(f"unknown recognizer backend '{backend}'")


if __name__ == "__main__":
    # machinery test: does the stand-in route gradients into the image?
    rec = get_recognizer("tinycrnn")
    img = torch.rand(2, 3, 48, 192, requires_grad=True)
    lc = rec.confidence_loss(img)
    lc.backward()
    print(f"confidence_loss={lc.item():.3f}  grad-to-image norm={img.grad.norm():.4f}")

    img2 = torch.rand(2, 3, 48, 192, requires_grad=True)
    lv = rec.text_nll(img2, ["PARACETAMOL 500MG", "BATCH A5R5"])
    lv.backward()
    print(f"text_nll       ={lv.item():.3f}  grad-to-image norm={img2.grad.norm():.4f}")
    assert img.grad.norm() > 0 and img2.grad.norm() > 0
    print("OK: both Stage-2 recognizer losses route gradients into the image.")
