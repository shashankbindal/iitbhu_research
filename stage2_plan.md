# Stage 2 — label-free recognition-supervised restoration on real VizWiz

**The contribution.** Not the Stage-1 architecture (a plain lightweight U-Net —
not novel, and Phase 5 showed architecture tweaks alone don't move CER). The
contribution is the **training method**: adapt R_θ to *real blind-captured photos*
(VizWiz) with **no clean ground truth**, supervised only by

- **L_conf** — a frozen recognizer (TrOCR) should read the restored crop confidently,
- **L_vqa** — the restored crop's recognized text should match VizWiz's weak crowd answer,
- **L_reblur** — a learned **blur-only** operator must turn the restoration back into
  the observed image (the anchor against hallucination — see below),
- **L_content** — low-frequency consistency (no invented large-scale structure),

optimizing for **downstream readability (OCR/VQA accuracy), not PSNR**, at on-device
scale (0.44 M params). This is a *method*, not a pipeline of frozen models — it clears
the professor's "arranging pretrained models is engineering" bar.

**Named novel mechanism (the ablation's subject): the constrained reblur operator.**
The original `ReblurNet` was an unconstrained residual CNN — it can represent
*sharpening*, so `L_reblur` could be satisfied by a garbage restoration (weak anchor).
The proposed operator predicts a per-image K×K kernel through a **softmax**, so weights
are **non-negative and sum to 1** → it is *provably* a weighted average, i.e. a pure
blur. Then `L_reblur` genuinely tests "there exists a blur that reproduces the
observation from my restoration", which is a real physical constraint. Verified:
kernel min ≥ 0, per-image sum = 1.000. The ablation (constrained vs unconstrained vs
off) is what earns the novelty claim.

**Decision structure (both, staged).** Core = the label-free method with the
constrained reblur (low-risk floor: a result even if everything else fails). Bonus =
re-test recognition-guided FiLM (Config B) with the *real* TrOCR recognizer, the
condition Phase 5 identified as missing. If it helps → architectural novelty on top;
if not → the Stage-1 negative + Stage-2 negative is still a clean, honest ablation.

---

## Phases (each gates the next — stop and report on failure)

### S2-P0 — verify TrOCR on GPU  ⟵ blocker
TrOCR is the real recognizer for L_conf/L_vqa *and* the FiLM re-test. `verify_trocr.py`
must load it on CUDA and show non-zero gradients into the image from both losses.
**Gate:** PASS printed. (Resolved: needed `transformers` installed + `USE_TF=0` to
avoid a broken TF DLL in `ankur_env`.)

### S2-P1 — constrained reblur + label-free loop trains without collapse (synthetic)
Using the synthetic degraded-only stream (no download), train ~1–2k iters with the
constrained reblur + TrOCR. **Gate:** loss decreases, the restored output does *not*
collapse to a trivial (blank/constant) high-confidence image, and L_reblur stays a
meaningful non-zero value. Sanity: constrained-vs-unconstrained produce visibly
different behavior. If it collapses, retune weights before touching real data.

### S2-P2 — acquire VizWiz text subset  ⟵ needs user/download
Download VizWiz-VQA annotations (small JSON) + images (several GB) into `data/`.
`data_vizwiz.py` filters to the ~21% text-reading questions and extracts consensus
answers. **Gate:** `load_annotations` returns a non-trivial text subset (target ≥ a
few thousand samples) with valid image paths.

### S2-P3 — real Stage-2 training on VizWiz
Warm-start from Config A (`r_theta_w48_stage1.pth`), train label-free on the VizWiz
text subset with TrOCR + constrained reblur. **Gate:** training completes; on a held-out
VizWiz slice, restored-image OCR/VQA accuracy ≥ the raw-image baseline (i.e. the method
helps on *real* photos, not just synthetic).

### S2-P4 — ablation + baselines (the paper's core table)
On a frozen real-VizWiz held-out slice + the frozen-61 synthetic set, compare:
raw / RT-Focuser / Stage-1 (Config A) / **Stage-2 full** / Stage-2 w/o reblur /
Stage-2 unconstrained reblur / oracle. **Gate:** filled table + a paragraph stating
whether (a) Stage-2 beats Stage-1 and the deblurrer on real readability, and (b) the
constrained reblur beats unconstrained/off (the novelty claim).

### S2-P5 — bonus: real-recognizer FiLM (Config B) at Stage 2
Re-run the A/B architecture comparison, now with the informative TrOCR recognizer
driving the FiLM. **Gate:** decision paragraph — carry B only if it clears the noise
floor; otherwise document the two-stage negative and keep A.

---

## Out of scope without separate sign-off
Spatially-varying reblur kernels; swapping any Stage-2 checkpoint into the live demo;
multi-recognizer ensembling; training TrOCR itself (it stays frozen throughout).
