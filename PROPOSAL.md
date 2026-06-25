
# Label-Free, Recognition-Supervised Lightweight Restoration for Reading Text in Blind-Captured Images

### Research Proposal

**Author:** Shashank — Research Intern, Dept. of CSE
**Supervisor:** Prof. Indradeep Mastan
**Institution:** Indian Institute of Technology (BHU) Varanasi

---

## 1. Problem and Motivation

People who are blind or have low vision (BLV) routinely photograph objects to ask *"what does this say?"* — medicine labels, nutrition facts, mail, product packaging. On the VizWiz benchmark of real photos taken by blind users, **~21% of all questions are about reading text in the image** (Gurari et al., CVPR 2018; Bigham et al.). But these photos are systematically degraded — motion blur from hand tremor, defocus, poor framing, low light — and off-the-shelf OCR fails on them.

The obvious fix is to deblur/restore the image before reading it. **The obstacle is data:** every learning-based restoration method needs *paired* clean/degraded images for supervision —
- general deblurrers (RT-Focuser, NAFNet) train on GoPro/RealBlur pairs;
- recognition-aware text super-resolution (TPGSR, TATT, TextSR, TextDiff) trains on TextZoom's paired low/high-resolution captures.

**No clean ground truth exists for a blind user's photo, and never can** — you cannot retroactively obtain the sharp version of a shaky medicine-label snapshot. Consequently, *no restoration model has been trained directly on the real distribution it must serve.* Existing methods train on clean benchmarks and hope the result transfers to the blind-captured domain — a domain gap the VizWiz authors and follow-up captioning work repeatedly identify as the core failure mode.

## 2. Research Question

> Can we train a **lightweight, mobile-deployable** restoration network **directly on real blind-captured images (VizWiz)**, with **no clean ground-truth images**, so that downstream **text becomes machine-readable** — supervised purely by a *frozen* recognizer and self-consistency rather than pixel fidelity?

## 3. Gap and Contribution (vs. prior art)

| Prior line | Needs paired/clean data? | Mobile? | Trained on real BLV photos? | Optimizes for |
|---|---|---|---|---|
| RT-Focuser, NAFNet (general deblur) | Yes (GoPro/RealBlur) | RT-Focuser yes | No | PSNR/pixel fidelity |
| TPGSR / TATT / TextSR / TextDiff (text SR) | Yes (TextZoom pairs) | No (most heavy) | No | text SR on benchmark |
| TAIR / TeReDiff (text-aware restoration) | Yes (synthetic clean) | No (diffusion) | No | text fidelity |
| **This work** | **No — label-free** | **Yes (tiny)** | **Yes (VizWiz)** | **downstream readability** |

**Claimed contributions:**
1. **The first restoration model trained directly on real blind-captured images** (VizWiz), without any paired or clean ground truth — bridging a domain gap that paired-data methods structurally cannot cross.
2. **A label-free training objective for *readability*** (not pixel fidelity): frozen-OCR confidence + reblur self-consistency + content preservation + weak text supervision from VizWiz VQA answers.
3. **A mobile-efficient restoration network** evaluated by **downstream OCR / text-VQA accuracy**, not PSNR — the metric that actually matters for BLV users.

This directly answers the critique that a general pretrained deblurrer "just rearranged into a pipeline" is not a contribution: here the network is **re-architected and re-trained at the objective level** for a problem no existing method addresses.

## 4. Method

### 4.1 Architecture

- **Restoration network `R_θ`** — a lightweight U-Net derived from RT-Focuser's design, pruned to a mobile budget (target **≤ 3M params**, dynamic input, ONNX-exportable). Input degraded image `x`; output restored `x̂ = R_θ(x)`. This is the **only trainable component**.
- **Frozen recognizer `F`** — an off-the-shelf, pretrained text detector+recognizer used purely as a *supervisor at training* and an *evaluator at test* (candidates: docTR, PARSeq, TrOCR-small). Never updated. At training it provides per-character logits and located text regions; this keeps the trainable model a single network and makes training cheap and stable.

`R_θ` restores the whole image; all readability losses are computed on the text regions `F` locates.

### 4.2 Two-stage training

**Stage 1 — Synthetic supervised pre-training (good init, GT available).**
Render sharp text/document crops (SynthText-style + real document scans) and apply a Real-ESRGAN-style high-order degradation pipeline (motion blur, defocus, sensor noise, JPEG, low-light) to make synthetic blurry→sharp pairs. Train `R_θ` with a standard pixel + perceptual loss. Here clean targets *do* exist, so this is ordinary supervised training and gives a strong initialization.

**Stage 2 — Label-free adaptation to real VizWiz (the novel part, no clean GT).**
Fine-tune `R_θ` on real VizWiz images using only:

- **`L_conf` — recognition-confidence loss.** Encourage `F` to be *more confident* (lower per-character softmax entropy / higher max-prob) reading `x̂` than reading `x`. Self-supervised; needs no labels. Learns to produce images the recognizer reads decisively.
- **`L_reblur` — reblur self-consistency** (after Nah et al., "Clean Images are Hard to Reblur"). A lightweight learned/estimated blur applied to `x̂` should reconstruct `x`. Prevents hallucination and anchors `x̂` to the true scene.
- **`L_content` — content preservation.** Low-frequency consistency between `x̂` and `x` (e.g. downsampled L1) so the network sharpens rather than invents new text.
- **`L_vqa` — weak text supervision (VizWiz subset).** For the ~21% of VizWiz samples whose VQA *answer is the text on the object*, require `F(x̂)` to contain the answer string (CTC/edit-distance loss). This is genuine label-free-of-*images* supervision: we have the *text*, never the clean image.

Total: `L = λ1·L_conf + λ2·L_reblur + λ3·L_content + λ4·L_vqa`.

**Degeneracy guards** (the main technical risk): confidence-maximization alone can collapse to trivial outputs (e.g. a blank image the OCR "confidently" reads as empty). `L_reblur` + `L_content` anchor the output to the real scene; we also clamp the confidence term and monitor for collapse.

## 5. Datasets

- **VizWiz-VQA** (primary, real BLV photos): text-question subset for training/eval; existing text-presence and image-quality flags help filter.
- **TextVQA / TextOCR** (real-world text images with OCR annotations): held-out evaluation of generalization and an optional weak-supervision source.
- **Synthetic** (Stage-1 only): SynthText + document scans through the degradation pipeline.

## 6. Evaluation

**Primary — readability (the metric that matters):**
- Downstream **OCR word accuracy** and **Character Error Rate (CER)** on a held-out VizWiz text set and on TextVQA, comparing recognition on `x` vs `x̂`.
- **End-to-end Text-VQA accuracy** (does restoring the image let a fixed VQA model answer text questions correctly?).

**Efficiency:** parameters, FLOPs, CPU/mobile latency (the model must be deployable).

**Baselines (must beat):**
1. **No restoration** (raw image) — the floor.
2. **RT-Focuser off-the-shelf** — the general deblurrer; shows our task-adapted model beats a strong general one *on text* (directly rebuts the "it's just a general model" critique).
3. **`R_θ` Stage-1 only** (synthetic-supervised, no label-free adaptation) — the key ablation: isolates the value of training on real VizWiz.
4. (Optional) a heavy text-SR method (TATT/TextSR) — show competitive readability at a fraction of the compute.

**Ablations:** each loss term (`L_conf`, `L_reblur`, `L_content`, `L_vqa`) added/removed; synthetic-pretrain vs. from-scratch; recognizer choice.

## 7. Why it is publishable

- A **clear, single, defensible claim** with an obvious success metric (readability up vs. baselines).
- A **genuine gap** (no paired data on the real target domain) that paired-data methods cannot address by construction.
- **Honest, contained risk** — the Stage-1-only baseline is the make-or-break ablation; if label-free adaptation helps, the contribution stands.
- Natural venues: CVPR/ICCV/WACV **workshops** on low-level vision or accessibility, **ASSETS**, **W4A**, or an applied-ML track.

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Confidence loss collapses to trivial output | `L_reblur` + `L_content` anchors; clamp/monitor confidence term |
| Frozen-OCR supervision is noisy on very degraded input | weak `L_vqa` text labels + robust losses; curriculum from mild→severe degradation |
| Label-free adaptation doesn't beat Stage-1-only | this is *the* experiment; if marginal, reposition as "synthetic-degradation design for BLV text" (still a result) |
| VizWiz text-answer subset too small for `L_vqa` | supplement with TextVQA; `L_conf`/`L_reblur` need no labels at all |

## 9. Timeline (one dedicated GPU)

| Weeks | Milestone |
|---|---|
| 1–2 | Data pipeline: VizWiz text subset, synthetic degradation generator, frozen-OCR + eval harness (CER/word-acc/VQA) |
| 3–4 | Stage-1 synthetic pre-training of `R_θ`; baseline numbers (raw, RT-Focuser) |
| 5–7 | Stage-2 label-free adaptation; loss-term ablations |
| 8–9 | Full evaluation, efficiency benchmarking, ablation table |
| 10–12 | Writing, figures, mobile/ONNX demo |

## 10. Relationship to the Existing VisAssist Project

The current browser pipeline (RT-Focuser → YOLOS → SmolVLM → MiDaS) remains a useful **deployment vehicle and qualitative demo**: the model `R_θ` produced here drops in as a stronger, task-specific replacement for the generic Stage-2 deblurrer, and the live demo becomes the system-level showcase of the trained research artifact.

---

### Key references this builds on

Gurari et al., *VizWiz Grand Challenge* (CVPR 2018) · Singh et al., *Towards VQA Models That Can Read / TextVQA* (CVPR 2019) · Wu et al., *RT-Focuser* (2025) · Wang et al., *Real-ESRGAN* (2021) · Nah et al., *Clean Images are Hard to Reblur* (2021) · Ma et al., *TPGSR / TATT* (2021–22) · Min et al., *Text-Aware Image Restoration (TAIR)* (2025).
